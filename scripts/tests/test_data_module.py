"""
python scripts/tests/test_data_module.py \
    --flagfile src/configs/cfg/data/dataloader.cfg \
    --flagfile src/configs/cfg/data/dataset/encode/tokenized_val_dataset.cfg \
    --trainer.val_check_interval=100 \
    --data.train_dataset.epoch_size=100 \
    --data.train_dataset.local=  \
    --data.val_dataset.local=
"""

import os
import pprint

import torch
from absl import app, flags, logging
from lightning import LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from omegaconf import OmegaConf

from src.config import define_flags
from src.data.trainer_data_module import TrainerDataModule


class DummyModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.module = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.module(x)

    def training_step(self, batch, batch_idx):
        print(f"cpg_idx: {batch['cpg_idx']}, sample_idx: {batch['sample_idx']}, group_idx: {batch['group_idx']}")
        return None

    def validation_step(self, batch, batch_idx):
        return None

    def on_train_epoch_start(self):
        local_rank = self.trainer.local_rank
        world_size = self.trainer.world_size
        train_dataset = self.trainer.train_dataloader.dataset
        val_dataset = self.trainer.val_dataloaders.dataset
        ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
        logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")
        logging.info(
            f"[NODE {local_rank}/{world_size}]: len(train_dataset)={len(train_dataset)}, len(val_dataset)={len(val_dataset)}; train_dataset.size={train_dataset.size}, val_dataset.size={val_dataset.size}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0)


DDP_ENV_VARS = {
    "WORLD_SIZE": None,
    "LOCAL_WORLD_SIZE": None,
    "RANK": None,
    "MASTER_ADDR": None,
    "MASTER_PORT": None,
    "GROUP_RANK": None,
    "LOCAL_RANK": None,
}

define_flags()


def main(_):
    FLAGS = flags.FLAGS
    module_names = FLAGS.flags_by_module_dict().keys()  # get the module names
    logging.info(f"Module names: {module_names}")
    module_name = None
    for name in module_names:
        if "config" in name and "src" in name:
            module_name = name
            break
    FLAGS.get_flags_for_module(module_name)

    # Get the user defined config, instead of the meta ones from absl.
    FLAGS_DEF = {}
    for flag_values in FLAGS.get_flags_for_module(module_name):
        # NOTE xk: flag_values.name is the key, flag_values.value is the ConfigDict, which has `to_dict`, `to_yaml` methods.
        FLAGS_DEF[flag_values.name] = flag_values.value

    config = OmegaConf.create({k: v.to_dict() for k, v in FLAGS_DEF.items()})
    print(OmegaConf.to_yaml(config))

    model = DummyModel(config.model)

    callbacks = []

    trainer_kwargs = config.trainer
    trainer_kwargs = OmegaConf.to_container(trainer_kwargs)
    # NOTE xk: Tuning the batch size is currently not supported with distributed strategies
    if not config.main.find_batch_size:
        # NOTE xk: find_unused_parameters is used as some parameters are not used in the forward pass during dev.
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
    trainer_kwargs.update({"callbacks": callbacks, "logger": None, "strategy": strategy})
    # XXX: "overfit_batches" is Union[int, float], which is not supported in ml_collections.
    if trainer_kwargs["overfit_batches"] is None:
        trainer_kwargs["overfit_batches"] = 0.0
    # NOTE xk: training epoch = 1, num_sanity_val_steps = 0
    trainer_kwargs.update(
        {
            "max_epochs": 1,
            "num_sanity_val_steps": 0,
            "max_steps": -1,
            "val_check_interval": None,
            "check_val_every_n_epoch": 1,
        }
    )
    # we need to specially handle it from the trainer_kwargs
    logging.info(f"Trainer kwargs: {pprint.pformat(trainer_kwargs)}")
    trainer = Trainer(**trainer_kwargs)

    config.data.train_dataloader.pop("collate_fn")
    config.data.val_dataloader.pop("collate_fn")
    trainer_data_module = TrainerDataModule(config.data)
    trainer.fit(model, datamodule=trainer_data_module)


if __name__ == "__main__":
    app.run(main)
