"""
WANDB_MODE=disabled python scripts/tests/test_model.py \
    --flagfile src/configs/cfg/data/dataloader.cfg \
    --flagfile src/configs/cfg/data/dataset/encode/dataset.cfg \
    --flagfile src/configs/cfg/data/dataset/encode/tokenized_val_dataset.cfg \
    --trainer.val_check_interval=7 \
    --trainer.log_every_n_steps=9 \
    --data.train_dataset.epoch_size=100 \
    --data.val_dataset.epoch_size=100 \
    --data.train_dataset.shuffle=False \
    --data.train_dataloader.shuffle=False \
    --trainer_model.eval_dir='misc/eval' \
    --trainer_model.ckpt_dir='misc/ckpt' \
    --data.train_dataset.local=data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards  \
    --data.val_dataset.local=data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards \
    --trainer.devices=1 \
    --model.sample_gene_mlp_config_dict.dim_in=24337 \
    --model.use_tissue_embedder=True \
    --model.use_chr_embedder=True \
"""

import os
import pprint

import torch
from absl import app, flags, logging
from dotenv import load_dotenv
from lightning import LightningModule, Trainer
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from src.config import define_flags
from src.data.trainer_data_module import TrainerDataModule
from src.models.model_factory import create_model_class, create_model_config_class
from src.trainer_model import TrainerModel


load_dotenv()


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

    # Create model
    # NOTE xk: Llama config check the type of `model_args["rope_scaling"]` to be `dict`.
    # So we convert omegaconf.dictconfig.DictConfig to dict.
    model_args = config.model
    model_class_name = model_args.pop("model_class")
    model_config_class_name = model_args.pop("model_config_class")
    model_class = create_model_class(model_class_name)
    model_config_class = create_model_config_class(model_config_class_name)

    model_args = OmegaConf.to_container(model_args)

    model_config = model_config_class(**model_args)
    model = model_class(model_config)
    # trainer_model = DummyTrainerModel(config.trainer_model, model)
    trainer_model = TrainerModel(config.trainer_model, model)

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

    # we need to specially handle it from the trainer_kwargs
    logging.info(f"Trainer kwargs: {pprint.pformat(trainer_kwargs)}")
    trainer_kwargs.pop("callbacks")
    trainer_kwargs.pop("logger")
    trainer = Trainer(
        **trainer_kwargs,
        callbacks=[pl_callbacks.ModelCheckpoint(filename="{step}")],
        logger=pl_loggers.WandbLogger(),
    )

    config.data.train_dataloader.pop("collate_fn")
    config.data.val_dataloader.pop("collate_fn")
    trainer_data_module = TrainerDataModule(config.data)
    trainer.fit(trainer_model, datamodule=trainer_data_module)


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


if __name__ == "__main__":
    app.run(main)
