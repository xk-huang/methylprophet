"""
# 1 device, 1 node
torchrun \
--nnodes 1 \
--nproc_per_node=1 \
--rdzv_endpoint localhost:11454 \
--rdzv_backend c10d  \
scripts/tests/test_methylformer_streaming_dataset_ddp.py \
--trainer.devices=1 \
--trainer.accelerator=cpu 

# 3 devices
torchrun \
--nnodes 1 \
--nproc_per_node=3 \
--rdzv_endpoint localhost:12312 \
--rdzv_backend c10d \
scripts/tests/test_methylformer_streaming_dataset_ddp.py \
--trainer.devices=3 \
--trainer.accelerator=cpu
"""

import json
import os
import pprint
from pathlib import Path

import pandas as pd
import torch
from absl import app, flags, logging
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from omegaconf import OmegaConf
from streaming import StreamingDataLoader, StreamingDataset

from src.config import define_flags, print_flags
from src.data.data_preprocessor_predictive import DatasetV2PreprocessorPredictive
from src.data.data_preprocessor_scgpt import DatasetV2PreprocessorSCGPT
from src.data.data_preprocessor_yf import DatasetV2PreprocessorYF


# NOTE xk: Define the flags and files in `config.py` instead of `main.py`
define_flags()
DDP_ENV_VARS = {
    "WORLD_SIZE": None,
    "LOCAL_WORLD_SIZE": None,
    "RANK": None,
    "MASTER_ADDR": None,
    "MASTER_PORT": None,
    "GROUP_RANK": None,
    "LOCAL_RANK": None,
}


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


def main(_):
    FLAGS = flags.FLAGS
    FLAGS.flags_by_module_dict().keys()
    FLAGS.get_flags_for_module("src.config")
    # Get the user defined config, instead of the meta ones from absl.
    FLAGS_DEF = {}
    for flag_values in FLAGS.get_flags_for_module("src.config"):
        # NOTE xk: flag_values.name is the key, flag_values.value is the ConfigDict, which has `to_dict`, `to_yaml` methods.
        FLAGS_DEF[flag_values.name] = flag_values.value
    print_flags(FLAGS, FLAGS_DEF)

    # Get DDP env
    # NOTE xk: get ddp env
    # e.g., {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '54669', 'NODE_RANK': '0', 'LOCAL_RANK': '6', 'WORLD_SIZE': '8'}
    ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
    logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")

    config = OmegaConf.create({k: v.to_dict() for k, v in FLAGS_DEF.items()})

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

    config.train_dataloader.pop("collate_fn")
    config.val_dataloader.pop("collate_fn")
    trainer_data_module = TrainerDataModule(config)
    trainer.fit(model, datamodule=trainer_data_module)


class TrainerDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.data_preprocessor = None

        # NOTE xk: For batch size finder
        # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#batch-size-finder
        self.batch_size = config.train_dataloader.batch_size

        rank = os.getenv("RANK", None)
        world_size = os.getenv("WORLD_SIZE", None)
        if rank is None:
            logging.warning("RANK is not set. Set RANK=0. Be sure that you are not in distributed training.")
            rank = 0
        if world_size is None:
            logging.warning(
                "WORLD_SIZE is not set. Set WORLD_SIZE=1. Be sure that you are not in distributed training."
            )
            world_size = 1
        rank = int(rank)
        world_size = int(world_size)

        # trainer_devices = int(config.trainer.devices)
        # if trainer_devices != world_size:
        #     raise ValueError(
        #         f"config.trainer.devices={trainer_devices} is not equal to world_size={world_size}"
        #         "use torchrun to start the jobs!"
        #     )

        logging.info(f"Split dataset with: rank={rank}, world_size={world_size}")
        self.rank = rank
        self.world_size = world_size

    def setup(self, stage=None):
        train_dataset_path = "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/train"
        val_dataset_path = "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val"

        gene_expr_df_path = "data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/gene_expr.filtered.parquet"

        data_preprocess_config = {
            # NOTE: data_preprocessor_type will be popped in `src/trainer_data_module.py:create_data_preprocess`
            # choices: DatasetV2PreprocessorSCGPT, DatasetV2PreprocessorYF, DatasetV2PreprocessorPredictive
            "data_preprocessor_type": "DatasetV2PreprocessorPredictive",
            "batched": False,
            "num_nbase": 2000,
            "num_gene_expr_bins": 51,
            "zero_gene_expr_filter": False,
            "cpg_nbase_type": "one_hot",  # ["one_hot", "tokenized"]
            "gene_expr_quantization": True,
        }
        train_shuffle_seed = 42
        train_batch_size = 3
        val_batch_size = 2

        data_preprocessor_type = data_preprocess_config.pop("data_preprocessor_type")
        data_preprocessor = create_data_preprocessor(
            data_preprocessor_type,
            gene_expr_df_path,
            **data_preprocess_config,
        )
        self.data_preprocessor = data_preprocessor

        ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
        logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")
        train_dataset = MethylformerStreamingDataset(
            data_preprocessor=data_preprocessor,
            group_idx_name_mapping_path=Path(train_dataset_path) / "group_idx_name_mapping.json",
            local=train_dataset_path,
            batch_size=train_batch_size,
            shuffle_seed=train_shuffle_seed,
            shuffle=False,
            epoch_size=12,
            sampling_method="fixed",
            sampling_granularity=12,
        )
        logging.info(f"Create train dataset: {train_dataset}")
        self.train_dataset = train_dataset

        ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
        logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")
        val_dataset = MethylformerStreamingDataset(
            data_preprocessor=data_preprocessor,
            group_idx_name_mapping_path=Path(val_dataset_path) / "group_idx_name_mapping.json",
            local=val_dataset_path,
            batch_size=val_batch_size,
            shuffle=False,
            epoch_size=13,
            sampling_method="fixed",
            sampling_granularity=13,
        )
        logging.info(f"Create val dataset: {val_dataset}")
        self.val_dataset = val_dataset

    def train_dataloader(self):
        config = self.config
        ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
        logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")
        train_dataloader = StreamingDataLoader(
            self.train_dataset,
            **config.train_dataloader,
            collate_fn=self.data_preprocessor.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        config = self.config
        ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
        logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")
        val_dataloader = StreamingDataLoader(
            self.val_dataset,
            **config.val_dataloader,
            collate_fn=self.data_preprocessor.collate_fn,
        )
        return val_dataloader


def create_data_preprocessor(data_preprocessor_type, gene_expr_df_path, **kwargs):
    gene_expr_df = pd.read_parquet(gene_expr_df_path)
    if data_preprocessor_type == "DatasetV2PreprocessorSCGPT":
        data_preprocessor = DatasetV2PreprocessorSCGPT(gene_expr_df=gene_expr_df, **kwargs)
    elif data_preprocessor_type == "DatasetV2PreprocessorYF":
        data_preprocessor = DatasetV2PreprocessorYF(gene_expr_df=gene_expr_df, **kwargs)
    elif data_preprocessor_type == "DatasetV2PreprocessorPredictive":
        data_preprocessor = DatasetV2PreprocessorPredictive(gene_expr_df=gene_expr_df, **kwargs)
    return data_preprocessor


class MethylformerStreamingDataset(StreamingDataset):
    def __init__(self, *, data_preprocessor=None, group_idx_name_mapping_path=None, **kwargs):
        super().__init__(**kwargs)
        self.data_preprocessor = data_preprocessor

        if group_idx_name_mapping_path is not None:
            logging.info(f"Load group_idx_name_mapping from {group_idx_name_mapping_path}")
            with open(group_idx_name_mapping_path, "r") as f:
                self.group_idx_name_mapping = json.load(f)
        else:
            logging.info("group_idx_name_mapping is None")
            self.group_idx_name_mapping = None

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.data_preprocessor(data)


if __name__ == "__main__":
    app.run(main)
