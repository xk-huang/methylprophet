"""
Check the data loading process.

1. Make sure all the data is loaded correctly, w/ or w/o multi-processing loading or not.
2. Profile the time it takes to load the data.
3. Test distributed loading is correct.
4. Check the cpg and sample intersection between different splits.
"""

import gc
import json
import os
import os.path as osp
import pprint
from collections import defaultdict
from pathlib import Path
from pprint import pformat

import pandas as pd
import torch
from absl import app, flags, logging
from dotenv import load_dotenv
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from omegaconf import MISSING, OmegaConf

from src.config import define_flags
from src.data.trainer_data_module import TrainerDataModule
from src.models.model_factory import create_model_class, create_model_config_class
from src.trainer_model import TrainerModel
from src.utils import prepare_version_dir


class DummyTainerModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.module = torch.nn.Linear(1, 1)

        self.train_samples = defaultdict(list)
        self.val_samples = defaultdict(list)

        self.eval_save_batch_interval = config.get("eval_save_batch_interval", 100_000)
        self.eval_save_idx = 0
        self.train_save_idx = 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0)

    def forward(self, batch: dict, batch_idx=None):
        if self.trainer.global_step == 0:
            logging.info(f"batch_idx: {batch_idx}")
            logging.info(f"batch: {pformat(batch)}")

        cpg_idx = batch.pop("cpg_idx")
        sample_idx = batch.pop("sample_idx")
        methylation = batch.pop("methylation")
        group_idx = batch.pop("group_idx")
        tokenized_sequence = batch.pop("tokenized_sequence_input_ids")
        if torch.any(methylation.isnan()):
            raise ValueError(
                f"nan in methylation, for cpg_idx: {cpg_idx}, sample_idx: {sample_idx} in batch_idx: {batch_idx}"
            )

        return {
            "cpg_idx": cpg_idx,
            "sample_idx": sample_idx,
            "methylation": methylation,
            "group_idx": group_idx,
            # NOTE xk: we only gather the tensors, so make them the same batch size
            "tokenized_sequence_input_ids_length": torch.ones(methylation.shape[0], dtype=torch.int64)
            * tokenized_sequence.shape[-1],
        }

    def training_step(self, batch, batch_idx):
        output = self.forward(batch, batch_idx)

        for k, v in output.items():
            self.train_samples[k].append(v)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.forward(batch, batch_idx)

        for k, v in output.items():
            self.val_samples[k].append(v)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.eval_save_batch_interval != 0 or batch_idx == 0:
            return

        for k, v in self.train_samples.items():
            self.train_samples[k] = self.check_and_gather_data_points(v)
        if self.trainer.is_global_zero:
            self._save_samples(
                self.train_samples, "train", self.train_save_idx, dataloader=self.trainer.train_dataloader
            )
            self.train_save_idx += 1
        self.trainer.strategy.barrier()
        self.train_samples.clear()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % self.eval_save_batch_interval != 0 or batch_idx == 0:
            return

        for k, v in self.val_samples.items():
            self.val_samples[k] = self.check_and_gather_data_points(v)
        if self.trainer.is_global_zero:
            self._save_samples(self.val_samples, "val", self.eval_save_idx, self.trainer.val_dataloaders)
            self.eval_save_idx += 1
        self.trainer.strategy.barrier()
        self.val_samples.clear()

    def on_train_epoch_start(self):
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is not None:
            prefix = "train"
            eval_dir = Path(self.config.eval_dir)
            output_dir = eval_dir / f"{prefix}-cpg_sample_idx_df"
            dataloader_ckpt_path = output_dir / f"{prefix}_dataloader.json"
            if dataloader_ckpt_path.exists():
                with open(dataloader_ckpt_path, "r") as f:
                    train_dataloader.load_state_dict(json.load(f))
                logging.info(f"Resuming train dataloader from {dataloader_ckpt_path}")
            save_idx_path = output_dir / f"{prefix}_save_idx.json"
            if save_idx_path.exists():
                with open(save_idx_path, "r") as f:
                    self.train_save_idx = json.load(f)["save_idx"]
                logging.info(f"Resuming train save_idx from {save_idx_path}")

    def on_validation_epoch_start(self):
        val_dataloader = self.trainer.val_dataloaders
        if val_dataloader is not None:
            prefix = "val"
            eval_dir = Path(self.config.eval_dir)
            output_dir = eval_dir / f"{prefix}-cpg_sample_idx_df"
            dataloader_ckpt_path = output_dir / f"{prefix}_dataloader.json"
            if dataloader_ckpt_path.exists():
                with open(dataloader_ckpt_path, "r") as f:
                    val_dataloader.load_state_dict(json.load(f))
                logging.info(f"Resuming val dataloader from {dataloader_ckpt_path}")
            save_idx_path = output_dir / f"{prefix}_save_idx.json"
            if save_idx_path.exists():
                with open(save_idx_path, "r") as f:
                    self.eval_save_idx = json.load(f)["save_idx"]
                logging.info(f"Resuming val save_idx from {save_idx_path}")

    def on_train_epoch_end(self) -> None:
        for k, v in self.train_samples.items():
            self.train_samples[k] = self.check_and_gather_data_points(v)

        if self.trainer.is_global_zero:
            self._save_samples(self.train_samples, "train", self.train_save_idx)
            self.train_save_idx += 1

        self.trainer.strategy.barrier()
        self.train_samples.clear()
        self.train_save_idx = 0
        gc.collect()

    def on_validation_epoch_end(self):
        for k, v in self.val_samples.items():
            self.val_samples[k] = self.check_and_gather_data_points(v)
        if self.trainer.is_global_zero:
            self._save_samples(self.val_samples, "val", self.eval_save_idx)
            self.eval_save_idx += 1

        self.trainer.strategy.barrier()
        self.val_samples.clear()
        self.eval_save_idx = 0
        gc.collect()

    def _save_samples(self, samples, prefix, save_idx, dataloader=None):
        df = pd.DataFrame(samples)

        eval_dir = Path(self.config.eval_dir)
        output_dir = eval_dir / f"{prefix}-cpg_sample_idx_df"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{save_idx:06d}.parquet"
        df.to_parquet(output_path)
        logging.info(f"describe: {df.describe()}")
        logging.info(f"Save {prefix} samples to {output_path}")

        if dataloader is not None:
            dataloader_ckpt_path = output_dir / f"{prefix}_dataloader.json"
            with open(dataloader_ckpt_path, "w") as f:
                json.dump(dataloader.state_dict(), f, indent=4)
            save_idx_path = output_dir / f"{prefix}_save_idx.json"
            with open(save_idx_path, "w") as f:
                # NOTE xk: we add 1 to the save_idx to indicate the next save_idx.
                json.dump({"save_idx": save_idx + 1}, f, indent=4)

    def check_and_gather_data_points(self, data_point_ls):
        num_batches = len(data_point_ls)

        batch_shape_before_gather = data_point_ls[0].shape
        data_point_ls = self.all_gather(data_point_ls)
        batch_shape_after_gather = data_point_ls[0].shape
        if num_batches != len(data_point_ls):
            raise ValueError("data_point_ls should have the same length after all_gather")

        strategy_type = type(self.trainer.strategy)
        if isinstance(self.trainer.strategy, SingleDeviceStrategy):
            logging.info(f"Using {strategy_type}, no need to flatten. Assuming (batch_size, ...)")
        else:
            logging.info(f"Using {strategy_type}, flatten the outputs. Assuming (world_size, batch_size, ...)")
            data_point_ls = [i.flatten(0, 1) for i in data_point_ls]
        data_points = torch.cat(data_point_ls, 0)

        logging.info(
            f"num_batches: {num_batches}, data_point_ls[0] shape: {batch_shape_before_gather} -> {batch_shape_after_gather}, data_point_ls shape: {data_points.shape}"
        )

        return data_points


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
    for flag_values in FLAGS.get_flags_for_module("src.config"):
        # NOTE xk: flag_values.name is the key, flag_values.value is the ConfigDict, which has `to_dict`, `to_yaml` methods.
        FLAGS_DEF[flag_values.name] = flag_values.value

    # Get DDP env
    # NOTE xk: get ddp env
    # e.g., {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '54669', 'NODE_RANK': '0', 'LOCAL_RANK': '6', 'WORLD_SIZE': '8'}
    ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
    logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")

    # Create config
    config = OmegaConf.create({k: v.to_dict() for k, v in FLAGS_DEF.items()})
    update_config_by_dotlist = None
    if config.main.update_config_by_dotlist is not None:
        update_config_by_dotlist = config.main.update_config_by_dotlist

    # Load configs either for resume training or loading trained model
    if config.main.weight_path is not None and config.main.ckpt_path is not None:
        logging.fatal("ckpt_path and weight_path are mutually exclusive")

    # NOTE xk: load all configs EXCEPT ckpt_path and resume the training.
    if config.main.resume_training_config_path is not None:
        if config.main.ckpt_path is None:
            logging.fatal("ckpt_path is required when resuming training from a checkpoint")
        if config.main.model_config_path is not None:
            logging.fatal(
                "model_config_path is not required when resuming training from a checkpoint, since we do not want to overwrite the model config."
            )
        ckpt_path = config.main.ckpt_path
        logging.info(f"Resume all config from {config.main.resume_training_config_path}")
        logging.info(f"Still using ckpt_path: {ckpt_path}")

        config = OmegaConf.load(config.main.resume_training_config_path)
        config.main.ckpt_path = ckpt_path
        # NOTE xk: Not all dataloaders are evaled when resuming. Only the first one runs.
        # Turn off the sanity check or the model checkpoint raise error for no metrics.
        # But it is not work. So we just change the monitor key to the name of the first val dataloader.
        logging.info("Turn off the sanity check for resuming training")
        config.trainer.num_sanity_val_steps = 0

    # NOTE xk: load only model config and keep the other configs.
    if config.main.model_config_path is not None:
        logging.info(f"Load model config from {config.main.model_config_path}")
        model_config = OmegaConf.load(config.main.model_config_path)
        config.model = model_config.model

    # Update config by dotlist
    def _convert_str_to_bool_or_numeric(str_value):
        if str_value.lower() == "true":
            return True
        elif str_value.lower() == "false":
            return False

        try:
            return int(str_value)
        except ValueError:
            pass
        try:
            return float(str_value)
        except ValueError:
            pass

        raise ValueError(f"Value: {str_value} is not a bool, int, or float")

    if update_config_by_dotlist is not None:
        logging.info("Update config by dotlist:")
        update_config_by_dotlist = update_config_by_dotlist.split(",")
        update_config_by_dotlist = (x.split("=") for x in update_config_by_dotlist)
        for key, value in update_config_by_dotlist:
            key = key.strip()
            new_value = value.strip()

            # NOTE: handle not found and none
            old_value = OmegaConf.select(config, key, default=MISSING)
            if old_value == MISSING:
                raise ValueError(f"Key {key} not found in config")
            elif old_value is None:
                try:
                    new_value = _convert_str_to_bool_or_numeric(new_value)
                except ValueError:
                    pass
            elif old_value is not None:
                new_value = type(old_value)(new_value)

            logging.info(f"\t{key}: {old_value} ({type(old_value)}) -> {new_value} ({type(new_value)})")
            OmegaConf.update(config, key, new_value)

    # Set seed
    seed_everything(config.main.seed)

    # Prepare output dirs
    output_dir = config.main.output_dir
    exp_name = config.main.exp_name
    job_name = config.main.job_name
    # NOTE xk: to distinguish the job_name from the main one.
    job_name = f"check_data_loading-{job_name}"
    output_exp_job_dir = osp.join(output_dir, exp_name, job_name)

    log_dir = osp.join(output_exp_job_dir, "log")
    config.main.log_dir = prepare_version_dir(log_dir)
    Path(config.main.log_dir).mkdir(parents=True, exist_ok=True)

    spike_detection_dir = osp.join(output_exp_job_dir, "spike_detection")
    config.main.spike_detection_dir = prepare_version_dir(spike_detection_dir)
    Path(config.main.spike_detection_dir).mkdir(parents=True, exist_ok=True)

    # NOTE xk: when resume, use the previous ckpt_dir.
    if config.main.ckpt_dir is None:
        ckpt_dir = osp.join(output_exp_job_dir, "ckpt")
        config.main.ckpt_dir = prepare_version_dir(ckpt_dir)
        # NOTE xk: Use the same ckpt_dir for TrainerModel.
        config.trainer_model.ckpt_dir = config.main.ckpt_dir
        Path(config.main.ckpt_dir).mkdir(parents=True, exist_ok=True)
    else:
        logging.info(f"Use previous ckpt_dir: {config.main.ckpt_dir}")

    # NOTE xk: when resume, use the previous eval_dir in trainer_model.
    if config.trainer_model.eval_dir is None:
        eval_dir = osp.join(output_exp_job_dir, "eval")
        # NOTE xk: eval_dir is used in TrainerModel.
        config.trainer_model.eval_dir = prepare_version_dir(eval_dir)
        Path(config.trainer_model.eval_dir).mkdir(parents=True, exist_ok=True)
    else:
        logging.info(f"Use previous eval_dir: {config.trainer_model.eval_dir}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0 and (not config.main.log_all_rank):
        logging.set_verbosity(logging.WARNING)

    # Set up logging
    # NOTE xk: Write logs to files
    # https://github.com/abseil/abseil-py/issues/83
    logging.get_absl_handler().use_absl_log_file(f"log-local_rank_{local_rank}", config.main.log_dir)
    # NOTE xk: log both file and stdout
    # https://github.com/abseil/abseil-py/issues/122
    logging.logging.root.addHandler(logging.ABSLHandler(logging.PythonFormatter()))

    # Save config to ckpt_dir
    if local_rank == 0:
        config_path = osp.join(config.main.ckpt_dir, "config.yaml")
        OmegaConf.save(config=config, f=config_path)
        logging.info(f"Save config to {config_path}")

    # Craete dataset
    # NOTE xk: use `pad_gene_collate_fn` to pad the gene data.
    config.data.train_dataloader.pop("collate_fn")
    config.data.val_dataloader.pop("collate_fn")
    trainer_data_module = TrainerDataModule(config.data)

    # Create model
    # NOTE xk: Llama config check the type of `model_args["rope_scaling"]` to be `dict`.
    # So we convert omegaconf.dictconfig.DictConfig to dict.
    model_args = config.model
    # model_class_name = model_args.pop("model_class")
    model_config_class_name = model_args.pop("model_config_class")
    # model_class = create_model_class(model_class_name)
    model_config_class = create_model_config_class(model_config_class_name)

    model_args = OmegaConf.to_container(model_args)

    model_config = model_config_class(**model_args)
    # model = model_class(model_config)
    trainer_model = DummyTainerModel(config.trainer_model)

    # Create trainer
    callbacks = []
    timer_callback = pl_callbacks.Timer(interval="step")
    rich_prog_bar_callback = pl_callbacks.RichProgressBar()

    callbacks.append(timer_callback)
    callbacks.append(rich_prog_bar_callback)

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

    ckpt_path = None
    if not config.main.test_only:
        trainer.fit(trainer_model, datamodule=trainer_data_module, ckpt_path=ckpt_path)
    # trainer.test(trainer_model, datamodule=trainer_data_module, ckpt_path=ckpt_path)


# NOTE: take environment variables from .env.
load_dotenv()

# NOTE xk: Define the flags and files in `config.py` instead of `main.py`
define_flags()

DDP_ENV_VARS = {
    "MASTER_ADDR": None,
    "MASTER_PORT": None,
    "NODE_RANK": None,
    "LOCAL_RANK": 0,
    "WORLD_SIZE": None,
}


if __name__ == "__main__":
    app.run(main)
