"""
Run the command below:

```bash
output_dir=outputs
exp_name=example_data
job_name=encode_wgbs

job_name="try-${job_name}"

val_num_workers=20

val_batch_size=12
num_nbase=1000

model_config_path=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/config.yaml
weight_path=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/finished.ckpt

dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg

local=data/examples/encode_wgbs/val_10_shards
group_idx_name_mapping_path=data/examples/encode_wgbs/val_10_shards/group_idx_name_mapping.json
gene_expr_df_path=data/examples/encode_wgbs/gene_expr.filtered.parquet
sample_idx_path=data/examples/encode_wgbs/sample_tissue_count_with_idx.csv

python scripts/examples/predict_example_data.py \
--trainer.devices=1 \
--trainer.num_nodes=1 \
--trainer.accelerator=gpu \
--trainer.precision=bf16 \
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--flagfile=${dataset_flagfile} \
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
--main.test_only \
--main.job_name=${job_name} \
--main.model_config_path="${model_config_path}" --main.weight_path="${weight_path}" \
--data.train_dataset.local=${local} \
--data.val_dataset.local=${local} \
--data.train_dataset.group_idx_name_mapping_path=${group_idx_name_mapping_path} \
--data.val_dataset.group_idx_name_mapping_path=${group_idx_name_mapping_path} \
--data.train_dataset.gene_expr_df_path=${gene_expr_df_path} \
--data.val_dataset.gene_expr_df_path=${gene_expr_df_path} \
--data.train_dataset.sample_idx_path=${sample_idx_path} \
--data.val_dataset.sample_idx_path=${sample_idx_path}
```
"""

import os
import os.path as osp
import pprint
from pathlib import Path

import torch
from absl import app, flags, logging
from dotenv import load_dotenv
from lightning import Trainer, seed_everything
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner import Tuner
from omegaconf import MISSING, OmegaConf
from wandb.util import generate_id

from src.config import define_flags
from src.data.trainer_data_module import TrainerDataModule
from src.models.model_factory import create_model_class, create_model_config_class
from src.trainer_model import TrainerModel
from src.utils import prepare_version_dir


DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def convert_data_type(data, float_dtype, device):
    if isinstance(data, torch.Tensor):
        if torch.is_floating_point(data):
            return data.to(dtype=float_dtype, device=device)
        return data.to(device=device)
    elif isinstance(data, dict):
        return {key: convert_data_type(value, float_dtype, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_data_type(value, float_dtype, device) for value in data]
    else:
        return data


def run_inference(trainer_model, trainer_data_module, config):
    # Prepare model
    dtype = config.trainer.precision
    device = "cuda"
    trainer_model.to(dtype=DTYPE_MAPPING[dtype], device=device)

    trainer_data_module.setup("val")
    val_dataloader = trainer_data_module.val_dataloader()
    batch = next(iter(val_dataloader))
    batch = convert_data_type(batch, DTYPE_MAPPING[dtype], device)

    logging.info(f"Batch: {pprint.pformat(batch)}")
    trainer_model.eval()
    with torch.no_grad():
        sample_idx = batch.pop("sample_idx", None)
        if sample_idx is None:
            raise ValueError("sample_idx should not be None.")

        cpg_idx = batch.pop("cpg_idx", None)
        if cpg_idx is None:
            raise ValueError("cpg_idx should not be None.")

        group_idx = batch.pop("group_idx", None)
        if group_idx is None:
            raise ValueError("group_idx should not be None")

        gt_me = batch.pop("methylation", None)

        outputs = trainer_model.model(**batch)
        pred_me = outputs["output_value"]

    outputs = {
        "pred_me": pred_me,
        "gt_me": gt_me,
    }
    logging.info(f"Outputs: {pprint.pformat(outputs)}")


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

    # Get wandb id
    # NOTE: use generate_id to identify each run https://github.com/wandb/wandb/issues/335#issuecomment-493284910
    # Also refer to https://github.com/xk-huang/segment-caption-anything/blob/0d3f0b4a9caa8d5f8d23f5a301b9048161e930bc/src/integrations.py#L63-L71
    wandb_id = config.main.wandb_id
    if wandb_id is None:
        logging.info("New run, generate wandb id")
        wandb_id = generate_id()
    else:
        logging.info(f"Resume run, use wandb id: {wandb_id}")
    config.main.wandb_id = wandb_id

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
    model_class_name = model_args.pop("model_class")
    model_config_class_name = model_args.pop("model_config_class")
    model_class = create_model_class(model_class_name)
    model_config_class = create_model_config_class(model_config_class_name)

    model_args = OmegaConf.to_container(model_args)

    model_config = model_config_class(**model_args)
    model = model_class(model_config)
    trainer_model = TrainerModel(config.trainer_model, model)

    # Create trainer
    callbacks = []
    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        filename="{step}",
        # filename="step={step}",
        # auto_insert_metric_name=False,
        # monitor="val/train_cpg_train_sample/mse_loss",
        # mode="min",
        # save_top_k=3,
        dirpath=config.main.ckpt_dir,
        # save_last=True,
    )
    # early_stopping_callback = EarlyStopping("val/mse_loss", mode="min", patience=3)
    learning_rate_monitor_callback = pl_callbacks.LearningRateMonitor(logging_interval="step")
    timer_callback = pl_callbacks.Timer(interval="step")
    rich_prog_bar_callback = pl_callbacks.RichProgressBar()
    # device_stats_monitor_callback = pl_callbacks.DeviceStatsMonitor()
    # NOTE xk: the same path https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/spike.html#SpikeDetection
    spike_detection = pl_callbacks.SpikeDetection(
        mode="min",
        window=100,
        warmup=1000,
        exclude_batches_path=osp.join(config.main.spike_detection_dir, "skip_batches.json"),
        finite_only=False,
    )

    callbacks.append(checkpoint_callback)
    # callbacks.append(early_stopping_callback)
    callbacks.append(learning_rate_monitor_callback)
    callbacks.append(timer_callback)
    callbacks.append(rich_prog_bar_callback)
    # callbacks.append(device_stats_monitor_callback)
    if config.main.detect_spike:
        callbacks.append(spike_detection)

    loggers = []
    pl_logger_type = config.main.pl_logger_type
    config.trainer_model.pl_logger_type = pl_logger_type
    if pl_logger_type == "tensorboard":
        loggers.append(pl_loggers.TensorBoardLogger(save_dir=osp.join(output_exp_job_dir, "tb_log")))
        # NOTE xk: To log hyperparameters, we need to convert the OmegaConf to dict.
        # Ref: https://github.com/xk-huang/segment-caption-anything/blob/0d3f0b4a9caa8d5f8d23f5a301b9048161e930bc/src/integrations.py#L78
        loggers[-1].log_hyperparams(OmegaConf.to_container(config))
    elif pl_logger_type == "wandb":
        wandb_log_dir = osp.join(output_exp_job_dir, "wandb_log")
        wandb_log_dir = prepare_version_dir(wandb_log_dir, mkdir=True)
        loggers.append(pl_loggers.WandbLogger(save_dir=wandb_log_dir, name=job_name, group=exp_name, id=wandb_id))
        # NOTE xk: To log hyperparameters, we need to convert the OmegaConf to dict.
        # Ref: https://github.com/xk-huang/segment-caption-anything/blob/0d3f0b4a9caa8d5f8d23f5a301b9048161e930bc/src/integrations.py#L78
        loggers[-1].log_hyperparams(OmegaConf.to_container(config))
    else:
        raise ValueError(f"Unsupported logger type: {pl_logger_type}")
    loggers.append(pl_loggers.CSVLogger(save_dir=osp.join(output_exp_job_dir, "csv_log")))

    trainer_kwargs = config.trainer
    trainer_kwargs = OmegaConf.to_container(trainer_kwargs)
    # NOTE xk: Tuning the batch size is currently not supported with distributed strategies
    if not config.main.find_batch_size:
        # NOTE xk: find_unused_parameters is used as some parameters are not used in the forward pass during dev.
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
    trainer_kwargs.update({"callbacks": callbacks, "logger": loggers, "strategy": strategy})
    # XXX: "overfit_batches" is Union[int, float], which is not supported in ml_collections.
    if trainer_kwargs["overfit_batches"] is None:
        trainer_kwargs["overfit_batches"] = 0.0
    # we need to specially handle it from the trainer_kwargs
    logging.info(f"Trainer kwargs: {pprint.pformat(trainer_kwargs)}")
    trainer = Trainer(**trainer_kwargs)

    if config.main.find_batch_size:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(
            trainer_model,
            datamodule=trainer_data_module,
            mode="binsearch",
            init_val=config.main.find_batch_size_init_val,
        )
        logging.info(f"Best batch size: {trainer_data_module.batch_size}")
        exit()

    ckpt_path = config.main.ckpt_path
    weight_path = config.main.weight_path
    if weight_path is not None:
        # NOTE xk: only load weights via torch api.
        # https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html
        weight_state_dict = torch.load(weight_path)
        weight_state_dict = weight_state_dict["state_dict"]
        for weight_key in list(weight_state_dict.keys()):
            if weight_key.startswith("model."):
                weight_state_dict[weight_key[len("model.") :]] = weight_state_dict.pop(weight_key)

        trainer_model.model.load_state_dict(weight_state_dict)
        del weight_state_dict

    run_inference(trainer_model, trainer_data_module, config)


# NOTE: take environment variables from .env.
load_dotenv()


# NOTE: Define the flags and files in `config.py` instead of `main.py`
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
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
    app.run(main)
