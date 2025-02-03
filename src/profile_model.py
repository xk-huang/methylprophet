"""
Check the data loading process.

1. Make sure all the data is loaded correctly, w/ or w/o multi-processing loading or not.
2. Profile the time it takes to load the data.
3. Test distributed loading is correct.
4. Check the cpg and sample intersection between different splits.
"""

import inspect
import os
import os.path as osp
import pprint
from collections import OrderedDict
from pathlib import Path
from pprint import pformat

import pandas as pd
from absl import app, flags, logging
from lightning import seed_everything
from omegaconf import MISSING, OmegaConf

from src.config import define_flags
from src.data.trainer_data_module import TrainerDataModule
from src.models.model_factory import create_model_class, create_model_config_class
from src.trainer_model import TrainerModel
from src.utils import get_model_complexity_info, prepare_version_dir


# NOTE xk: Define the flags and files in `config.py` instead of `main.py`
define_flags()

DDP_ENV_VARS = {
    "MASTER_ADDR": None,
    "MASTER_PORT": None,
    "NODE_RANK": None,
    "LOCAL_RANK": 0,
    "WORLD_SIZE": None,
}


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

    if config.main.ckpt_dir is None:
        ckpt_dir = osp.join(output_exp_job_dir, "ckpt")
        config.main.ckpt_dir = prepare_version_dir(ckpt_dir)
        # NOTE xk: Use the same ckpt_dir for TrainerModel.
        config.trainer_model.ckpt_dir = config.main.ckpt_dir
        Path(config.main.ckpt_dir).mkdir(parents=True, exist_ok=True)
    else:
        logging.info(f"Use previous ckpt_dir: {config.main.ckpt_dir}")

    eval_dir = osp.join(output_exp_job_dir, "eval")
    # NOTE xk: eval_dir is used in TrainerModel.
    config.trainer_model.eval_dir = prepare_version_dir(eval_dir)
    Path(config.trainer_model.eval_dir).mkdir(parents=True, exist_ok=True)

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

    # Craete dataset
    # NOTE xk: use `pad_gene_collate_fn` to pad the gene data.
    config.data.train_dataloader.pop("collate_fn")
    config.data.val_dataloader.pop("collate_fn")
    logging.info(f"update train dataloader batch size from {config.data.train_dataloader.batch_size} to 1")
    config.data.train_dataloader.batch_size = 1
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
    trainer_model.configure_optimizer_params()
    fwd_flops = trainer_model.num_fwd_flops
    bck_flops = trainer_model.num_bck_flops
    logging.info(f"Forward Flops: {fwd_flops}, Backward Flops: {bck_flops}, Total Flops: {fwd_flops + bck_flops}")

    # Prepare dataloader
    trainer_data_module.setup()
    train_dataloader = trainer_data_module.train_dataloader()
    train_batch = next(iter(train_dataloader))
    train_dataloader_batch_size = config.data.train_dataloader.batch_size

    # Log number of batches
    # estimate_batch_and_infer_time_for_all(config, trainer_data_module)

    # Log model complexity
    input_tuple = []
    for k in inspect.signature(model.forward).parameters.keys():
        if k in ["kwargs", "args"]:
            continue
        if k in train_batch:
            input_tuple.append(train_batch[k])
    input_tuple = tuple(input_tuple)

    analize_model_complexity(model, train_dataloader_batch_size, input_tuple)
    logging.info(f"Logging model complexity to {log_dir}")


def analize_model_complexity(model, train_dataloader_batch_size, input_tuple):
    logging.info("Model Complexity:")
    analysis_results = get_model_complexity_info(model, inputs=input_tuple)

    logging.debug(f"Model Flops:\n{analysis_results['flops_str']}")
    logging.debug(f"Model Parameters:\n{analysis_results['params_str']}")

    logging.debug(f"Train batch size: {train_dataloader_batch_size}")
    logging.debug(f"Detailed Model Complexity with Arch:\n{analysis_results['out_arch']}")

    logging.info(f"Train batch size: {train_dataloader_batch_size}")
    logging.info(f"Detailed Model Complexity with Table:\n{analysis_results['out_table']}")
    logging.warning(
        "The code for FLOPS in fvcore/detectron is not FLOPS, but MACs, which means it does not multiply by 2 for add operations."
    )


def estimate_batch_and_infer_time_for_all(config, trainer_data_module):
    logging.info("Number of batches for each dataloader:")
    dataset_stats = trainer_data_module.dataset_stats

    all_estimate_batch_and_infer_time = []
    train_dataset_name = trainer_data_module.train_dataset.name
    train_dataloader_batch_size = config.train_dataloader.batch_size
    estimate_batch_and_infer_time_dict = estimate_batch_and_infer_time(
        dataset_stats, "train", train_dataset_name, train_dataloader_batch_size
    )
    all_estimate_batch_and_infer_time.append(estimate_batch_and_infer_time_dict)

    for val_dataset in trainer_data_module.val_dataset_ls:
        val_dataset_name = val_dataset.name
        val_dataloader_batch_size = config.val_dataloader.batch_size
        estimate_batch_and_infer_time_dict = estimate_batch_and_infer_time(
            dataset_stats, "val", val_dataset_name, val_dataloader_batch_size
        )
        all_estimate_batch_and_infer_time.append(estimate_batch_and_infer_time_dict)

    all_estimate_batch_and_infer_time = pd.DataFrame(all_estimate_batch_and_infer_time)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
    ):
        logging.info(f"Estimate batch and infer time:\n{all_estimate_batch_and_infer_time}")

    output_csv_path = Path(config.trainer_model.eval_dir) / "estimate_batch_and_infer_time.csv"
    all_estimate_batch_and_infer_time.to_csv(output_csv_path)
    logging.info(f"Save the estimate batch and infer time to {output_csv_path}")


def estimate_batch_and_infer_time(
    dataset_stats,
    dataset_type,
    dataset_name,
    dataloader_batch_size,
    batches_per_second=1,
):
    if dataset_type not in ["train", "val", "debug"]:
        raise ValueError(f"dataset_type should be one of ['train', 'val', 'debug'], but got {dataset_type}")

    log_dict = get_stats(dataset_stats, dataset_name)

    num_batches = log_dict["num_cpg_sample_pairs"] // dataloader_batch_size

    est_infer_hours = num_batches / batches_per_second / 3600
    raw_log_dict = {
        "dataset_type": dataset_type,
        "dataset_name": dataset_name,
        **log_dict,
        "dataloader_batch_size": dataloader_batch_size,
        "num_batches": num_batches,
        "batches_per_second": batches_per_second,
        "est_infer_hours": est_infer_hours,
    }

    log_dict = OrderedDict()
    for k, v in raw_log_dict.items():
        if isinstance(v, float):
            log_dict[k] = f"{v:.2f}"
        elif isinstance(v, int):
            log_dict[k] = f"{v:,}"
        else:
            log_dict[k] = v
    logging.debug(f"Estimate training time:\n{pformat(log_dict)}")

    return log_dict


def get_stats(dataset_stats, name):
    # Convert "val_cpg_val_sample" to "val_cpg-val_sample.parquet"
    name = name.split("_")
    name = f"{name[0]}_{name[1]}-{name[2]}_{name[3]}.parquet"
    if name not in dataset_stats:
        raise ValueError(
            f"Dataset {name} not found in dataset_stats, may need to re-run"
            "scripts/tools/data_preprocessing/stats_processed_dataset_shards.py"
        )
    return {
        "num_cpg": dataset_stats[name]["num_cpg"],
        "num_sample": dataset_stats[name]["num_sample"],
        "num_cpg_sample_pairs": dataset_stats[name]["num_cpg_sample_pairs"],
    }


if __name__ == "__main__":
    app.run(main)
