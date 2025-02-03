"""
The config file of ml_collections is a python file that contains the configuration of the project.

https://github.com/google/ml_collections/tree/master?tab=readme-ov-file#config-flags
https://github.com/google/ml_collections/tree/master?tab=readme-ov-file#parameterising-the-get_config-function
"""

from ml_collections import config_dict as ml_config_dict

from src.config import create_config_dict


def get_config():
    kwargs = {
        "seed": 42,
        "output_dir": "outputs/",
        "exp_name": "debug",
        "job_name": "dummy",
        "ckpt_dir": None,
        "log_dir": None,
        "wandb_id": None,
        "spike_detection_dir": None,
        "ckpt_path": ml_config_dict.placeholder(
            object
        ),  # NOTE xk: we load the checkpoint, incl. weights, optimizer, scheduler, etc.
        "weight_path": ml_config_dict.placeholder(object),  # NOTE xk: we only load weight
        "resume_training_config_path": ml_config_dict.placeholder(
            object
        ),  # NOTE xk: we load all configs to resume the training.
        "model_config_path": ml_config_dict.placeholder(object),  # NOTE xk: we only load the model config.
        "train_data_random_sampling_seed": 0,
        "log_all_rank": False,
        "test_only": False,
        "find_batch_size": False,
        "find_batch_size_init_val": 2,
        "detect_spike": False,
        "pl_logger_type": "wandb",  # NOTE xk: "wandb" or "tensorboard"
        # NOTE: xk: Use dotlist in omegaconf, split by ','.
        # https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-a-dot-list
        # e.g., "trainer.max_steps=500,train_dataloader.back_size=10,main.wandb_id=null"
        # To enbale None overriding, use case-insensitive "null" in the dotlist.
        "update_config_by_dotlist": ml_config_dict.placeholder(object),
    }
    config_dict = create_config_dict(**kwargs)

    return config_dict
