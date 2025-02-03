"""
The config file of ml_collections is a python file that contains the configuration of the project.

https://github.com/google/ml_collections/tree/master?tab=readme-ov-file#config-flags
https://github.com/google/ml_collections/tree/master?tab=readme-ov-file#parameterising-the-get_config-function
"""

from ml_collections import config_dict as ml_config_dict

from src.config import create_config_dict


def get_config(config_type):
    kwargs = {
        "full_eval": True,
        "save_eval_results": True,  # NOTE xk: Unused
        "eval_save_batch_interval": 100_000,
        "plot_eval_results": True,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "scheduler_type": "constant",
        "scheduler_num_training_steps": ml_config_dict.placeholder(object),
        "scheduler_num_warmup_steps": ml_config_dict.placeholder(object),
        "eval_dir": ml_config_dict.placeholder(object),
        "ckpt_dir": ml_config_dict.placeholder(object),
        "min_lr_rate": 0.1,  # Only used for `cosine_with_min_lr` scheduler type
        "pl_logger_type": None,  # NOTE xk: should be overwritten by the main.py `config.main.pl_logger_type`
        "use_bin_logits_cls_loss": False,
        "num_bins": 101,
        "bin_min_val": 0.0,
        "bin_max_val": 1.0,
        "gradient_checkpointing": False,
        "speed_monitor": create_config_dict(
            window_size=100,
            gpu_flops_available=ml_config_dict.placeholder(object),
        ),
        "estimated_sequence_length_for_flops": 200,
    }
    config_dict = create_config_dict(
        **kwargs,
    )
    config_dict_collections = {"default": config_dict}

    return config_dict_collections[config_type]
