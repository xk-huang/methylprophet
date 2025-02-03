"""
The config system is copied from tux (used by LWM).
https://github.com/forhaoliu/tux/
The code is modified to support multiple types.

The basic element is the `ConfigDict`, we use `create_config_dict` to create a `ConfigDict` object.

The `function_args_to_config` function is used to convert the function arguments to a `ConfigDict` object.

The `define_flags_with_default` function is used to define the flags with default values, the values can be `ConfigDict`, `bool`, `int`, `float`, `str`. Note that this function is called to craete top-level flags.
The same goes for `define_files_with_default`.
"""

import inspect
import pprint

from absl import app, flags, logging
from lightning import Trainer
from ml_collections import config_dict, config_flags
from torch.utils.data import DataLoader

from src.data.dataset import create_methylformer_streaming_dataset


def create_config_dict(**kwargs):
    return config_dict.ConfigDict(dict(**kwargs))


def function_args_to_config(fn, none_arg_types=None, exclude_args=None, override_args=None):
    config = config_dict.ConfigDict()
    # NOTE xk: Inspect the parameters of the function
    # https://stackoverflow.com/questions/43293241/how-to-print-init-function-arguments-in-python
    for name, parameter in inspect.signature(fn).parameters.items():
        value = parameter.default
        if exclude_args is not None and name in exclude_args:
            continue
        elif override_args is not None and name in override_args:
            if override_args[name] is None:
                config[name] = config_dict.placeholder(object)
            else:
                config[name] = override_args[name]
        elif none_arg_types is not None and value is None and name in none_arg_types:
            config[name] = config_dict.placeholder(none_arg_types[name])
        elif value is None or value is inspect._empty:
            # FIXME xk: object is any type, which means there is no type check
            # https://stackoverflow.com/questions/49171189/whats-the-correct-way-to-check-if-an-object-is-a-typing-generic
            # We should use `get_origin` and `get_args` to check if the type is a generic type
            # https://stackoverflow.com/questions/49171189/whats-the-correct-way-to-check-if-an-object-is-a-typing-generic
            config[name] = config_dict.placeholder(object)
        else:
            config[name] = value

    return config


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            val, help_str = val
        else:
            help_str = ""

        if isinstance(val, config_dict.ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
            # FIXME xk: --flagfile is not compatible with DEFINE_config_dict.
            # The argv is not eval via absl.flags, so the config_dict is not parsed.
            # We can fix this by: https://github.com/google/ml_collections/pull/29/files
            # FIXME xk: install the latest version of ml_collections via github and edit the code.

        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            flags.DEFINE_bool(key, val, help_str)
        elif isinstance(val, int):
            flags.DEFINE_integer(key, val, help_str)
        elif isinstance(val, float):
            flags.DEFINE_float(key, val, help_str)
        elif isinstance(val, str):
            flags.DEFINE_string(key, val, help_str)
        else:
            raise ValueError("Incorrect value type")
    return flags.FLAGS, kwargs


def define_files_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, tuple):
            val, help_str = val
        else:
            help_str = ""

        config_flags.DEFINE_config_file(key, val, help_str)
    return flags.FLAGS, kwargs


def print_flags(flags, flags_def):
    flag_srings = ["{}: {}".format(key, val) for key, val in get_user_flags(flags, flags_def).items()]

    msg = "Hyperparameter configs: \n{}".format(pprint.pformat(flag_srings))
    if logging.get_absl_logger() is not None:
        logging.info(msg)
    else:
        print(msg)


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, config_dict.ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, config_dict.ConfigDict) or isinstance(val, dict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def define_flags():
    # NOTE xk: Define the flags and files in `config.py` instead of `main.py`
    define_flags_with_default(
        data=create_config_dict(
            train_dataset=function_args_to_config(create_methylformer_streaming_dataset),
            val_dataset=function_args_to_config(create_methylformer_streaming_dataset),
            train_dataloader=function_args_to_config(DataLoader, exclude_args=["dataset"]),
            val_dataloader=function_args_to_config(DataLoader, exclude_args=["dataset"]),
        ),
        trainer=function_args_to_config(
            Trainer, override_args={"check_val_every_n_epoch": None, "overfit_batches": None}
        ),
    )
    define_files_with_default(
        model=("src/configs/models/methylformer_bert.py:debug", "Model config file"),
        trainer_model=("src/configs/trainer_model.py:default", "Trainer model config file"),
        main=("src/configs/main.py", "Main config file"),
    )
