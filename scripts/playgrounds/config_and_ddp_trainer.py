import inspect
import os
import pprint

import torch
from absl import app, flags
from lightning import LightningModule, Trainer
from ml_collections import config_dict, config_flags


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, config=None):
        self.config = config

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(1)


class DummyModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.module = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.module(x)

    def test_step(self, batch, batch_idx):
        return batch.mean()

    def test_dataloader(self):
        return torch.utils.data.DataLoader(DummyDataset(self.config), batch_size=1)


# NOTE xk: The config system is copied from tux (used by LWM).
# https://github.com/forhaoliu/tux/
# The code is modified to support multiple types.
def function_args_to_config(fn, none_arg_types=None, exclude_args=None, override_args=None):
    config = config_dict.ConfigDict()
    # NOTE xk: Inspect the parameters of the function
    # https://stackoverflow.com/questions/43293241/how-to-print-init-function-arguments-in-python
    for name, parameter in inspect.signature(fn).parameters.items():
        value = parameter.default
        if exclude_args is not None and name in exclude_args:
            continue
        elif override_args is not None and name in override_args:
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


def print_flags(flags, flags_def):
    flag_srings = ["{}: {}".format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
    print("Hyperparameter configs: \n{}".format(pprint.pformat(flag_srings)))


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


FLAGS, FLAGS_DEF = define_flags_with_default(
    trainer=function_args_to_config(Trainer), model=function_args_to_config(DummyModel)
)

DDP_ENV_VARS = {
    "MASTER_ADDR": None,
    "MASTER_PORT": None,
    "NODE_RANK": None,
    "LOCAL_RANK": 0,
    "WORLD_SIZE": None,
}


def main(_):
    # NOTE xk: get ddp env
    # e.g., {'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '54669', 'NODE_RANK': '0', 'LOCAL_RANK': '6', 'WORLD_SIZE': '8'}
    ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
    print(ddp_env_vars)

    model = DummyModel(FLAGS.model)

    trainer = Trainer(**FLAGS.trainer)
    trainer.test(model)


if __name__ == "__main__":
    app.run(main)
    app.run(main)
