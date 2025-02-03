from absl import app, flags, logging
from omegaconf import OmegaConf

from src.config import define_flags

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


if __name__ == "__main__":
    app.run(main)
