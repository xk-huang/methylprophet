"""
Flags updated for eval:

```
--main.test_only
--main.job_name=???
--main.model_config_path=???
--main.weight_path=???

--trainer.devices=???
# torchrun --nproc_per_node=???

--val_dataloader.num_workers=???
--data.val_num_shards.train_cpg_train_sample=???
--data.val_num_shards.train_cpg_val_sample=???
--data.val_num_shards.val_cpg_train_sample=???
--data.val_num_shards.val_cpg_val_sample=???

# For ENCODE WGBS 10% val cpg: 8, 2824, 2824, 2824
# For TCGA ARRAY: TBD

--trainer_model.full_eval=True
--trainer_model.save_eval_results=True
--trainer_model.plot_eval_results=True

--main.ckpt_dir=None
--main.eval_dir=None
--main.log_dir=None
--main.spike_detection_dir=None
--main.wandb_id=None
```


Flags for resume:

```
# torchrun --nproc_per_node=???

--main.resume_training_config_path=???
--main.ckpt_path=???
```


Usage:

```python
NUM_PROCESS=1
JOB_NAME=
INPUT_CONFIG_PATH=

python scripts/tools/generate_eval_resume_config.py \
    --i "${INPUT_CONFIG_PATH}" \
    --job_name "${JOB_NAME}" \
    --devices "${NUM_PROCESS}" \
    --num_shards_train_cpg_train_sample 8 \
    --num_shards_train_cpg_val_sample 2824 \
    --num_shards_val_cpg_train_sample 2824 \
    --num_shards_val_cpg_val_sample 2824 \
    --overwrite
    # --val_dataloader_num_workers 8


RESUME_TRAINING_CONFIG_PATH=
CKPT_PATH=

torchrun --nproc_per_node ${NUM_PROCESS} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 11453 \
-m src.main \
--main.resume_training_config_path="${RESUME_TRAINING_CONFIG_PATH}" --main.ckpt_path="${CKPT_PATH}"
# --main.update_config_by_dotlist="train_dataloader.num_workers=5,val_dataloader.num_workers=5"
```


Misc: profile model and estimate infer time

```python
python \
-m src.profile_model \
--main.resume_training_config_path="${RESUME_TRAINING_CONFIG_PATH}" --main.ckpt_path="${CKPT_PATH}" \
--main.update_config_by_dotlist="trainer.devices=1,model._attn_implementation=eager"
```
"""

import copy
from pathlib import Path

from absl import app, flags, logging
from omegaconf import OmegaConf


flags.DEFINE_string("input_config_path", None, "The input config path.")
flags.mark_flag_as_required("input_config_path")
flags.DEFINE_alias("i", "input_config_path")
flags.DEFINE_string("input_ckpt_path", None, "The input ckpt path.")

flags.DEFINE_string("output_config_dir", None, "The output config dir.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite the existing files.")

flags.DEFINE_string("job_name", None, "The job name.")
flags.DEFINE_string("devices", None, "The devices.")
flags.DEFINE_integer("val_dataloader_num_workers", None, "The model config path.")
flags.DEFINE_integer("num_shards_train_cpg_train_sample", None, "The model config path.")
flags.DEFINE_integer("num_shards_train_cpg_val_sample", None, "The model config path.")
flags.DEFINE_integer("num_shards_val_cpg_train_sample", None, "The model config path.")
flags.DEFINE_integer("num_shards_val_cpg_val_sample", None, "The model config path.")

FLAGS = flags.FLAGS


def main(_):
    input_config_path = Path(FLAGS.input_config_path)
    logging.info(f"input config path: {input_config_path}")
    input_ckpt_path = FLAGS.input_ckpt_path
    if input_ckpt_path is None:
        input_ckpt_path = input_config_path.parent / "last.ckpt"
        logging.info(f"input_ckpt_path is None, try default {input_ckpt_path}.")
    else:
        input_ckpt_path = Path(input_ckpt_path)
        logging.info(f"input ckpt path: {input_ckpt_path}")
    if not input_config_path.exists():
        raise ValueError(f"{input_config_path} does not exist.")
    if not input_ckpt_path.exists():
        raise ValueError(f"{input_ckpt_path} does not exist.")

    if FLAGS.output_config_dir is None:
        output_config_dir = input_config_path.parent
    output_eval_config_path = output_config_dir / f"eval-{input_config_path.stem}.yaml"
    output_resume_config_path = output_config_dir / f"resume-{input_config_path.stem}.yaml"

    check_overwrite(output_eval_config_path)
    check_overwrite(output_resume_config_path)

    config = OmegaConf.load(input_config_path)

    # Prepare resume config
    logging.info(f"Prepare resume config: {output_resume_config_path}")
    resume_config = prepare_resume_config(input_config_path, input_ckpt_path, config)

    # Prepare eval config
    logging.info(f"Prepare eval config: {output_eval_config_path}")
    eval_config = prepare_eval_config(input_config_path, input_ckpt_path, config)

    OmegaConf.save(resume_config, output_resume_config_path)
    logging.info(f"Saved resume config: {output_resume_config_path}")
    OmegaConf.save(eval_config, output_eval_config_path)
    logging.info(f"Saved eval config: {output_eval_config_path}")


def prepare_resume_config(input_config_path, input_ckpt_path, config):
    resume_training_config_path = get_default_value(config, "main.resume_training_config_path", str(input_config_path))
    ckpt_path = get_default_value(config, "main.ckpt_path", str(input_ckpt_path))

    resume_config = copy.deepcopy(config)
    OmegaConf.update(resume_config, "main.resume_training_config_path", resume_training_config_path)
    OmegaConf.update(resume_config, "main.ckpt_path", ckpt_path)

    return resume_config


def prepare_eval_config(input_config_path, input_ckpt_path, config):
    test_only = get_default_value(config, "main.test_only", True)
    job_name = get_default_value(config, "main.job_name", FLAGS.job_name)
    model_config_path = get_default_value(config, "main.model_config_path", str(input_config_path))
    weight_path = get_default_value(config, "main.weight_path", str(input_ckpt_path))

    devices = get_default_value(config, "trainer.devices", FLAGS.devices)

    num_workers = get_default_value(config, "val_dataloader.num_workers", FLAGS.val_dataloader_num_workers)
    num_shards_train_cpg_train_sample = get_default_value(
        config, "data.val_num_shards.train_cpg_train_sample", FLAGS.num_shards_train_cpg_train_sample
    )
    num_shards_train_cpg_val_sample = get_default_value(
        config, "data.val_num_shards.train_cpg_val_sample", FLAGS.num_shards_train_cpg_val_sample
    )
    num_shards_val_cpg_train_sample = get_default_value(
        config, "data.val_num_shards.val_cpg_train_sample", FLAGS.num_shards_val_cpg_train_sample
    )
    num_shards_val_cpg_val_sample = get_default_value(
        config, "data.val_num_shards.val_cpg_val_sample", FLAGS.num_shards_val_cpg_val_sample
    )

    full_eval = get_default_value(config, "trainer_model.full_eval", True)
    save_eval_results = get_default_value(config, "trainer_model.save_eval_results", True)
    plot_eval_results = get_default_value(config, "trainer_model.plot_eval_results", True)

    eval_config = copy.deepcopy(config)

    OmegaConf.update(eval_config, "main.test_only", test_only)
    OmegaConf.update(eval_config, "main.job_name", job_name)
    OmegaConf.update(eval_config, "main.model_config_path", model_config_path)
    OmegaConf.update(eval_config, "main.weight_path", weight_path)

    OmegaConf.update(eval_config, "trainer.devices", devices)

    OmegaConf.update(eval_config, "val_dataloader.num_workers", num_workers)
    OmegaConf.update(eval_config, "data.val_num_shards.train_cpg_train_sample", num_shards_train_cpg_train_sample)
    OmegaConf.update(eval_config, "data.val_num_shards.train_cpg_val_sample", num_shards_train_cpg_val_sample)
    OmegaConf.update(eval_config, "data.val_num_shards.val_cpg_train_sample", num_shards_val_cpg_train_sample)
    OmegaConf.update(eval_config, "data.val_num_shards.val_cpg_val_sample", num_shards_val_cpg_val_sample)

    OmegaConf.update(eval_config, "trainer_model.full_eval", full_eval)
    OmegaConf.update(eval_config, "trainer_model.save_eval_results", save_eval_results)
    OmegaConf.update(eval_config, "trainer_model.plot_eval_results", plot_eval_results)

    OmegaConf.update(eval_config, "main.ckpt_dir", None)
    OmegaConf.update(eval_config, "main.eval_dir", None)
    OmegaConf.update(eval_config, "main.log_dir", None)
    OmegaConf.update(eval_config, "main.spike_detection_dir", None)
    OmegaConf.update(eval_config, "main.wandb_id", None)

    return eval_config


def get_default_value(config, key, default_value=None):
    old_value = OmegaConf.select(config, key)
    if default_value is None:
        logging.info(f"keep {key}: {old_value}")
        return old_value
    logging.info(f"update {key}: {old_value} -> {default_value}")
    return default_value


def check_overwrite(file_path):
    if file_path.exists():
        if FLAGS.overwrite:
            logging.warning(f"Overwriting {file_path}")
        else:
            raise ValueError(f"{file_path} exists. Set --overwrite to overwrite it.")


if __name__ == "__main__":
    app.run(main)
