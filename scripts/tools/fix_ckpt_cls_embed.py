from pathlib import Path

import torch
from absl import app, flags, logging


flags.DEFINE_string("input_ckpt_path", None, "Path to the input checkpoint")
flags.mark_flag_as_required("input_ckpt_path")
flags.DEFINE_alias("i", "input_ckpt_path")

flags.DEFINE_string("output_dir", None, "Path to the output checkpoint")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output checkpoint if it exists")

FLAGS = flags.FLAGS


def main(_):
    input_ckpt_path = Path(FLAGS.input_ckpt_path)
    ckpt_name, ckpt_suffix = input_ckpt_path.stem, input_ckpt_path.suffix

    # Prepare output directory
    output_dir = FLAGS.output_dir
    if output_dir is None:
        output_dir = input_ckpt_path.parent
    else:
        output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_ckpt_path = output_dir / f"{ckpt_name}-cls_embed_fixed{ckpt_suffix}"
    if output_ckpt_path.exists() and not FLAGS.overwrite:
        logging.error(f"Output checkpoint already exists: {output_ckpt_path}")
        return

    # Load checkpoint
    logging.info(f"Reading checkpoint: {input_ckpt_path}")
    ckpt = torch.load(input_ckpt_path, map_location="cpu")

    if "model.cls_embed" not in ckpt["state_dict"]:
        if "model.cls_embed.weight" not in ckpt["state_dict"]:
            raise ValueError(
                "Unfortunatedly, no 'model.cls_embed.weight' or 'model.cls_embed' found in the checkpoint"
            )
        else:
            logging.info(
                "Good news! No 'model.cls_embed' found in the checkpoint, "
                "and 'model.cls_embed.weight' found in the checkpoint."
            )
        return
    else:
        logging.info("Found 'model.cls_embed' in the checkpoint, we need to fix it")

    ckpt["state_dict"]["model.cls_embed.weight"] = ckpt["state_dict"]["model.cls_embed"]
    del ckpt["state_dict"]["model.cls_embed"]

    # Save checkpoint
    logging.info(f"Saving checkpoint: {output_ckpt_path}")
    torch.save(ckpt, output_ckpt_path)


if __name__ == "__main__":
    app.run(main)
