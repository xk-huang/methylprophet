import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_chr_df_parquet", None, "Path to the parquet file")
flags.mark_flag_as_required("input_chr_df_parquet")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output directory name")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_integer("seed", 42, "Random seed for split")
flags.DEFINE_float("val_ratio", 0.1, "Validation ratio")
flags.DEFINE_float("train_ratio", 0.01, "Train ratio")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")

FLAGS = flags.FLAGS


def main(_):
    input_chr_df_parquet = Path(FLAGS.input_chr_df_parquet)

    output_dir = Path(FLAGS.output_dir)
    output_dir = output_dir / FLAGS.output_file_name
    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_dir} already exists. Skipping...")
            return
        else:
            logging.warning(f"Overwriting {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Reading {input_chr_df_parquet}")
    chr_df = pd.read_parquet(input_chr_df_parquet)

    seed = FLAGS.seed
    val_ratio = FLAGS.val_ratio
    train_ratio = FLAGS.train_ratio
    logging.info(f"Creating train and val split with seed {seed}, val_ratio {val_ratio}, train_ratio {train_ratio}")
    if val_ratio + train_ratio > 1:
        raise ValueError(
            f"val_ratio + train_ratio should be less than or equal to 1, but got {val_ratio}+{train_ratio}={val_ratio + train_ratio}"
        )

    rng = np.random.default_rng(seed)
    chr_values = chr_df["chr"].unique()
    all_val_df = []
    all_train_df = []
    pbar = tqdm.tqdm(chr_values, desc="Splitting by chr")
    for chr_value in chr_values:
        chr_df_value = chr_df[chr_df["chr"] == chr_value]
        chr_df_value = chr_df_value.sample(frac=1, random_state=rng)

        num_samples = len(chr_df_value)
        num_val_samples = int(num_samples * val_ratio)
        num_train_samples = int(num_samples * train_ratio)

        val_df = chr_df_value.iloc[:num_val_samples]
        train_df = chr_df_value.iloc[num_val_samples : num_val_samples + num_train_samples]

        all_val_df.append(val_df)
        all_train_df.append(train_df)
        pbar.update()

    all_val_df = pd.concat(all_val_df)
    all_train_df = pd.concat(all_train_df)

    # NOTE: keep the index sorted in val split
    all_val_df.sort_index(inplace=True)
    # NOTE: shuffle the index in train split
    all_train_df = all_train_df.sample(frac=1, random_state=rng)

    all_val_df.to_parquet(output_dir / "val.parquet")
    all_train_df.to_parquet(output_dir / "train.parquet")
    logging.info(f"Saved val.parquet and train.parquet to {output_dir}")


if __name__ == "__main__":
    app.run(main)
