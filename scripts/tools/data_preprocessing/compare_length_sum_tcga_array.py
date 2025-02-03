"""
python scripts/tools/data_preprocessing/compare_length_sum_tcga_array.py \
    --input_dataset_dir_a data/processed/241023-tcga_array-train_0_9_val_0_1-ind_cancer-nan \
    --input_dataset_dir_b data/processed/241023-tcga_array-train_0_9_val_0_1-ind_cancer-nan-resharded
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_dataset_dir_a", None, "Input dataset directory A")
flags.DEFINE_string("input_dataset_dir_b", None, "Input dataset directory B")
flags.mark_flag_as_required("input_dataset_dir_a")


FLAGS = flags.FLAGS
ME_CPG_BG_SPLIT_NAMES = [
    "train_cpg-train_sample.parquet",
    "train_cpg-val_sample.parquet",
    "val_cpg-train_sample.parquet",
    "val_cpg-val_sample.parquet",
]


def main(_):
    input_dataset_dir_a = Path(FLAGS.input_dataset_dir_a)
    if FLAGS.input_dataset_dir_b is None:
        input_dataset_dir_b = input_dataset_dir_a
    else:
        input_dataset_dir_b = Path(FLAGS.input_dataset_dir_b)

    logging.info("Comparing length sum of datasets")
    logging.info(f"Dataset A: {input_dataset_dir_a}")
    logging.info(f"Dataset B: {input_dataset_dir_b}")

    for split_name in ME_CPG_BG_SPLIT_NAMES:
        input_dataset_split_a = input_dataset_dir_a / "shard_stats" / (split_name + ".csv")
        input_dataset_split_b = input_dataset_dir_b / "shard_stats" / (split_name + ".csv")
        df_a = pd.read_csv(input_dataset_split_a)
        df_b = pd.read_csv(input_dataset_split_b)

        logging.info(f"Compare Split: {split_name}")
        log_str_a = f"Dataset A (shape {df_a.shape}): length sum {df_a['length'].sum()}"
        log_str_b = f"Dataset B (shape {df_b.shape}): length sum {df_b['length'].sum()}"
        logging.info(log_str_a)
        logging.info(log_str_b)
        if df_a["length"].sum() != df_b["length"].sum():
            output_log_file_a = input_dataset_dir_a / "shard_stats" / f"{split_name}.compare.log"
            output_log_file_b = input_dataset_dir_b / "shard_stats" / f"{split_name}.compare.log"
            with open(output_log_file_a, "w") as f:
                f.write(log_str_a)
            with open(output_log_file_b, "w") as f:
                f.write(log_str_b)
            logging.warning(f"Length sum mismatch: {log_str_a} vs {log_str_b}")
            raise ValueError("Length sum mismatch")
        else:
            logging.info("Length sum match")


if __name__ == "__main__":
    app.run(main)
