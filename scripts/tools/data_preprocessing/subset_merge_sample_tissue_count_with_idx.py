"""
python scripts/tools/data_preprocessing/subset_merge_sample_tissue_count_with_idx.py \
    --input_me_parquet_file data/parquet/241231-tcga_array/me.parquet/00000.parquet \
    --sample_tissue_count_with_idx_file data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_dir data/parquet/241231-tcga_array/metadata/ \
    --output_file_name subset_sample_split
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_me_parquet_file", None, "Path to the input ME parquet file")
flags.DEFINE_string("sample_tissue_count_with_idx_file", None, "Path to the sample tissue count file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output file name")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")


FLAGS = flags.FLAGS


def main(_):
    input_me_parquet_file = Path(FLAGS.input_me_parquet_file)
    sample_tissue_count_with_idx_file = Path(FLAGS.sample_tissue_count_with_idx_file)

    output_dir = Path(FLAGS.output_dir)
    output_file_name = FLAGS.output_file_name
    output_dir = output_dir / output_file_name

    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_dir} already exists. Skipping...")
            return
        else:
            logging.warning(f"Overwriting {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Reading {input_me_parquet_file}")
    df = pd.read_parquet(input_me_parquet_file)

    logging.info(f"Reading {sample_tissue_count_with_idx_file}")
    sample_tissue_count_with_idx_df = pd.read_csv(sample_tissue_count_with_idx_file, index_col=0, na_filter=False)

    # Get the sample names
    columns = df.columns[1:]  # NOTE: Remove the first column, "Unnamed: 0", which is the cpg_chr_pos

    # get the rows from the sample_tissue_count_with_idx_df with me columns
    subset_sample_tissue_count_with_idx_condition = sample_tissue_count_with_idx_df["sample_name"].isin(columns)
    if len(columns) != subset_sample_tissue_count_with_idx_condition.sum():
        raise ValueError(
            f"Columns in the input ME parquet file and the sample_tissue_count_with_idx_file are not equal: {len(columns)} != {subset_sample_tissue_count_with_idx_condition.sum()}"
        )
    subset_sample_tissue_count_with_idx_df = sample_tissue_count_with_idx_df[
        subset_sample_tissue_count_with_idx_condition
    ]

    output_file = output_dir / "sample_tissue_count_with_idx.csv"
    subset_sample_tissue_count_with_idx_df.to_csv(output_file)


if __name__ == "__main__":
    app.run(main)
