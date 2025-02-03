"""
Merge sample tissue count with idx files

python scripts/tools/data_preprocessing/merge_sample_tissue_count_with_idx.py \
    --input_sample_tissue_counts_iwth_idx_files data/parquet/241213-encode_wgbs/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_path misc/sample_tissue_count_with_idx.csv
"""

import shutil
from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_list("input_sample_tissue_counts_iwth_idx_files", None, "List of paths to sample tissue count files")
flags.mark_flag_as_required("input_sample_tissue_counts_iwth_idx_files")

flags.DEFINE_string("output_path", None, "Output path")
flags.mark_flag_as_required("output_path")

flags.DEFINE_bool("overwrite", False, "Overwrite existing output files")

FLAGS = flags.FLAGS


def main(_):
    output_path = prepare_output_path(FLAGS.output_path, FLAGS.overwrite)
    sample_tissue_count_with_idx_files = FLAGS.input_sample_tissue_counts_iwth_idx_files
    logging.info(f"sample_tissue_count_with_idx_files: {sample_tissue_count_with_idx_files}")

    sample_tissue_count_with_idx_dfs = []
    for file in sample_tissue_count_with_idx_files:
        _df = pd.read_csv(file, index_col=0)
        sample_tissue_count_with_idx_dfs.append(_df)

    df = pd.concat(sample_tissue_count_with_idx_dfs, axis=0, ignore_index=True)
    if df["sample_idx"].nunique() != df.shape[0]:
        raise ValueError(
            "Duplicate sample_idx found, thus cannot merge. "
            f"sample_idx nunique: {df['sample_idx'].nunique()}, "
            f"total length: {df.shape[0]}"
        )

    logging.info(f"save merged sample tissue count with idx to {output_path}")
    df.to_csv(output_path)


def prepare_output_path(output_path, overwrite):
    output_path = Path(output_path)
    if output_path.exists():
        if overwrite:
            logging.info(f"Remove existing output path: {output_path}")
            shutil.rmtree(output_path)
        else:
            logging.warning(f"Output path already exists: {output_path}")
            exit()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_path


if __name__ == "__main__":
    app.run(main)
