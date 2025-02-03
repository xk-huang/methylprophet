"""
python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-BRCA,TCGA-LAML,TCGA-GBM \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx.csv

python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-BRCA \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx-brca.csv

python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-LAML \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx-laml.csv

python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-GBM \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx-gbm.csv
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_sample_idx_csv", None, "Path to the input sample index csv file.")

flags.DEFINE_list("filtered_tissue_names", None, "List of tissue names to filter the sample index by.")

flags.DEFINE_string("output_dir", None, "Path to the output directory.")
flags.DEFINE_string("output_filename", None, "Filename of the output file.")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output file if it already exists.")


FLAGS = flags.FLAGS


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / FLAGS.output_filename
    if output_path.exists():
        if FLAGS.overwrite:
            logging.warning(f"Overwriting existing file: {output_path}")
        else:
            logging.warning(f"Output file already exists: {output_path}")
            return

    input_sample_idx_csv = Path(FLAGS.input_sample_idx_csv)
    filtered_tissue_names = FLAGS.filtered_tissue_names

    logging.info(f"Reading sample index from {input_sample_idx_csv}")
    sample_idx_df = pd.read_csv(input_sample_idx_csv, index_col=0)

    logging.info(f"Filtering sample index by tissue names: {filtered_tissue_names}")
    filtered_sample_idx_df = sample_idx_df[sample_idx_df["tissue_name"].isin(filtered_tissue_names)]
    # stats number of samples per tissue
    logging.info("Number of samples per tissue:")
    logging.info(filtered_sample_idx_df["tissue_name"].value_counts())

    logging.info(f"Writing filtered sample index to {output_path}")
    filtered_sample_idx_df.to_csv(output_path)


if __name__ == "__main__":
    app.run(main)
