"""
python scripts/tools/get_num_tissues.py --i data/parquet/241231-tcga_array/metadata/subset_sample_split/sample_tissue_count_with_idx.csv
python scripts/tools/get_num_tissues.py --i data/parquet/241231-tcga_epic/metadata/subset_sample_split/sample_tissue_count_with_idx.csv
python scripts/tools/get_num_tissues.py --i data/parquet/241231-tcga_wgbs/metadata/subset_sample_split/sample_tissue_count_with_idx.csv
"""

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_csv", None, "Path to the input CSV file")
flags.mark_flag_as_required("input_csv")
flags.DEFINE_alias("i", "input_csv")


FLAGS = flags.FLAGS


def main(_):
    df = pd.read_csv(FLAGS.input_csv)
    num_tissues = len(df["tissue_idx"].unique())
    logging.info(f"Number of tissues: {num_tissues}")


if __name__ == "__main__":
    app.run(main)
