from pathlib import Path

import pandas as pd
from absl import app, flags, logging

flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")
flags.DEFINE_alias("i", "input_parquet_file")
flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")
flags.DEFINE_boolean("verbose", False, "Enable IPython at the end of the script")
flags.DEFINE_list("columns", None, "List of columns to display")

FLAGS = flags.FLAGS


def main(argv):
    input_parquet_file = Path(FLAGS.input_parquet_file)
    logging.info(f"Reading parquet file: {input_parquet_file}")
    df = pd.read_parquet(input_parquet_file, columns=FLAGS.columns)
    logging.info(df)

    if FLAGS.ipython:
        from IPython import embed

        embed()

    if FLAGS.verbose:
        if "sample_name" in df.columns:
            print(f"Number of sample_name: {df['sample_name'].unique().shape[0]}")
            logging.info(df["sample_name"].unique())

        if "cpg_chr_pos" in df.columns:
            print(f"Number of cpg_chr_pos: {df['cpg_chr_pos'].unique().shape[0]}")
            logging.info(df["cpg_chr_pos"].unique())


if __name__ == "__main__":
    app.run(main)
