from pathlib import Path

import pandas as pd
from absl import app, flags


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.DEFINE_alias("i", "input_parquet_file")
FLAGS = flags.FLAGS
flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")


def main(argv):
    input_parquet_file = Path(FLAGS.input_parquet_file)
    df = pd.read_parquet(input_parquet_file)
    print(df)

    if "sample_name" in df.columns:
        print(f"Number of sample_name: {df['sample_name'].unique().shape[0]}")
        print(df["sample_name"].unique())

    if "cpg_chr_pos" in df.columns:
        print(f"Number of cpg_chr_pos: {df['cpg_chr_pos'].unique().shape[0]}")
        print(df["cpg_chr_pos"].unique())

    if FLAGS.ipython:
        from IPython import embed

        embed()


if __name__ == "__main__":
    app.run(main)
