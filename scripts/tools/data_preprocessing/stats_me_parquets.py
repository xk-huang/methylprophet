"""
ME_DIR=data/parquet/241213-encode_wgbs
python scripts/tools/data_preprocessing/stats_me_parquets.py \
    --i "${ME_DIR}/me.parquet" \
    --output_dir "${ME_DIR}/metadata/me_stats"
"""

import shutil
from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")
flags.DEFINE_alias("i", "input_parquet_file")
flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")
flags.DEFINE_boolean("verbose", False, "Enable IPython at the end of the script")
flags.DEFINE_list("columns", None, "List of columns to display")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_bool("overwrite", False, "Overwrite output directory")

FLAGS = flags.FLAGS


def main(argv):
    output_dir = Path(FLAGS.output_dir)
    if output_dir.exists():
        if FLAGS.overwrite:
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory already exists: {output_dir}, skipping")
            return
    output_dir.mkdir(parents=True, exist_ok=True)

    input_parquet_file = Path(FLAGS.input_parquet_file)
    logging.info(f"Reading parquet file: {input_parquet_file}")
    df = pd.read_parquet(input_parquet_file, columns=FLAGS.columns)
    logging.info(df)

    # rename "Unnamed: 0" column to "cpg_chr_pos", and set it as index
    df.rename(columns={"Unnamed: 0": "cpg_chr_pos"}, inplace=True)
    df.set_index("cpg_chr_pos", inplace=True)

    num_cpg = df.shape[0]
    num_samples = df.shape[1]

    num_cpg_sample_pairs = num_cpg * num_samples
    num_pair_w_me = df.notna().sum().sum()
    num_pair_wo_me = num_cpg_sample_pairs - num_pair_w_me

    me_stats_dict = {
        "num_cpg": num_cpg,
        "num_samples": num_samples,
        "num_cpg_sample_pairs": num_cpg_sample_pairs,
        "num_pair_w_me": num_pair_w_me,
        "num_pair_wo_me": num_pair_wo_me,
    }
    me_stats_df = pd.DataFrame(me_stats_dict, index=[str(input_parquet_file)])
    me_stats_output_path = output_dir / "me_stats.tsv"
    logging.info(f"Writing ME stats to: {me_stats_output_path}")
    me_stats_df.to_csv(me_stats_output_path, sep="\t")


if __name__ == "__main__":
    app.run(main)
