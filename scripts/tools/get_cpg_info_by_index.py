from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_cpg_bg_dir", None, "Path to cpg bg parquet file")
flags.mark_flag_as_required("input_cpg_bg_dir")
flags.DEFINE_alias("d", "input_cpg_bg_dir")

flags.DEFINE_integer("cpg_id", None, "CpG ID to get the information")
flags.mark_flag_as_required("cpg_id")
flags.DEFINE_alias("i", "cpg_id")

FLAGS = flags.FLAGS


def main(argv):
    input_cpg_bg_dir = Path(FLAGS.input_cpg_bg_dir)
    logging.info(f"Reading cpg bg parquet {input_cpg_bg_dir}")
    cpg_bg_files = sorted(input_cpg_bg_dir.glob("*.parquet"))

    chunk_size = pd.read_parquet(cpg_bg_files[0]).shape[0]
    logging.info(f"chunk size: {chunk_size}")

    cpg_id = FLAGS.cpg_id
    shard_id = FLAGS.cpg_id // chunk_size
    logging.info(f"cpg id: {cpg_id}, shard id: {shard_id}")

    cpg_bg_file = cpg_bg_files[shard_id]
    logging.info(f"Reading cpg bg file: {cpg_bg_file}")
    cpg_bg_df = pd.read_parquet(cpg_bg_file)
    cpg_info = cpg_bg_df.loc[cpg_id]

    logging.info(f"CpG info for cpg id {cpg_id}:\n{cpg_info}")


if __name__ == "__main__":
    app.run(main)
