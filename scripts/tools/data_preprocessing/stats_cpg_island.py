"""
python scripts/tools/data_preprocessing/stats_cpg_island.py \
    --input_cpg_island_parquet=data/parquet/241213-encode_wgbs/cpg_island.parquet \
    --output_dir=data/parquet/241213-encode_wgbs/metadata/cpg_island_stats
"""

import json
import shutil
from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_cpg_island_parquet", None, "Path to input CpG island parquet file.")
flags.mark_flag_as_required("input_cpg_island_parquet")
flags.DEFINE_string("output_dir", None, "Path to output directory.")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output directory.")

FLAGS = flags.FLAGS


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, FLAGS.overwrite)

    cpg_island_df = pd.read_parquet(FLAGS.input_cpg_island_parquet)
    cpg_island_location_counter = cpg_island_df["location"].value_counts().sort_index()
    cpg_island_index_counter = cpg_island_df["cgiIndex"].value_counts().sort_index()

    json_dict = {
        "cpg_island_location_counter": cpg_island_location_counter.to_dict(),
        "cpg_island_index_counter": cpg_island_index_counter.to_dict(),
    }
    output_json_path = output_dir / "cpg_island_stats.json"
    with open(output_json_path, "w") as f:
        json.dump(json_dict, f, indent=4)
    logging.info(f"Save CpG island stats to {output_json_path}")


def prepare_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Remove existing output directory {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory {output_dir} already exists. Skip.")
            exit()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    app.run(main)
    app.run(main)
