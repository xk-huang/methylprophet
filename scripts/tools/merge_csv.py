"""
Merge to csv files

python scripts/tools/merge_csv.py \
    --input_files data/parquet/241231-tcga_epic/metadata/check_nan/gene_expr/nan_count_per_col.csv,data/parquet/241231-tcga_epic/metadata/check_nan/me/nan_count_per_col.csv \
    --output_file misc/250108-tcga_epic-num_nan_per_cols.csv \
    --index_col 0 \
    --overwrite
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_list("input_files", None, "List of input files")
flags.DEFINE_string("output_file", None, "Output file")
flags.DEFINE_bool("overwrite", False, "Overwrite output file")
flags.DEFINE_integer("index_col", None, "Column index to merge on")

FLAGS = flags.FLAGS


def main(_):
    if not FLAGS.input_files:
        raise ValueError("input_files is required")
    if not FLAGS.output_file:
        raise ValueError("output_file is required")

    input_files = [Path(f) for f in FLAGS.input_files]
    output_file = Path(FLAGS.output_file)
    output_dir = output_file.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if output_file.exists() and not FLAGS.overwrite:
        raise ValueError(f"Output file {output_file} already exists")

    dfs = [pd.read_csv(f, index_col=FLAGS.index_col) for f in input_files]
    # check dfs have the same columns
    df = pd.concat(dfs, axis=1)
    df.to_csv(output_file)
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    app.run(main)
