"""
python scripts/tools/eval/subset_cpg_split.py \
    --input_cpg_split_parquet data/parquet/241231-tcga_array/metadata/cpg_split/index_files/train.parquet \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis-cpg_split/ \
    --output_filename train.parquet

python scripts/tools/eval/subset_cpg_split.py \
    --input_cpg_split_parquet data/parquet/241231-tcga_array/metadata/cpg_split/index_files/val.parquet \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis-cpg_split/ \
    --output_filename val.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_cpg_split_parquet", None, "Path to the input cpg split csv file.")

flags.DEFINE_integer("num_cpg", 10000, "Number of CpG sites to use.")
flags.DEFINE_integer("seed", 42, "Random seed.")
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

    input_cpg_split_parquet = Path(FLAGS.input_cpg_split_parquet)

    logging.info(f"Reading cpg split from {input_cpg_split_parquet}")
    cpg_split_df = pd.read_parquet(input_cpg_split_parquet)
    logging.info(f"Read {len(cpg_split_df)} CpG sites")

    num_cpg = FLAGS.num_cpg
    # randomly select num_cpg CpG sites
    rng = np.random.default_rng(FLAGS.seed)
    random_idx = rng.choice(len(cpg_split_df), num_cpg, replace=False)
    if len(set(random_idx)) != num_cpg:
        raise ValueError("Randomly selected duplicate indices")

    cpg_split_df = cpg_split_df.iloc[random_idx]
    logging.info(f"Selected {len(cpg_split_df)} CpG sites")

    logging.info(f"Writing filtered cpg split to {output_path}")
    cpg_split_df.to_parquet(output_path)


if __name__ == "__main__":
    app.run(main)
