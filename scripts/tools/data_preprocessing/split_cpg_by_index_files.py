import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from absl import app, flags, logging

flags.DEFINE_string("input_chr_df_parquet", None, "Path to the parquet file")
flags.mark_flag_as_required("input_chr_df_parquet")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output directory name")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_string("index_method", None, "Index method for splitting")
# "null", "all"
flags.mark_flag_as_required("index_method")
# NOTE xk: first include, then exclude
flags.DEFINE_list("include_index_files", None, "Include index files")
flags.DEFINE_list("exclude_index_files", None, "Exclude index files")

flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_bool("shuffle", False, "Shuffle the data before splitting")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")

FLAGS = flags.FLAGS


def main(_):
    input_chr_df_parquet = Path(FLAGS.input_chr_df_parquet)

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / FLAGS.output_file_name
    if output_path.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_path} already exists. Skipping...")
            return
        else:
            logging.warning(f"Overwriting {output_path}")
            output_path.unlink()

    logging.info(f"Reading {input_chr_df_parquet}")
    chr_df = pd.read_parquet(input_chr_df_parquet)

    index_method = FLAGS.index_method

    chr_pos = chr_df["chr_pos"]
    # check duplication
    if chr_pos.duplicated().any():
        raise ValueError("Duplicated chr_pos found")

    all_chr_pos = set(chr_pos)

    chr_pos_pool = None
    if index_method == "null":
        chr_pos_pool = set()
    elif index_method == "all":
        chr_pos_pool = set(chr_pos)
    else:
        raise ValueError(f"Invalid index method: {index_method}, should be 'null' or 'all'")

    # NOTE xk: first include, then exclude
    include_index_files = FLAGS.include_index_files
    if include_index_files is None:
        include_index_files = []
    for include_index_file in include_index_files:
        logging.info(f"Including {include_index_file}")
        with open(include_index_file) as f:
            include_chr_pos_pool = {line.strip() for line in f}

        # check the read chr_pos is subset of chr_pos
        if not include_chr_pos_pool.issubset(all_chr_pos):
            raise ValueError(f"Invalid chr_pos in {include_index_file}, not a subset of all chr_pos")

        # include the read chr_pos into the pool
        chr_pos_pool.update(include_chr_pos_pool)

    exclude_index_files = FLAGS.exclude_index_files
    if exclude_index_files is None:
        exclude_index_files = []
    for exclude_index_file in exclude_index_files:
        logging.info(f"Excluding {exclude_index_file}")
        with open(exclude_index_file) as f:
            exclude_chr_pos_pool = {line.strip() for line in f}

        # check the read chr_pos is subset of chr_pos
        if not exclude_chr_pos_pool.issubset(all_chr_pos):
            raise ValueError(f"Invalid chr_pos in {exclude_index_file}, not a subset of all chr_pos")

        # exclude the read chr_pos from the pool
        chr_pos_pool.difference_update(exclude_chr_pos_pool)

    selected_chr_df = chr_df[chr_df["chr_pos"].isin(chr_pos_pool)]
    logging.info(f"Selected {len(selected_chr_df)} rows from {len(chr_df)} rows")

    if FLAGS.shuffle:
        seed = FLAGS.seed
        logging.info(f"Shuffling with seed {seed}")
        rng = np.random.default_rng(seed)
        selected_chr_df = selected_chr_df.sample(frac=1, random_state=rng)
    else:
        logging.info("Not shuffling, sort by index")
        selected_chr_df.sort_index(inplace=True)

    selected_chr_df.to_parquet(output_path)
    logging.info(f"Saved to {output_path}")
    # logging.info(f"Saved val.parquet and train.parquet to {output_dir}")


if __name__ == "__main__":
    app.run(main)
