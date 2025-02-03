import multiprocessing as mp
import shutil
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_processed_dataset_dir", None, "Path to the input processed dataset dir")
flags.mark_flag_as_required("input_processed_dataset_dir")
flags.DEFINE_alias("i", "input_processed_dataset_dir")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("output_name", "me_std_stats", "Output name")

flags.DEFINE_integer("num_workers", 8, "Number of worker processes for parallel processing")
flags.DEFINE_bool("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS


def main(_):
    input_processed_dataset_dir = Path(FLAGS.input_processed_dataset_dir)
    logging.info(f"Reading {input_processed_dataset_dir}")

    me_cpg_bg_dir = input_processed_dataset_dir / "me_cpg_bg"
    split_dir_list = sorted(me_cpg_bg_dir.glob("*/"))

    logging.info(f"Found {len(split_dir_list)} split directories:\n{pformat(split_dir_list)}")

    output_dir = FLAGS.output_dir
    if output_dir is None:
        output_dir = input_processed_dataset_dir
    else:
        output_dir = Path(FLAGS.output_dir)
    output_dir = output_dir / FLAGS.output_name
    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_dir} already exists. Skipping...")
            return
        else:
            shutil.rmtree(output_dir)
            logging.warning(f"Overwriting {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    num_workers = FLAGS.num_workers
    debug = FLAGS.debug

    for split_dir in split_dir_list:
        compute_methylation_statistics(output_dir, num_workers, debug, split_dir)


def compute_methylation_statistics(output_dir, num_workers, debug, split_dir):
    split_name = split_dir.name
    logging.info(f"Processing {split_name}")
    me_cpg_idx_sample_idx_df = get_std_me_by_cpg_id_parquet_files_in_dir(split_dir, num_workers, debug)

    logging.info(f"Computing std by cpg_id for {split_name}")
    me_std_by_cpg_id = me_cpg_idx_sample_idx_df.groupby("cpg_idx")["methylation"].std()
    logging.info(f"Std by cpg_id for {split_name} computed")

    logging.info(f"Computing std by sample_id for {split_name}")
    me_std_by_sample_id = me_cpg_idx_sample_idx_df.groupby("sample_idx")["methylation"].std()
    logging.info(f"Std by sample_id for {split_name} computed")

    me_std_by_cpg_id.to_csv(output_dir / f"{split_name}_me_std_by_cpg_id.csv")
    me_std_by_sample_id.to_csv(output_dir / f"{split_name}_me_std_by_sample_id.csv")
    plot_stats(me_std_by_cpg_id, output_dir, split_name, "cpg_id")
    plot_stats(me_std_by_sample_id, output_dir, split_name, "sample_id")


def plot_stats(me_std, output_dir, split_name, group_name=""):
    sns.histplot(me_std)
    plt.title(f"{split_name} me std by {group_name}")
    plt.savefig(output_dir / f"{split_name}-by_{group_name}-me_std_by_cpg_id.png")
    plt.cla()


def get_std_me_by_cpg_id_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df[["cpg_idx", "sample_idx", "methylation"]]


def get_std_me_by_cpg_id_parquet_files_in_dir(dir_path: Path, num_workers=8, debug=False):
    parquet_files = sorted(dir_path.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {dir_path}")

    if debug:
        me_df = []
        for idx, parquet_file in enumerate(tqdm.tqdm(parquet_files, desc="Processing files")):
            if idx > 2:
                break
            me_df.append(get_std_me_by_cpg_id_parquet(parquet_file))
        breakpoint()
        me_df = pd.concat(me_df, axis=0)
        me_df.reset_index(drop=True, inplace=True)
        return me_df

    with mp.Pool(processes=num_workers) as pool:
        me_df = list(
            tqdm.tqdm(
                pool.imap(get_std_me_by_cpg_id_parquet, parquet_files),
                total=len(parquet_files),
                desc="Getting me, cpg_idx, sample_idx",
            )
        )
    me_df = pd.concat(me_df, axis=0)
    me_df.reset_index(drop=True, inplace=True)
    return me_df


if __name__ == "__main__":
    app.run(main)
