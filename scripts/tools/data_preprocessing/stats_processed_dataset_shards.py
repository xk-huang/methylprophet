import json
import multiprocessing as mp
import shutil
from pathlib import Path
from pprint import pformat

import pandas as pd
import pyarrow.parquet as pq
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_processed_dataset_dir", None, "Path to the input processed dataset dir")
flags.mark_flag_as_required("input_processed_dataset_dir")
flags.DEFINE_alias("i", "input_processed_dataset_dir")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("output_name", "shard_stats", "Output name")

flags.DEFINE_integer("num_workers", 8, "Number of worker processes for parallel processing")
flags.DEFINE_bool("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS


def get_parquet_length(file_path):
    parquet_file = pq.ParquetFile(file_path)
    length = parquet_file.metadata.num_rows
    # pd_length = len(pd.read_parquet(file_path))
    # if length != pd_length:
    #     raise ValueError(f"Length mismatch: {length} != {pd_length}")
    return {"name": file_path.name, "length": length}


def get_length_of_parquet_files_in_dir(dir_path: Path, num_workers=8, debug=False):
    parquet_files = sorted(dir_path.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {dir_path}")

    if debug:
        parquet_files_stats = []
        for parquet_file in tqdm.tqdm(parquet_files, desc="Processing files"):
            parquet_files_stats.append(get_parquet_length(parquet_file))
            break
        return pd.DataFrame(parquet_files_stats)

    with mp.Pool(processes=num_workers) as pool:
        parquet_files_stats = list(
            tqdm.tqdm(
                pool.imap(get_parquet_length, parquet_files), total=len(parquet_files), desc="Getting parquet length"
            )
        )

    return pd.DataFrame(parquet_files_stats)


def get_sample_idx_cpg_idx_from_parquet_file(parquet_file):
    cpg_idx_sample_idx_df = pd.read_parquet(parquet_file, columns=["cpg_idx", "sample_idx"])
    return cpg_idx_sample_idx_df


def get_cpg_idx_sample_idx_from_parquet_files(dir_path: Path, num_workers=8, debug=False):
    parquet_files = sorted(dir_path.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {dir_path}")

    if debug:
        cpg_id_sample_id_list = []
        for parquet_file in tqdm.tqdm(parquet_files, desc="Processing files"):
            cpg_id_sample_id = get_sample_idx_cpg_idx_from_parquet_file(parquet_file)
            cpg_id_sample_id_list.append(cpg_id_sample_id)
            break
        return pd.concat(cpg_id_sample_id_list)

    with mp.Pool(processes=num_workers) as pool:
        cpg_id_sample_id_list = list(
            tqdm.tqdm(
                pool.imap(get_sample_idx_cpg_idx_from_parquet_file, parquet_files),
                total=len(parquet_files),
                desc="Getting cpg_idx_sample_idx",
            )
        )

    return pd.concat(cpg_id_sample_id_list)


def main(_):
    input_processed_dataset_dir = Path(FLAGS.input_processed_dataset_dir)

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

    logging.info(f"Reading {input_processed_dataset_dir}")

    me_cpg_bg_dir = input_processed_dataset_dir / "me_cpg_bg"
    split_dir_list = sorted(me_cpg_bg_dir.glob("*/"))

    logging.info(f"Found {len(split_dir_list)} split directories:\n{pformat(split_dir_list)}")

    num_workers = FLAGS.num_workers
    debug = FLAGS.debug

    stats = {}
    for split_dir in split_dir_list:
        split_name = split_dir.name
        logging.info(f"Processing {split_name}")
        stats[split_name] = get_length_of_parquet_files_in_dir(split_dir, num_workers, debug)

    logging.info(f"Saving stats to {output_dir}")
    for split_name, df in stats.items():
        output_path = output_dir / f"{split_name}.csv"
        df.to_csv(output_path)
        logging.info(f"Saved {output_path}")

    metainfo_stats = {}
    for split_dir in split_dir_list:
        split_name = split_dir.name
        df = stats[split_name]

        num_shards = len(df)
        num_cpg_sample_pairs = int(df["length"].sum())

        cpg_idx_sample_idx_df = get_cpg_idx_sample_idx_from_parquet_files(split_dir, num_workers, debug)
        if len(cpg_idx_sample_idx_df) != num_cpg_sample_pairs:
            no_match_file_path = output_dir / f"{split_name}_no_match.log"
            no_match_str = (
                f"Number of cpg_idx_sample_idx_df ({len(cpg_idx_sample_idx_df)}) "
                f"does not match num_cpg_sample_pairs ({num_cpg_sample_pairs})"
            )
            with open(no_match_file_path, "w") as f:
                f.write(no_match_str)
            logging.warning(no_match_str)

        num_sample = cpg_idx_sample_idx_df["sample_idx"].nunique()
        num_cpg = cpg_idx_sample_idx_df["cpg_idx"].nunique()
        estimated_num_cpg_sample_pairs = num_sample * num_cpg
        if num_cpg_sample_pairs != estimated_num_cpg_sample_pairs:
            no_match_file_path = output_dir / f"{split_name}_no_match_estimation.log"
            no_match_str = (
                f"Number of cpg_sample_pairs ({num_cpg_sample_pairs}) does not match "
                f"estimated number of cpg_sample_pairs ({estimated_num_cpg_sample_pairs})"
            )
            with open(no_match_file_path, "w") as f:
                f.write(no_match_str)
            logging.warning(no_match_str)

        metainfo_stats[split_name] = {
            "num_shards": num_shards,
            "num_cpg_sample_pairs": num_cpg_sample_pairs,
            "num_sample": num_sample,
            "num_cpg": num_cpg,
        }

    metainfo_stats_path = output_dir / "shard_stats.json"
    with open(metainfo_stats_path, "w") as f:
        json.dump(metainfo_stats, f, indent=4)
    logging.info(f"Saved {metainfo_stats_path}.")


if __name__ == "__main__":
    app.run(main)
