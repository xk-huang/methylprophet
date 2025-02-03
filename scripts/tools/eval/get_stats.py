"""
python scripts/tools/eval/get_stats.py \
    --input_parquet_dir data/me_parquets_after_mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/train_cpg-train_sample.parquet \
    --num_workers 40
"""

import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_dir", None, "Input directory")
flags.DEFINE_integer("num_workers", None, "Number of workers")
flags.DEFINE_bool("overwrite", False, "Overwrite output file")
FLAGS = flags.FLAGS


def process_file(file_path):
    df = pd.read_parquet(file_path)
    result = {
        "num_me": len(df),
        "cpg_idx": set(df["cpg_idx"].unique()),
        "sample_idx": set(df["sample_idx"].unique()),
        "tissue_idx": set(df["tissue_idx"].unique()),
    }
    return result


def merge_results(results):
    merged = {
        "num_me": 0,
        "cpg_idx": set(),
        "sample_idx": set(),
        "tissue_idx": set(),
    }

    for result in results:
        merged["num_me"] += result["num_me"]
        merged["cpg_idx"].update(result["cpg_idx"])
        merged["sample_idx"].update(result["sample_idx"])
        merged["tissue_idx"].update(result["tissue_idx"])

    return merged


def main(_):
    input_dir = Path(FLAGS.input_parquet_dir)
    parquet_files = sorted(input_dir.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {input_dir}")

    output_path = str(input_dir) + ".json"
    output_path = Path(output_path)
    if output_path.exists():
        if FLAGS.overwrite:
            logging.info(f"Overwriting {output_path}")
            output_path.unlink()
        else:
            logging.warning(f"Output file {output_path} already exists. Exiting...")
            return

    # Create a pool of workers
    num_cpus = mp.cpu_count() if FLAGS.num_workers is None else FLAGS.num_workers
    pool = mp.Pool(processes=num_cpus)

    # Process files in parallel with progress bar
    results = list(tqdm.tqdm(pool.imap(process_file, parquet_files), total=len(parquet_files)))

    # Close the pool
    pool.close()
    pool.join()

    # Merge results from all processes
    merged = merge_results(results)
    logging.info("Merged results")

    # Create final log dictionary
    log_dict = {
        "num_me": merged["num_me"],
        "num_cpg": len(merged["cpg_idx"]),
        "num_sample": len(merged["sample_idx"]),
        "num_tissue": len(merged["tissue_idx"]),
    }
    log_dict["num_cpg_sample_pair"] = log_dict["num_cpg"] * log_dict["num_sample"]

    # Save results
    output_path = str(input_dir) + ".json"
    with open(output_path, "w") as f:
        json.dump(log_dict, f, indent=4)


if __name__ == "__main__":
    app.run(main)
