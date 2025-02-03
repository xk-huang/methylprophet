"""
python scripts/tools/data_preprocessing/save_unique_idx_for_double_check_mp.py \
    --input_processed_parquet_dir data/processed/241231-tcga_array-index_files-ind_cancer/me_cpg_bg/train_cpg-val_sample.parquet \
    --output_dir data/processed/241231-tcga_array-index_files-ind_cancer/unique_idx_for_double_check/train_cpg-val_sample.parquet \
    --overwrite --num_workers=20
"""

import multiprocessing as mp
import shutil
from functools import partial
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_processed_parquet_dir", None, "Input processed parquet directory")
flags.mark_flag_as_required("input_processed_parquet_dir")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_bool("overwrite", False, "Overwrite output directory if exists")

flags.DEFINE_integer("num_workers", None, "Number of worker processes. Defaults to CPU count.")


def prepare_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            logging.info(f"Removed existing output directory {output_dir}")
        else:
            logging.warning(f"Output directory already exists: {output_dir}")
            exit()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_parquet_file(file_path, target_columns):
    df = pd.read_parquet(file_path)
    return df[target_columns]


def combine_dataframes(dfs):
    combined_df = pd.concat(dfs)
    return combined_df.drop_duplicates()


FLAGS = flags.FLAGS


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, FLAGS.overwrite)
    num_workers = FLAGS.num_workers or mp.cpu_count()

    input_processed_parquet_dir = Path(FLAGS.input_processed_parquet_dir)
    input_processed_parquet_list = sorted(input_processed_parquet_dir.glob("*.parquet"))

    target_cpg_columns = ["cpg_idx", "cpg_chr_pos"]
    target_sample_columns = ["sample_idx", "sample_name"]

    # Process CpG data
    logging.info("Processing CpG data...")
    with mp.Pool(num_workers) as pool:
        process_func = partial(process_parquet_file, target_columns=target_cpg_columns)
        cpg_dfs = list(
            tqdm.tqdm(pool.imap(process_func, input_processed_parquet_list), total=len(input_processed_parquet_list))
        )

    cpg_idx_name_df = combine_dataframes(cpg_dfs)
    output_cpg_idx_name_file = output_dir / "cpg_idx_name.parquet"
    cpg_idx_name_df.to_parquet(output_cpg_idx_name_file)
    logging.info(f"Saved {output_cpg_idx_name_file}")

    # Process sample data
    logging.info("Processing sample data...")
    with mp.Pool(num_workers) as pool:
        process_func = partial(process_parquet_file, target_columns=target_sample_columns)
        sample_dfs = list(
            tqdm.tqdm(pool.imap(process_func, input_processed_parquet_list), total=len(input_processed_parquet_list))
        )

    sample_idx_name_df = combine_dataframes(sample_dfs)
    output_sample_idx_name_file = output_dir / "sample_idx_name.parquet"
    sample_idx_name_df.to_parquet(output_sample_idx_name_file)
    logging.info(f"Saved {output_sample_idx_name_file}")


if __name__ == "__main__":
    app.run(main)
