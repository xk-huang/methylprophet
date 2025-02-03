"""
python scripts/tools/data_preprocessing/reshard_me_cpg_bg_parquets.py \
    --i data/processed/241023-tcga_array-train_0_9_val_0_1-ind_cancer-nan \
    --num_workers 4
"""

import math
import multiprocessing as mp
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from absl import app, flags, logging

flags.DEFINE_string("input_me_cpg_bg_dataset_dir", None, "Path to the directory with the ME CpG BG dataset")
flags.mark_flag_as_required("input_me_cpg_bg_dataset_dir")
flags.DEFINE_alias("i", "input_me_cpg_bg_dataset_dir")
flags.DEFINE_string("output_me_cpg_bg_dataset_dir", None, "Path to the output directory")
flags.DEFINE_alias("o", "output_me_cpg_bg_dataset_dir")

flags.DEFINE_integer("chunk_size", None, "Desired number of rows per output shard")
flags.DEFINE_integer("num_workers", 1, "Number of worker processes")
flags.DEFINE_bool("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS

DUPLICATED_GENE_ID_CSV_FILE_NAME = "duplicated_gene_id.csv"
GENE_EXPR_FILTERED_PARQUET_FILE_NAME = "gene_expr.filtered.parquet"
ME_CPG_BG_SPLIT_NAMES = [
    "train_cpg-train_sample.parquet",
    "train_cpg-val_sample.parquet",
    "val_cpg-train_sample.parquet",
    "val_cpg-val_sample.parquet",
]


def process_split(args):
    """Process a single split."""
    (
        input_me_cpg_bg_dataset_dir,
        output_dir,
        me_cpg_bg_split_name,
        chunk_size,
        debug,
    ) = args

    logging.info(f"Processing split {me_cpg_bg_split_name}")

    input_me_cpg_bg_split = input_me_cpg_bg_dataset_dir / "me_cpg_bg" / me_cpg_bg_split_name
    if not input_me_cpg_bg_split.exists():
        raise ValueError(f"ME CpG BG split {input_me_cpg_bg_split} does not exist")

    output_me_cpg_bg_split = output_dir / "me_cpg_bg" / me_cpg_bg_split_name
    output_me_cpg_bg_split.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output split directory: {output_me_cpg_bg_split}")

    input_me_cpg_bg_split_parquets = sorted(input_me_cpg_bg_split.glob("*.parquet"))

    # Read shard stats to get the total number of rows
    shard_stats_df_path = input_me_cpg_bg_dataset_dir / "shard_stats" / f"{me_cpg_bg_split_name}.csv"
    shard_stats_df = pd.read_csv(shard_stats_df_path)
    total_rows = shard_stats_df["length"].sum()

    if chunk_size is None:
        logging.info("Chunk size is not provided, calculating the chunk size")
        num_input_shards = len(shard_stats_df)
        if num_input_shards != len(input_me_cpg_bg_split_parquets):
            raise ValueError(
                f"Number of shards {num_input_shards} != number of parquets {len(input_me_cpg_bg_split_parquets)}"
            )
        chunk_size = math.ceil(total_rows / num_input_shards)
        if chunk_size == 0:
            raise ValueError("Chunk size is 0, decrease the number of shards")

    num_output_shards = math.ceil(total_rows / chunk_size)
    logging.info(
        f"Total rows: {total_rows}, Desired chunk size: {chunk_size}, Number of output shards: {num_output_shards}"
    )

    # Initialize variables for processing
    buffer = []
    buffer_size = 0
    output_shard_idx = 0

    # Define the schema from the first file
    first_parquet_file = input_me_cpg_bg_split_parquets[0]
    # NOTE xk: save with index=False, as we do not need ("__index_level_0__").
    schema = pq.ParquetFile(first_parquet_file).schema_arrow

    # Iterate over input Parquet files
    pbar = tqdm.tqdm(input_me_cpg_bg_split_parquets, desc="Processing")
    for input_parquet in pbar:
        pbar.set_description_str(f"Reading: {input_parquet.name}")
        parquet_file = pq.ParquetFile(input_parquet)

        # Read data in batches
        for batch in parquet_file.iter_batches():
            buffer.append(batch)
            buffer_size += len(batch)

            # When buffer reaches chunk_size, write it out
            while buffer_size >= chunk_size:
                # Concatenate the batches in the buffer
                table = pa.Table.from_batches(buffer, schema=schema)

                # Write to Parquet
                output_parquet_path = output_me_cpg_bg_split / f"{output_shard_idx:05d}.parquet"
                pq.write_table(table.slice(0, chunk_size), output_parquet_path)
                pbar.set_postfix_str(f"Writing: {output_parquet_path.name}")

                # Update the buffer with leftover rows
                remaining_rows = buffer_size - chunk_size
                if remaining_rows > 0:
                    # Keep leftover rows in the buffer
                    buffer = table.slice(chunk_size).to_batches()
                else:
                    buffer = []
                buffer_size = remaining_rows
                output_shard_idx += 1

                if debug and output_shard_idx >= 2:
                    logging.info("Debug mode: stopping after writing two shards")
                    return
        pbar.update()

    # Write any remaining data in the buffer
    if buffer_size > 0:
        table = pa.Table.from_batches(buffer, schema=schema)
        output_parquet_path = output_me_cpg_bg_split / f"{output_shard_idx:05d}.parquet"
        pq.write_table(table, output_parquet_path)
        logging.info(f"Written final output shard: {output_parquet_path} with {buffer_size} rows")

    logging.info(f"Completed processing split {me_cpg_bg_split_name} for {input_me_cpg_bg_dataset_dir}")


def main(_):
    input_me_cpg_bg_dataset_dir = Path(FLAGS.input_me_cpg_bg_dataset_dir)
    if not input_me_cpg_bg_dataset_dir.exists():
        logging.fatal(f"Input ME CpG BG dataset directory {input_me_cpg_bg_dataset_dir} does not exist")
    logging.info(f"Reading ME CpG BG dataset from {input_me_cpg_bg_dataset_dir}")

    # Prepare output_dir
    output_dir = FLAGS.output_me_cpg_bg_dataset_dir
    if output_dir is None:
        output_dir = input_me_cpg_bg_dataset_dir.parent / f"{input_me_cpg_bg_dataset_dir.name}-resharded"
    else:
        output_dir = Path(output_dir)

    if output_dir.exists():
        if FLAGS.overwrite:
            logging.warning(f"Output directory {output_dir} exists, overwriting")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} exists, set overwrite to True to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Copy gene related files to the output directories
    for file_name in [DUPLICATED_GENE_ID_CSV_FILE_NAME, GENE_EXPR_FILTERED_PARQUET_FILE_NAME]:
        shutil.copy(input_me_cpg_bg_dataset_dir / file_name, output_dir / file_name)

    # Prepare arguments for multiprocessing
    args_list = []
    for me_cpg_bg_split_name in ME_CPG_BG_SPLIT_NAMES:
        args = (
            input_me_cpg_bg_dataset_dir,
            output_dir,
            me_cpg_bg_split_name,
            FLAGS.chunk_size,
            FLAGS.debug,
        )
        args_list.append(args)

    # Use multiprocessing to process splits in parallel
    if FLAGS.num_workers > 1:
        with mp.Pool(FLAGS.num_workers) as pool:
            pool.map(process_split, args_list)
    else:
        for args in args_list:
            process_split(args)


if __name__ == "__main__":
    app.run(main)
