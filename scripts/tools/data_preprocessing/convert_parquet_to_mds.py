"""
python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/me_cpg_bg/val_cpg-train_sample.parquet \
    --output_dir data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/debug.mds \
    --overwrite --debug --remove_unused_columns --tokenize_sequence
"""

import gc
import json
import math
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Optional, Tuple

import psutil
import pyarrow.parquet as pq
import tqdm
from absl import app, flags, logging
from omegaconf import OmegaConf
from streaming import MDSWriter
from streaming.base.util import merge_index

from src.data.constants import CHR_IDX_MAPPING
from transformers import AutoTokenizer


FLAGS = flags.FLAGS

flags.DEFINE_list("input_parquet_dir_list", None, "List of input parquet directories")
flags.mark_flag_as_required("input_parquet_dir_list")
flags.DEFINE_list("num_shards_list", None, "List of number of shards for each input parquet directory")
flags.DEFINE_bool("remove_unused_columns", False, "Remove unused columns")


flags.DEFINE_integer("num_workers", 1, "Number of job workers, split the whole data into parts")
flags.DEFINE_integer("num_mp_workers", None, "Number of workers for multiprocessing")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")
flags.DEFINE_bool("debug", False, "Debug mode")

flags.DEFINE_list("filter_by_chr", None, "Filter by chromosome")

flags.DEFINE_bool("tokenize_sequence", False, "Tokenize sequence")
flags.DEFINE_string("dna_tokenizer_name", "zhihan1996/DNABERT-2-117M", "DNA tokenizer name")
flags.DEFINE_integer("num_nbase", 1000, "Number of bases for DNA tokenizer")
flags.DEFINE_integer("batch_size", 10000, "Batch size to iter over parquet files")


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, FLAGS.overwrite)
    logging.info(f"Output directory: {output_dir}")
    num_workers = FLAGS.num_workers
    logging.info(f"Number of workers: {num_workers}")
    num_shards_list = FLAGS.num_shards_list
    if num_shards_list is None:
        num_shards_list = [None] * len(FLAGS.input_parquet_dir_list)
    else:
        num_shards_list = list(map(int, num_shards_list))
    if len(num_shards_list) != len(FLAGS.input_parquet_dir_list):
        raise ValueError(
            f"Number of input parquet directories and number of shards list should be equal, {len(FLAGS.input_parquet_dir_list)} vs {len(num_shards_list)}"
        )
    logging.info(f"Number of shards list: {num_shards_list}")

    # Write group_idx_to_input_parquet_dir.json
    group_idx_to_input_parquet_dir = {}
    for group_idx, input_parquet_dir in enumerate(FLAGS.input_parquet_dir_list):
        group_idx_to_input_parquet_dir[group_idx] = input_parquet_dir
    with open(output_dir / "group_idx_name_mapping.json", "w") as f:
        json.dump(group_idx_to_input_parquet_dir, f, indent=4)

    remove_unused_columns = FLAGS.remove_unused_columns
    logging.info(f"Remove unused columns: {remove_unused_columns}")

    filter_by_chr = FLAGS.filter_by_chr
    logging.info(f"Filter by chromosome: {filter_by_chr}")

    tokenize_sequence = FLAGS.tokenize_sequence
    dna_tokenizer_name = FLAGS.dna_tokenizer_name
    num_nbase = FLAGS.num_nbase
    logging.info(f"Tokenize sequence: {tokenize_sequence}")
    if tokenize_sequence:
        logging.info(f"dna_tokenizer_name: {dna_tokenizer_name}, num_nbase: {num_nbase}")

    config = OmegaConf.create(FLAGS.flag_values_dict())
    for group_idx, (input_parquet_dir, num_shards) in enumerate(zip(FLAGS.input_parquet_dir_list, num_shards_list)):
        logging.info(f"Processing {input_parquet_dir}")
        write_mds_one_parquet_dir_mp(
            input_parquet_dir,
            group_idx,
            output_dir,
            num_shards,
            config,
        )

    merge_index(str(output_dir), keep_local=True)


# Initialize the worker process
def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f"\nInitialize Worker PID: {pid}", flush=True, end="")


def get_arg_tuple_iter(
    input_parquet_dir: Path,
    group_idx: int,
    output_dir: Path,
    num_shards: int,
    config,
) -> Iterator[Tuple[Tuple[Path], int, Path]]:
    num_workers: int
    num_shards: Optional[int]

    num_workers = config.num_workers

    input_parquet_list = sorted(input_parquet_dir.glob("*.parquet"))
    logging.info(f"Number of input parquet files for {input_parquet_dir}: {len(input_parquet_list)}")
    if num_shards is not None:
        input_parquet_list = input_parquet_list[:num_shards]
        logging.info(f"Number of input parquet files after slicing: {len(input_parquet_list)}")

    if len(input_parquet_list) < num_workers:
        raise ValueError(
            f"Number of input parquet files is less than number of workers: {len(input_parquet_list)} vs {num_workers}, decrease number of workers"
        )

    num_shards_per_worker = math.ceil(len(input_parquet_list) / num_workers)
    logging.info(f"Number of rows per worker: {num_shards_per_worker}")

    for worker_idx, i in enumerate(range(0, len(input_parquet_list), num_shards_per_worker)):
        output_dir_per_worker = output_dir / f"group_{group_idx:05d}-worker_{worker_idx:05d}"
        yield (
            input_parquet_list[i : i + num_shards_per_worker],
            group_idx,
            output_dir_per_worker,
            config,
        )


def write_mds_one_parquet_dir_mp(
    input_parquet_dir,
    group_idx,
    output_dir,
    num_shards,
    config,
):
    input_parquet_dir = Path(input_parquet_dir)
    output_dir = Path(output_dir)

    arg_tuple_iter = get_arg_tuple_iter(
        input_parquet_dir,
        group_idx,
        output_dir,
        num_shards,
        config,
    )
    # Process group of data in parallel into directories of shards.
    num_workers = config.num_workers
    num_mp_workers = config.num_mp_workers
    if num_mp_workers is None:
        num_mp_workers = num_workers
    if num_mp_workers > num_workers:
        raise ValueError(
            f"Number of multiprocessing workers is greater than number of job workers: {num_mp_workers} > {num_workers}"
        )

    if num_workers == 1:
        for args in arg_tuple_iter:
            write_mds_one_parquet(args)
    else:
        with Pool(initializer=init_worker, processes=num_workers) as pool:
            for _ in pool.imap(write_mds_one_parquet, arg_tuple_iter):
                pass
    logging.info(f"Finished processing {input_parquet_dir}")


USED_COLUMNS = {
    "cpg_idx": "int",
    "sequence": "str",
    "cpg_island_tuple": "str",
    "methylation": "float64",
    "sample_idx": "int",
    "group_idx": "int",
    "chr_idx": "int",
    "tissue_idx": "int",
}

UNUSED_COLUMNS = {
    "sample_name": "str",
    "cpg_chr_pos": "str",
}


def write_mds_one_parquet(args: Iterator[Tuple[Path, int, Path]]):
    (
        input_parquet_list,
        group_idx,
        output_dir,
        config,
    ) = args

    remove_unused_columns: bool = config.remove_unused_columns
    filter_by_chr: str = config.filter_by_chr
    tokenize_sequence: bool = config.tokenize_sequence
    dna_tokenizer_name: str = config.dna_tokenizer_name
    num_nbase: int = config.num_nbase
    batch_size: int = config.batch_size

    size_limit = "100mb"
    output_dir = str(output_dir)

    if remove_unused_columns:
        columns = USED_COLUMNS
    else:
        columns = USED_COLUMNS.copy()
        columns.update(UNUSED_COLUMNS)

    dna_tokenizer = None
    if tokenize_sequence:

        dna_tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer_name, trust_remote_code=True)

        # NOTE xk: we use uint16, 0 - 65,535
        if dna_tokenizer.vocab_size > 65535:
            tokenized_sequence_dtype = "uint32"
        else:
            tokenized_sequence_dtype = "uint16"

        # NOTE xk: mds data field format, https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html
        columns.pop("sequence")
        columns["tokenized_sequence"] = f"ndarray:{tokenized_sequence_dtype}"

    with MDSWriter(out=output_dir, columns=columns, size_limit=size_limit) as out:
        pbar = tqdm.tqdm(input_parquet_list, desc="Processing")
        for input_parquet_path in pbar:
            # Read parquet in chunks
            parquet_file = pq.ParquetFile(input_parquet_path)
            for df_chunk in parquet_file.iter_batches(batch_size=batch_size):
                df_chunk = df_chunk.to_pandas()

                # Add group_idx and chr_idx
                df_chunk["group_idx"] = group_idx
                try:
                    df_chunk["chr_idx"] = df_chunk["cpg_chr_pos"].apply(
                        lambda x: CHR_IDX_MAPPING[x.split("_")[0].lower()]
                    )
                except AttributeError as e:
                    logging.error(f"Error in {input_parquet_path}, df_chunk {df_chunk}")
                    df_chunk.to_csv("misc/error.csv", index=False)
                    raise e

                # Filter columns
                if remove_unused_columns:
                    df_chunk.drop(UNUSED_COLUMNS.keys(), axis=1, inplace=True)

                # Filter rows by chromosome
                if filter_by_chr is not None:
                    filter_by_chr_idx = []
                    for chr_name in filter_by_chr:
                        if chr_name.lower() not in CHR_IDX_MAPPING:
                            raise ValueError(f"Invalid chromosome: {chr_name}")
                        filter_by_chr_idx.append(CHR_IDX_MAPPING[chr_name.lower()])
                    df_chunk = df_chunk[df_chunk["chr_idx"].isin(filter_by_chr_idx)]

                # if df_chunk is empty, skip
                # NOTE xk: it is possible if we filter by chromosome
                if len(df_chunk) == 0:
                    continue

                if tokenize_sequence:
                    sequence = df_chunk.pop("sequence")
                    sequence_len = len(sequence.iloc[0])
                    if sequence_len < num_nbase:
                        raise ValueError(f"sequence length is less than num_nbase: {sequence_len} < {num_nbase}")

                    mid_pos = sequence_len // 2
                    left_pos = mid_pos - num_nbase // 2
                    right_pos = mid_pos + num_nbase // 2
                    if left_pos < 0:
                        raise ValueError(f"left_pos is less than 0: {left_pos}")
                    if right_pos > sequence_len:
                        raise ValueError(f"right_pos is greater than sequence length: {right_pos} > {sequence_len}")

                    sequence = sequence.str.upper()
                    sequence = sequence.apply(lambda x: x[left_pos:right_pos])

                    tokenized_sequence = dna_tokenizer(
                        sequence.tolist(),
                        add_special_tokens=False,
                        return_tensors="np",
                        return_token_type_ids=False,
                        return_attention_mask=False,
                    )
                    num_n_in_seq = sequence.apply(lambda x: x.count("N"))
                else:
                    tokenized_sequence = None
                    num_n_in_seq = df_chunk["sequence"].str.upper().apply(lambda x: x.count("N"))

                df_chunk = df_chunk.to_dict("records")
                for idx, row in enumerate(df_chunk):
                    if tokenized_sequence is not None:
                        row["tokenized_sequence"] = tokenized_sequence["input_ids"][idx].astype(
                            tokenized_sequence_dtype
                        )
                    # NOTE xk: filter any seq with N, as there is not too much of them
                    if num_n_in_seq.iloc[idx] > 0:
                        continue
                    out.write(row)

                # Clear memory
                tokenized_sequence = None
                sequence = None
                df_chunk = None
                gc.collect()
            memory_in_mb = get_memory_usage_in_gb()
            pbar.set_postfix_str(
                f"{input_parquet_path.name}; mem: {memory_in_mb:.2f} GB; est total: {memory_in_mb * config.num_workers:.2f} GB"
            )


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory already exists: {output_dir}")
            exit()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_memory_usage_in_gb():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024


if __name__ == "__main__":
    app.run(main)
