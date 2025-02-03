"""
python scripts/tools/data_preprocessing/stats_n_in_cpg_dna_seq.py \
    --input_cpg_bg_parquet_dir_list data/parquet/241213-encode_wgbs/cpg_bg.parquet \
    --output_dir data/parquet/241213-encode_wgbs/metadata \
    --output_file_name cpg_nda_seq_n_stats \
    --overwrite

python scripts/tools/data_preprocessing/stats_n_in_cpg_dna_seq.py \
    --input_cpg_bg_parquet_dir_list data/parquet/241231-tcga_array/cpg_bg.parquet,data/parquet/241231-tcga_epic/cpg_bg.parquet,data/parquet/241231-tcga_wgbs/cpg_bg.parquet \
    --output_dir data/parquet/241231-tcga/metadata \
    --output_file_name cpg_nda_seq_n_stats \
    --overwrite
"""

import json
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import tqdm
from absl import app, flags, logging

from transformers import AutoTokenizer


flags.DEFINE_list("input_cpg_bg_parquet_dir_list", None, "List of input parquet directories")
flags.DEFINE_integer("num_nbase", 1000, "Number of bases for DNA tokenizer")
flags.DEFINE_string("dna_tokenizer_name", "zhihan1996/DNABERT-2-117M", "DNA tokenizer name")

flags.DEFINE_integer("batch_size", 10000, "Batch size for processing")
flags.DEFINE_integer("num_processes", 20, "Number of processes to use (defaults to CPU count)")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("output_file_name", None, "Output file name")

flags.DEFINE_bool("overwrite", False, "Overwrite existing output file")

FLAGS = flags.FLAGS


def process_parquet_file(args):
    input_parquet_path, dna_tokenizer_name, batch_size, num_nbase = args

    dna_tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer_name, trust_remote_code=True)
    cpg_dna_n_stats_df_list = []

    parquet_file = pq.ParquetFile(input_parquet_path)
    for df_chunk in parquet_file.iter_batches(batch_size=batch_size):
        df_chunk = df_chunk.to_pandas()
        cpg_dna_n_stats_df = get_cpg_dna_n_stats(df_chunk, dna_tokenizer, num_nbase)
        cpg_dna_n_stats_df_list.append(cpg_dna_n_stats_df)

    return pd.concat(cpg_dna_n_stats_df_list, ignore_index=True)


def get_cpg_dna_n_stats(cpg_bg_df, dna_tokenizer, num_nbase):
    sequence = cpg_bg_df["sequence"]
    cpg_chr_pos = cpg_bg_df["CpG_location"]

    sequence_len = len(cpg_bg_df["sequence"].iloc[0])
    mid_pos = sequence_len // 2
    left_pos = mid_pos - num_nbase // 2
    right_pos = mid_pos + num_nbase // 2
    if left_pos < 0:
        raise ValueError(f"left_pos is less than 0: {left_pos}")
    if right_pos > sequence_len:
        raise ValueError(f"right_pos is greater than sequence length: {right_pos} > {sequence_len}")

    sequence = sequence.str.upper()
    sequence = sequence.apply(lambda x: x[left_pos:right_pos])
    cpg_bg_df["num_n_in_seq"] = sequence.apply(lambda x: x.count("N"))
    tokenized_sequence = dna_tokenizer(
        sequence.tolist(),
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    tokenized_seq_len = [len(seq) for seq in tokenized_sequence["input_ids"]]

    return pd.DataFrame(
        {
            "CpG_location": cpg_chr_pos,
            "num_n_in_seq": cpg_bg_df["num_n_in_seq"],
            "tokenized_seq_len": tokenized_seq_len,
        }
    )


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_dir = output_dir / FLAGS.output_file_name
    if output_dir.exists():
        if FLAGS.overwrite:
            shutil.rmtree(output_dir)
            logging.info(f"Removed existing output directory: {output_dir}")
        else:
            logging.warning(f"Output directory already exists: {output_dir}")
            return
    output_dir.mkdir(parents=True, exist_ok=True)

    num_processes = FLAGS.num_processes or cpu_count()
    batch_size = FLAGS.batch_size
    dna_tokenizer_name = FLAGS.dna_tokenizer_name
    num_nbase = FLAGS.num_nbase

    parquet_dir_idx_mapping = dict(enumerate(FLAGS.input_cpg_bg_parquet_dir_list))
    parquet_mapping_path = output_dir / "parquet_dir_idx_mapping.json"
    with open(parquet_mapping_path, "w") as f:
        json.dump(parquet_dir_idx_mapping, f, indent=4)
    logging.info(f"Saved parquet_dir_idx_mapping to {parquet_mapping_path}")

    logging.info("Loading CpG background data")
    for parquet_dir_idx, input_cpg_bg_parquet_dir in enumerate(FLAGS.input_cpg_bg_parquet_dir_list):
        input_parquet_list = list(Path(input_cpg_bg_parquet_dir).rglob("*.parquet"))

        # Prepare arguments for multiprocessing
        process_args = [(path, dna_tokenizer_name, batch_size, num_nbase) for path in input_parquet_list]

        # Process files in parallel
        with Pool(num_processes) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(process_parquet_file, process_args),
                    total=len(input_parquet_list),
                    desc="Processing parquet files",
                )
            )

        # Combine results
        cpg_dna_n_stats_df = pd.concat(results, ignore_index=True)

        output_path = output_dir / f"cpg_nda_n_stats-{parquet_dir_idx:05d}.csv"
        cpg_dna_n_stats_df.to_csv(output_path, index=False)
        logging.info(f"Saved to {output_path}")

        stats_cpg_dna_df(cpg_dna_n_stats_df, output_dir, parquet_dir_idx)


def stats_cpg_dna_df(df, output_dir, idx):
    df = df.sort_values("num_n_in_seq", ascending=False)
    df = df.reset_index(drop=True)

    first_zero_index = df[df["num_n_in_seq"] == 0].index[0]

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    # first plot num_n_in_seq, then tokenized_seq_len
    sns.lineplot(data=df["num_n_in_seq"].iloc[:first_zero_index], ax=ax[0])
    sns.lineplot(data=df["tokenized_seq_len"].iloc[:first_zero_index], ax=ax[1])
    fig_path = output_dir / f"n_seq_stats-{idx:05d}.png"
    fig.savefig(fig_path)
    logging.info(f"Saved to {fig_path}")

    n_seq_df = df[df["num_n_in_seq"] > 0]
    n_seq_df["chr"] = n_seq_df["CpG_location"].apply(lambda x: x.split("_")[0])
    n_seq_df_count = n_seq_df["chr"].value_counts().sort_index()

    df["chr"] = df["CpG_location"].apply(lambda x: x.split("_")[0])
    df_chr_count = df["chr"].value_counts().sort_index()

    n_seq_ratio = n_seq_df_count / df_chr_count
    n_seq_ratio_path = output_dir / f"n_seq_ratio-{idx:05d}.csv"
    n_seq_ratio.to_csv(n_seq_ratio_path)
    logging.info(f"Saved to {n_seq_ratio_path}")


if __name__ == "__main__":
    app.run(main)
