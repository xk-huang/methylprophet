import gzip
from pathlib import Path

import pandas as pd
import pyranges as pr
from absl import app, flags, logging
from Bio import SeqIO


"""
# Process gene bg
gtf_file = "data/raw/grch38.v41.gtf"
print(f"Reading {gtf_file}...")
gtf_df = pr.read_gtf(gtf_file)
gtf_df = gtf_df.df

# Read chr background
hg38_fa_gz = "data/raw/hg38.fa.gz"
chr_bg_df = []
with gzip.open(hg38_fa_gz, "rt") as f:
    for record in SeqIO.parse(f, "fasta"):
        chr = record.id
        seq = record.seq
        seq_len = len(seq)
        chr_bg_df.append(
            {
                "chr": chr,
                # "seq": seq,
                "seq_len": seq_len,
            }
        )
chr_bg_df = pd.DataFrame.from_records(chr_bg_df)
chr_bg_df.set_index("chr", inplace=True)
"""

flags.DEFINE_string("grch38_file", "data/raw/grch38.v41.gtf", "grch38.v41.gtf file path")
flags.DEFINE_string("hg38_file", "data/raw/hg38.fa.gz", "hg38.fa.gz file path")
flags.DEFINE_string("save_dir", "data/parquet/grch38_hg38", "Directory to save the parquet files")
flags.DEFINE_boolean("overwrite", False, "Overwrite the existing parquet files")

FLAGS = flags.FLAGS


def main(_):
    grch38_file = Path(FLAGS.grch38_file)
    hg38_file = Path(FLAGS.hg38_file)
    save_dir = Path(FLAGS.save_dir)
    overwrite = FLAGS.overwrite

    save_dir.mkdir(parents=True, exist_ok=True)

    write_grch38(grch38_file, save_dir, overwrite)
    write_hg38(hg38_file, save_dir, overwrite)


def write_hg38(hg38_file, save_dir, overwrite):
    hg38_save_path = save_dir / hg38_file.with_suffix(".parquet").name
    if hg38_save_path.exists() and not overwrite:
        logging.info(f"{hg38_save_path} already exists. Skipping...")
        return

    logging.info(f"Reading {hg38_file}...")
    hg38_df = []
    with gzip.open(hg38_file, "rt") as f:
        for record in SeqIO.parse(f, "fasta"):
            chr = record.id
            seq = record.seq
            seq = str(seq)  # NOTE xk: Arrow does not support Bio.Seq.Seq
            seq_len = len(seq)
            hg38_df.append(
                {
                    "chr": chr,
                    "seq": seq,
                    "seq_len": seq_len,
                }
            )
    hg38_df = pd.DataFrame.from_records(hg38_df)
    logging.info(f"Fetched {len(hg38_df)} records.")

    hg38_df.to_parquet(hg38_save_path)


def write_grch38(grch38_file, save_dir, overwrite):
    grch38_save_path = save_dir / grch38_file.with_suffix(".parquet").name
    if grch38_save_path.exists() and not overwrite:
        logging.info(f"{grch38_save_path} already exists. Skipping...")
        return

    logging.info(f"Reading {grch38_file}...")
    grch38_df = pr.read_gtf(grch38_file)
    grch38_df = grch38_df.df
    logging.info(f"Fetched {len(grch38_df)} records.")

    grch38_df.to_parquet(grch38_save_path)


if __name__ == "__main__":
    app.run(main)
