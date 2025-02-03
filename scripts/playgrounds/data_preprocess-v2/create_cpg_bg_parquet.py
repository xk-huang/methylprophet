"""
The content of the cpg_bg.parquet file is as follows:
      CpG_name  CpG_location                                           sequence
0        10469    chr1_10469  AACCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAACCCTAAC...
1        10471    chr1_10471  CCCTAACCCTAACCCTAACCCCTAACCCTAACCCTAACCCTAACCC...
2        10484    chr1_10484  CCTAACCCCTAACCCTAACCCTAACCCTAACCCTCGCGGTACCCTC...
3        10489    chr1_10489  CCCCTAACCCTAACCCTAACCCTAACCCTCGCGGTACCCTCAGCCG...
4        10493    chr1_10493  TAACCCTAACCCTAACCCTAACCCTCGCGGTACCCTCAGCCGGCCC...
...        ...           ...                                                ...
9995   1012484  chr1_1012484  TGATTTGCATTTCCCCAATTAATAATGATGTTGAACATCACTTTAC...
9996   1012748  chr1_1012748  GTAGAAAATAAAAGATAGGTCTCTTTTATTAAAAAACAATCTGAGG...
9997   1012764  chr1_1012764  AGGTCTCTTTTATTAAAAAACAATCTGAGGCTCCGGGTGCAGTGGC...
9998   1012792  chr1_1012792  GGCTCCGGGTGCAGTGGCTCACGCCTGTAATCCCAGCAGTTTCAGA...
9999   1012824  chr1_1012824  CCAGCAGTTTCAGAGGCCGAGGCAGGTGGATCACTTGAGCCCAGGA...
"""

import shutil
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("me_parquet_dir", "data/parquet/encode_wgbs-240802/me.parquet", "Directory of the me parquet data")
flags.DEFINE_string(
    "save_cpg_bg_parquet_dir",
    "data/parquet/encode_wgbs-240802/cpg_bg.parquet",
    "Directory to save the cpg_bg parquet files",
)
flags.DEFINE_string("hg38_parquet_file", "data/parquet/grch38_hg38/hg38.fa.parquet", "hg38 parquet file")
flags.DEFINE_boolean("overwrite", False, "Overwrite the existing parquet files")
flags.DEFINE_integer("cpg_bg_nbase_length", 1200, "Number of bases to extract from the cpg_bg file")


FLAGS = flags.FLAGS


def add_sequence_to_chr_pos_df(hg38_df, cpg_bg_nbase_length):
    def _add_sequence_to_chr_pos_df(row):
        chr, pos = row["chr"], row["pos"]
        seq, seq_len = hg38_df.loc[chr]

        # NOTE xk: very slow
        # seq = seq.upper()
        try:
            pos = int(pos)
        except ValueError:
            # XXX xk: ValueError: invalid literal for int() with base 10: '9e+06'
            pos = int(eval(pos))
        cg_nbase = seq[pos]
        if cg_nbase not in ("C", "G", "c", "g"):
            raise ValueError(f"Invalid CpG base {cg_nbase} at {chr}:{pos}")

        left_pos = pos - cpg_bg_nbase_length // 2
        right_pos = pos + cpg_bg_nbase_length // 2

        left_pad_len = max(0, -left_pos)
        right_pad_len = max(0, right_pos - seq_len)

        left_pos = max(0, left_pos)
        right_pos = min(seq_len, right_pos)

        cpg_bg_seq = seq[left_pos:right_pos]
        cpg_bg_seq = "N" * left_pad_len + cpg_bg_seq + "N" * right_pad_len

        if left_pad_len > 0 or right_pad_len > 0:
            logging.warning(f"Padding {chr}:{pos} with {left_pad_len} left and {right_pad_len} right")
        return cpg_bg_seq

    return _add_sequence_to_chr_pos_df


def main(_):
    me_parquet_dir = Path(FLAGS.me_parquet_dir)
    save_cpg_bg_parquet_dir = Path(FLAGS.save_cpg_bg_parquet_dir)

    hg38_parquet_file = Path(FLAGS.hg38_parquet_file)

    logging.info(f"Reading {hg38_parquet_file}")
    hg38_df = pd.read_parquet(hg38_parquet_file)
    logging.info(f"Read {hg38_df.shape[0]} rows")

    # NOTE: make sure there is no duplicated chr
    if hg38_df["chr"].duplicated().sum() > 0:
        raise ValueError("Duplicated chr in hg38_df")
    hg38_df.set_index("chr", inplace=True)

    if save_cpg_bg_parquet_dir.exists():
        if not FLAGS.overwrite:
            print(f"{save_cpg_bg_parquet_dir} already exists. Skipping...")
            return
        else:
            shutil.rmtree(save_cpg_bg_parquet_dir)
    save_cpg_bg_parquet_dir.mkdir(parents=True, exist_ok=True)

    cpg_bg_nbase_length = FLAGS.cpg_bg_nbase_length
    me_parquet_files = sorted(me_parquet_dir.glob("*.parquet"))
    pbar = tqdm.tqdm(me_parquet_files)
    for me_parquet_file in pbar:
        pbar.set_postfix_str(f"Processing {me_parquet_file.name}")
        me_sharded_df = pd.read_parquet(me_parquet_file)

        # XXX xk: chr and pos are in "Unnamed: 0" column, be aware that the name varies.
        me_sharded_df.rename(columns={"Unnamed: 0": "CpG_location"}, inplace=True)
        cpg_bg_df = me_sharded_df["CpG_location"].str.split("_", expand=True)
        cpg_bg_df.columns = ["chr", "pos"]

        cpg_bg_df["sequence"] = cpg_bg_df.apply(add_sequence_to_chr_pos_df(hg38_df, cpg_bg_nbase_length), axis=1)
        cpg_bg_df.rename(columns={"chr": "cpg_chr", "pos": "cpg_pos"}, inplace=True)

        cpg_bg_df["CpG_location"] = me_sharded_df["CpG_location"]
        cpg_bg_df["CpG_name"] = me_sharded_df.index
        cpg_bg_df = cpg_bg_df[["CpG_name", "CpG_location", "sequence"]]

        cpg_bg_df.to_parquet(save_cpg_bg_parquet_dir / me_parquet_file.name)

    logging.info(f"Saving to {save_cpg_bg_parquet_dir}")


if __name__ == "__main__":
    app.run(main)
