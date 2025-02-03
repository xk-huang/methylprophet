"""
Save the data as ThunderDB format.
The mapping is `sample_name` -> all cpg me values whose index is `cpg_id`
The other keys include `keys-sample_name` and `keys-cpg_id`.
See `read_tdb.py` for how to read the data.

NOTE xk: CpG pos_name has duplicated values. We need cpg id to differentiate them.
Specifically, 393309 CpG sites -> 393292 unique CpG sites.
"""

import datetime
import multiprocessing
import shutil
import time
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags
from thunderpack import ThunderDB


flags.DEFINE_string("data_dir", "data/processed/tcga_450k-240724", "Directory of the data")
flags.DEFINE_alias("d", "data_dir")

FLAGS = flags.FLAGS


def main(_):
    data_dir = FLAGS.data_dir
    data_dir = Path(data_dir)

    dnam_file_name = "me_rownamesloc.csv"  # 393309 cpg sites X 8578 samples, 57GB
    gene_expr_file_name = "ge.csv"  # 58560 genes X 8578 samples, 4.5GB
    cancer_type_file_name = "project.csv"  # 8578 samples X 2 {sample name, cancer type} pair, 0.2MB
    cpg_bg_file_name = "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB

    cpg_bg_dna_seq_path = data_dir / cpg_bg_file_name
    gene_expr_path = data_dir / gene_expr_file_name
    dnam_path = data_dir / dnam_file_name

    cpg_bg_df = pd.read_csv(cpg_bg_dna_seq_path, nrows=10)
    cpg_bg_df.rename(columns={"CpG_name": "cpg_id", "CpG_location": "cpg_chr_pos"}, inplace=True)

    gene_expr_df = pd.read_csv(gene_expr_path, nrows=10)

    me_df = pd.read_csv(dnam_path, nrows=10)

    print(cpg_bg_dna_seq_path, gene_expr_path, dnam_path)
    print(cpg_bg_df.head())
    print(gene_expr_df.head())
    print(me_df.head())

    # cpg_bg_df = pd.read_csv(cpg_bg_dna_seq_path, usecols=["CpG_name", "CpG_location"])


if __name__ == "__main__":
    app.run(main)
