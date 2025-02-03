import datetime
import gzip
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from Bio import SeqIO
from thunderpack import ThunderDB, ThunderReader


data_dir = "data/raw"
data_dir = Path(data_dir)

dnam_file_name = "me_rownamesloc.csv"  # 393309 cpg sites X 8578 samples, 57GB
gene_expr_file_name = "ge.csv"  # 58560 genes X 8578 samples, 4.5GB
cancer_type_file_name = "project.csv"  # 8578 samples X 2 {sample name, cancer type} pair, 0.2MB
cpg_bg_file_name = (
    "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB
)

dnam_path = data_dir / dnam_file_name
gene_expr_path = data_dir / gene_expr_file_name
cancer_type_path = data_dir / cancer_type_file_name
cpg_bg_dna_seq_path = data_dir / cpg_bg_file_name

save_dir = "data/processed"
save_dir = Path(save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

save_cpg_me_path = save_dir / "cpg_me.tdb"
save_gene_expr_path = save_dir / "gene_expr.tdb"
save_gene_cpg_chr_bg_path = save_dir / "gene_cpg_chr_bg.tdb"

# Load data
cpg_me_reader = ThunderReader(str(save_cpg_me_path))
gene_expr_reader = ThunderReader(str(save_gene_expr_path))
gene_cpg_chr_bg_reader = ThunderReader(str(save_gene_cpg_chr_bg_path))

cpg_bg_df = gene_cpg_chr_bg_reader["cpg_bg"]
gene_bg_df = gene_cpg_chr_bg_reader["gene_bg"]
chr_bg_df = gene_cpg_chr_bg_reader["chr_bg"]

sample_name_keys = cpg_me_reader["keys-sample_name"]
cpg_id_keys = cpg_me_reader["keys-cpg_id"]
gene_id_keys = gene_expr_reader["keys-gene_id"]

# Validate gene on chr
gene_chr_set = set(gene_bg_df["chr"].unique())
chr_set = set(chr_bg_df.index.unique())
joint_chr = gene_chr_set.intersection(chr_set)
if len(joint_chr) != len(gene_chr_set):
    raise ValueError("Gene_bg and chr_bg do not have the same chromosomes.")

# NOTE xk: find the intersection between gene_expr and gene_bg, filter both expr and bg by the intersection.
gene_expr_set = set(gene_id_keys)
gene_bg_set = set(gene_bg_df.index)
joint_gene_id = gene_expr_set.intersection(gene_bg_set)
print(
    f"Number of expressed genes:\t{len(gene_expr_set)}\n"
    f"Number of genes in gene_bg:\t{len(gene_bg_set)}\n"
    f"Number of joint genes:\t{len(joint_gene_id)}"
)
"""
Number of expressed genes:      58560
Number of genes in gene_bg:     61806
Number of joint genes:  55138
"""
joint_gene_id = list(joint_gene_id)
gene_id_keys = pd.Index(joint_gene_id)
gene_bg_df = gene_bg_df.loc[joint_gene_id]
if sum(gene_id_keys.duplicated()) > 0:
    raise ValueError("Duplicated gene_id in intersection of gene_bg_df and gene expr. Should be on Chr X and Y.")

# Build gene_id to index mapping
gene_id_to_idx = pd.Series(data=np.arange(len(gene_id_keys)), index=gene_id_keys.to_list())
cpg_id_to_idx = pd.Series(
    data=np.arange(len(cpg_id_keys)), index=cpg_id_keys.to_list()
)  # NOTE xk: cpg_id_keys is a pd.Index object
chr_list = gene_bg_df["chr"].unique().sort_values()
# chr_list = gene_bg_df["chr"].cat.categories
chr_to_idx = pd.Series(data=np.arange(len(chr_list)), index=chr_list)
sample_name_to_idx = pd.Series(
    data=np.arange(len(sample_name_keys)), index=sample_name_keys.to_list()
)  # NOTE xk: sample_name_keys is a pd.Index object

num_gene = len(gene_id_to_idx)
num_cpg = len(cpg_id_to_idx)
num_chr = len(chr_to_idx)
num_sample = len(sample_name_to_idx)

# Load one cpg's data
sample_name = "TCGA-BK-A0CA-01"
cpg_id = "cg00000029"


# NOTE xk: find chr and pos of the given cpg_id.
cpg_bg_series = cpg_bg_df.loc[cpg_id]
cpg_chr_pos = cpg_bg_series["cpg_chr_pos"]
cpg_chr, cpg_pos = cpg_chr_pos.split("_")  # e.g., chr16_53434200
cpg_pos = int(cpg_pos)

cpg_dna_seq = cpg_bg_series["DNA_sequence"]

# NOTE xk: find genes that are in the same chromosome of the given cpg_id.
gene_bg_by_chr_df = gene_bg_df[gene_bg_df["chr"] == cpg_chr]
cpg_vs_gene_rel_pos_series = np.abs(gene_bg_by_chr_df["gene_pos"] - cpg_pos)

NUM_CLOSEST_GENES = 5
closest_gene_rel_pos_series = cpg_vs_gene_rel_pos_series.nsmallest(n=NUM_CLOSEST_GENES)
closest_gene_by_chr_df = gene_bg_by_chr_df.loc[closest_gene_rel_pos_series.index]
closest_gene_by_chr_df["cpg_gene_rel_pos"] = closest_gene_rel_pos_series


def assign_chr_seq_to_gene(row):
    return chr_bg_df.loc[row["chr"]]


closest_gene_by_chr_df["chr_len"] = closest_gene_by_chr_df.apply(assign_chr_seq_to_gene, axis=1)

gene_rel_pos_ratio = closest_gene_by_chr_df["cpg_gene_rel_pos"] / closest_gene_by_chr_df["chr_len"]
gene_rel_pos_ratio = gene_rel_pos_ratio.to_numpy()

gene_idx = gene_id_to_idx[closest_gene_by_chr_df.index]
gene_idx = gene_idx.to_numpy()

gene = gene_expr_reader[sample_name]
gene = gene[closest_gene_by_chr_df.index]
gene = gene.to_numpy()

chr_idx = chr_to_idx[cpg_chr]
cpg_idx = cpg_id_to_idx[cpg_id]
sample_idx = sample_name_to_idx[sample_name]

# TODO xk: convert cpg_dna_seq to one-hot encoding, for data collate in torch.
DNA_MAPPING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}


def get_nda_seq_vec(dna_seq):
    dna_seq = dna_seq.upper()
    dna_seq_vec = [DNA_MAPPING[base] for base in dna_seq]
    dna_seq_vec = np.array(dna_seq_vec)
    # onehot_dna_seq = torch.nn.functional.one_hot(dna_seq_vec, num_classes=len(DNA_MAPPING))
    return dna_seq_vec


cpg_dna_seq_vec = get_nda_seq_vec(cpg_dna_seq)

cpg_me = cpg_me_reader[sample_name][cpg_id]

one_data = {
    "cpg_me": cpg_me,
    "cpg_dna_seq_vec": cpg_dna_seq_vec,
    "gene": gene,
    "gene_idx": gene_idx,
    "gene_rel_pos_ratio": gene_rel_pos_ratio,
    "chr_idx": chr_idx,
    "cpg_idx": cpg_idx,
    "sample_idx": sample_idx,
}

from pprint import pprint


pprint(one_data)
for k, v in one_data.items():
    print(f"{k}: {type(v)}, {v.shape}, {v.dtype}")
