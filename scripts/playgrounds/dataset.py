import datetime
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import thunderpack
import torch
import tqdm
from thunderpack import ThunderDB, ThunderReader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cpg_me_tdb_path, gene_expr_tdb_path, gene_cpg_chr_bg_tdb_path, num_closest_genes=1200):
        cpg_me_reader = ThunderReader(str(cpg_me_tdb_path))
        gene_expr_reader = ThunderReader(str(gene_expr_tdb_path))
        gene_cpg_chr_bg_reader = ThunderReader(str(gene_cpg_chr_bg_tdb_path))

        self.cpg_me_reader = cpg_me_reader
        self.gene_expr_reader = gene_expr_reader
        self.gene_cpg_chr_bg_reader = gene_cpg_chr_bg_reader

        self.cpg_bg_df = gene_cpg_chr_bg_reader["cpg_bg"]
        self.gene_bg_df = gene_cpg_chr_bg_reader["gene_bg"]
        self.chr_bg_df = gene_cpg_chr_bg_reader["chr_bg"]

        self.sample_name_keys = cpg_me_reader["keys-sample_name"]
        self.cpg_id_keys = cpg_me_reader["keys-cpg_id"]
        self.gene_id_keys = gene_expr_reader["keys-gene_id"]

        self.validate_gene_on_chr()
        self.filter_gene_id_by_function()
        self.filter_gene_id_by_intersection()
        self.build_index_mapping()

        self.num_closest_genes = num_closest_genes

    def validate_gene_on_chr(self):
        gene_chr_set = set(self.gene_bg_df["chr"].unique())
        chr_set = set(self.chr_bg_df.index.unique())
        joint_chr = gene_chr_set.intersection(chr_set)
        if len(joint_chr) != len(gene_chr_set):
            raise ValueError("Gene_bg and chr_bg do not have the same chromosomes.")

    def filter_gene_id_by_function(self):
        """_summary_
        ["Feature"] == "gene"
        ["gene_type"] == "protein_coding"
        ["protein_id"].notnull()
        """
        pass

    def filter_gene_id_by_intersection(self):
        # NOTE xk: find the intersection between gene_expr, gene_bg, and chr_bg.
        gene_expr_set = set(self.gene_id_keys)
        gene_bg_set = set(self.gene_bg_df.index)
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

        self.gene_id_keys = pd.Index(joint_gene_id)
        self.gene_bg_df = self.gene_bg_df.loc[joint_gene_id]

        if sum(self.gene_id_keys.duplicated()) > 0:
            raise ValueError(
                "Duplicated gene_id in intersection of gene_bg_df and gene expr. Should be on Chr X and Y."
            )

    def build_index_mapping(self):
        gene_id_keys = self.gene_id_keys
        cpg_id_keys = self.cpg_id_keys
        gene_bg_df = self.gene_bg_df
        sample_name_keys = self.sample_name_keys

        self.gene_id_to_idx = pd.Series(data=np.arange(len(gene_id_keys)), index=gene_id_keys.to_list())

        self.cpg_id_to_idx = pd.Series(
            data=np.arange(len(cpg_id_keys)), index=cpg_id_keys.to_list()
        )  # NOTE xk: cpg_id_keys is a pd.Index object

        chr_list = gene_bg_df["chr"].unique().sort_values()
        # chr_list = gene_bg_df["chr"].cat.categories
        self.chr_to_idx = pd.Series(data=np.arange(len(chr_list)), index=chr_list)

        self.sample_name_to_idx = pd.Series(
            data=np.arange(len(sample_name_keys)), index=sample_name_keys.to_list()
        )  # NOTE xk: sample_name_keys is a pd.Index object

        self.num_gene = len(self.gene_id_to_idx)
        self.num_cpg = len(self.cpg_id_to_idx)
        self.num_chr = len(self.chr_to_idx)
        self.num_sample = len(self.sample_name_to_idx)

    def __len__(self):
        return self.num_sample * self.num_cpg

    def __getitem__(self, index):
        # NOTE xk: Index order is (sample_name_idx, cpg_idx)
        sample_name_idx = index // self.num_cpg
        cpg_idx = index % self.num_cpg

        sample_name = self.sample_name_keys[sample_name_idx]
        cpg_id = self.cpg_id_keys[cpg_idx]

        # Get CpG background, chr, pos, and DNA sequence
        cpg_bg_dict = self.get_cpg_bg(cpg_id)
        cpg_chr, cpg_pos, cpg_dna_seq = cpg_bg_dict["chr"], cpg_bg_dict["pos"], cpg_bg_dict["dna_seq"]

        cpg_dna_seq_vec = get_nda_seq_vec(cpg_dna_seq)

        # Get gene by chr and pos, and filter by the closest genes
        filtered_gene_by_chr_df = self.filter_gene(cpg_chr, cpg_pos)

        # Get gene by sample_name and filtered gene_id
        gene_expr_by_sample_name = self.gene_expr_reader[sample_name]
        # Normalize gene expr values by sample name
        gene_expr_by_sample_name = self.normalize_gene_by_sample_name(gene_expr_by_sample_name)

        gene_expr = gene_expr_by_sample_name.loc[filtered_gene_by_chr_df.index]
        gene_idx = self.gene_id_to_idx[filtered_gene_by_chr_df.index]
        gene_position_ids = filtered_gene_by_chr_df["gene_pos"]

        # Convert Series to numpy array
        gene_expr = gene_expr.to_numpy()
        gene_idx = gene_idx.to_numpy()
        gene_position_ids = gene_position_ids.to_numpy()

        # Pad gene, gene_idx, gene_pos
        gene_expr, gene_idx, gene_position_ids, gene_attention_mask = self.pad_gene(
            gene_expr, gene_idx, gene_position_ids
        )

        max_len = self.num_closest_genes
        if len(gene_expr) != max_len:
            raise ValueError(f"Number of `gene_expr` is not equal to num_cloest_gene: {len(gene_expr)} vs {max_len}.")
        if len(gene_idx) != max_len:
            raise ValueError(f"Number of `gene_idx` is not equal to num_cloest_gene: {len(gene_idx)} vs {max_len}.")
        if len(gene_position_ids) != max_len:
            raise ValueError(
                f"Number of `gene_position_ids` is not equal to num_cloest_gene: {len(gene_position_ids)} vs {max_len}."
            )

        # Get additional indexes
        chr_idx = self.chr_to_idx[cpg_chr]
        cpg_idx = self.cpg_id_to_idx[cpg_id]
        sample_idx = self.sample_name_to_idx[sample_name]

        # Get CpG methylatio value
        cpg_me = self.cpg_me_reader[sample_name][cpg_id]
        cpg_position_ids = cpg_pos

        return {
            # CpG related
            "cpg_me": cpg_me,
            "cpg_position_ids": cpg_position_ids,
            "cpg_dna_seq_vec": cpg_dna_seq_vec,
            # Gene related
            "gene_expr": gene_expr,
            "gene_position_ids": gene_position_ids,
            "gene_attention_mask": gene_attention_mask,
            # Index related
            "sample_idx": sample_idx,
            "cpg_idx": cpg_idx,
            "gene_idx": gene_idx,
            "chr_idx": chr_idx,
        }

    def normalize_gene_by_sample_name(self, gene_expr_by_sample_name):
        max_gene_expr = gene_expr_by_sample_name.max()
        min_gene_expr = gene_expr_by_sample_name.min()
        normalized_gene_expr_by_sample_name = (gene_expr_by_sample_name - min_gene_expr) / (
            max_gene_expr - min_gene_expr
        )

        return normalized_gene_expr_by_sample_name

    def pad_gene(self, gene_expr, gene_idx, gene_position_ids):
        # NOTE xk: we pad the gene_expr, gene_idx, gene_pos with 0, and and gene mask for fro transformer input.
        max_len = self.num_closest_genes
        if max_len < len(gene_idx):
            raise ValueError("Number of closest genes is larger than expected, which is impossible.")
        original_len = len(gene_idx)
        padded_len = max_len - original_len

        gene_expr = np.pad(gene_expr, (0, padded_len), constant_values=0)
        gene_idx = np.pad(gene_idx, (0, padded_len), constant_values=0)
        gene_position_ids = np.pad(gene_position_ids, (0, padded_len), constant_values=0)

        gene_attention_mask = np.ones(max_len, dtype=bool)
        gene_attention_mask[original_len:] = False

        return gene_expr, gene_idx, gene_position_ids, gene_attention_mask

    def get_gene_pos_ratio(self, filtered_gene_by_chr_df):
        filtered_gene_by_chr_df["chr_len"] = filtered_gene_by_chr_df.apply(self.assign_chr_seq_to_gene, axis=1)
        gene_pos_ratio = filtered_gene_by_chr_df["gene_pos"] / filtered_gene_by_chr_df["chr_len"]

        return gene_pos_ratio

    def assign_chr_seq_to_gene(self, row):
        return self.chr_bg_df.loc[row["chr"]]

    def get_cpg_bg(self, cpg_id):
        # NOTE xk: find chr and pos of the given cpg_id.
        cpg_bg_series = self.cpg_bg_df.loc[cpg_id]
        cpg_chr_pos = cpg_bg_series["cpg_chr_pos"]
        cpg_chr, cpg_pos = cpg_chr_pos.split("_")  # e.g., chr16_53434200
        # NOTE xk: convert cpg_pos to np.int64, which supports `shape`.
        cpg_pos = np.int64(cpg_pos)
        cpg_dna_seq = cpg_bg_series["DNA_sequence"]

        return {
            "chr": cpg_chr,
            "pos": cpg_pos,
            "dna_seq": cpg_dna_seq,
        }

    def filter_gene(self, cpg_chr, cpg_pos):
        gene_bg_df = self.gene_bg_df

        gene_bg_by_chr_df = gene_bg_df[gene_bg_df["chr"] == cpg_chr]
        cpg_vs_gene_rel_pos_series = np.abs(gene_bg_by_chr_df["gene_pos"] - cpg_pos)

        num_closest_genes = self.num_closest_genes
        closest_gene_rel_pos_series = cpg_vs_gene_rel_pos_series.nsmallest(n=num_closest_genes)
        closest_gene_by_chr_df = gene_bg_by_chr_df.loc[closest_gene_rel_pos_series.index]
        closest_gene_by_chr_df["cpg_gene_rel_pos"] = closest_gene_rel_pos_series

        return closest_gene_by_chr_df


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


if __name__ == "__main__":
    save_dir = "data/processed"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_cpg_me_path = save_dir / "cpg_me.tdb"
    save_gene_expr_path = save_dir / "gene_expr.tdb"
    save_gene_cpg_chr_bg_path = save_dir / "gene_cpg_chr_bg.tdb"

    dataset = Dataset(
        cpg_me_tdb_path=save_cpg_me_path,
        gene_expr_tdb_path=save_gene_expr_path,
        gene_cpg_chr_bg_tdb_path=save_gene_cpg_chr_bg_path,
    )
    one_data = dataset[0]
    print(one_data)
    for k, v in one_data.items():
        print(f"{k}: {type(v)}, {v.shape}, {v.dtype}")

    # NOTE xk: Concatenation takes time, batch_size not be too large.
    num_workers = 64
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=num_workers,
        # shuffle=True,
    )

    pbar_print_interval = 100
    from pprint import pformat

    pbar = tqdm.tqdm(dataloader)
    for one_data in pbar:
        if pbar.n % pbar_print_interval == 0:
            pbar.write(pformat(one_data))
