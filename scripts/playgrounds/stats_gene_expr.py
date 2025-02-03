from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import tqdm
from thunderpack import ThunderReader


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
gene_id_to_idx = pd.Series(data=np.range(len(gene_id_keys)), index=gene_id_keys.to_list())

min_max_gene_expr_ls = []
pbar = tqdm.tqdm(sample_name_keys)
for idx, sample_name in enumerate(pbar):
    gene_expr = gene_expr_reader[sample_name]

    min_gene_expr = gene_expr.min()
    max_gene_expr = gene_expr.max()

    num_nan_values = gene_expr.isna().sum()
    nan_value_id = gene_id_keys[gene_expr.isna()].to_list()
    nan_value_idx = gene_id_to_idx[gene_expr.isna()].to_list()

    min_max_gene_expr_ls.append(
        {
            "sample_name": sample_name,
            "sample_idx": idx,
            "min_gene_expr": min_gene_expr,
            "max_gene_expr": max_gene_expr,
            "num_nan_values": num_nan_values,
            "nan_value_idx": nan_value_idx,
            "nan_value_id": nan_value_id,
        }
    )
    pbar.write(pformat(min_max_gene_expr_ls[-1]))

min_max_gene_expr_df = pd.DataFrame.from_records(min_max_gene_expr_ls)
Path("misc").mkdir(exist_ok=True, parents=True)
min_max_gene_expr_df.to_csv("misc/min_max_gene_expr.csv", index=False)
