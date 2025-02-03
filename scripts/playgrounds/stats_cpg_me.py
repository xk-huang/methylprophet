import pickle
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
cpg_id_to_idx = pd.Series(data=np.arange(len(cpg_id_keys)), index=cpg_id_keys.to_list())

min_max_cpg_me_ls = []
nan_idx_dict = {}
nan_id_dict = {}
pbar = tqdm.tqdm(sample_name_keys)
for idx, sample_name in enumerate(pbar):
    cpg_me = cpg_me_reader[sample_name]
    min_cpg_me = cpg_me.min()
    max_cpg_me = cpg_me.max()

    num_nan_values = cpg_me.isna().sum()
    nan_value_id = cpg_id_keys[cpg_me.isna()].to_list()
    nan_value_idx = cpg_id_to_idx[cpg_me.isna()].to_list()

    nan_idx_dict[sample_name] = nan_value_idx
    nan_id_dict[sample_name] = nan_value_id

    min_max_cpg_me_ls.append(
        {
            "sample_name": sample_name,
            "sample_idx": idx,
            "min_cpg_me": min_cpg_me,
            "max_cpg_me": max_cpg_me,
            "num_nan_values": num_nan_values,
        }
    )
    pbar.write(pformat(min_max_cpg_me_ls[-1]))


min_max_cpg_me_df = pd.DataFrame.from_records(min_max_cpg_me_ls)
Path("misc").mkdir(exist_ok=True, parents=True)
min_max_cpg_me_df.to_csv("misc/min_max_cpg_me.csv", index=False)

with open("misc/cpg_me-nan_idx_dict.pkl", "wb") as f:
    pickle.dump(nan_idx_dict, f)
with open("misc/cpg_me-nan_id_dict.pkl", "wb") as f:
    pickle.dump(nan_id_dict, f)
