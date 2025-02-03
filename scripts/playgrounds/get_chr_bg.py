import gzip
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# NOTE xk: Load DNA sequence for each chromosome from `hg38.fa.gz`. Sort of slow reading.
# `id`, `name`, `description` are the same.
records = []
with gzip.open("./data/raw/hg38.fa.gz", "rt") as df_seq:
    for record in SeqIO.parse(df_seq, "fasta"):
        seq = record.seq
        id = record.id
        name = record.name
        description = record.description
        dbxrefs = record.dbxrefs
        seq_len = len(seq)
        records.append(
            {
                # "seq": seq,
                "id": id,
                "name": name,
                "description": description,
                "dbxrefs": dbxrefs,
                "seq_len": seq_len,
            }
        )
chr_df = pd.DataFrame.from_records(records)
assert sum(chr_df["id"] != chr_df["name"]) == 0
assert sum(chr_df["id"] != chr_df["description"]) == 0
# chr_df.to_csv("misc/chr_df.csv", index=False)


data_dir = "data/raw"
data_dir = Path(data_dir)
cpg_bg_file_name = (
    "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB
)

cpg_bg_path = data_dir / cpg_bg_file_name
cpg_bg_df = pd.read_csv(cpg_bg_path)
cpg_bg_df.rename(columns={"CpG_name": "cpg_id", "CpG_location": "cpg_chr_pos"}, inplace=True)
cpg_bg_df.set_index("cpg_id", inplace=True)

cpg_bg_chr = cpg_bg_df.apply(lambda x: x["cpg_chr_pos"].split("_")[0], axis=1)
cpg_bg_chr.unique()
cpg_bg_chr_set = set(cpg_bg_chr)
len(cpg_bg_chr_set)

chr_set = set(chr_df["id"])
assert chr_set.intersection(cpg_bg_chr_set) == cpg_bg_chr_set
