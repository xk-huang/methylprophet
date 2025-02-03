"""
Save the data as ThunderDB format.
Two dataframe: `gene_chr_pos`, `cpg_bg`, and `chr_bg`.
See `read_tdb.py` for how to read the data.
"""

import gzip
import sys
from pathlib import Path

import pandas as pd
import pyranges as pr
from Bio import SeqIO
from thunderpack import ThunderDB


# Process gene bg
gtf_file = "data/raw/grch38.v41.gtf"
print(f"Reading {gtf_file}...")
gtf_df = pr.read_gtf(gtf_file)
gtf_df = gtf_df.df

# NOTE xk: Besides 'gene', there are other features like 'transcript', 'exon', 'CDS', 'start_codon', 'stop_codon', 'UTR', 'Selenocysteine'.
# NOTE xk: duplicated gene_id on Chr XY. allela genes. on xy, the suffix `.?_PAR_Y`
kept_cols = ["Chromosome", "Feature", "Start", "End", "Strand", "gene_id", "gene_type", "protein_id"]
gtf_df = gtf_df[kept_cols].copy()

# NOTE xk: gene_id is duplicated, e.g. ENSG00000223972.5, ENSG00000227232.5
gtf_df.rename(columns={"Chromosome": "chr", "gene_id": "original_gene_id"}, inplace=True)
gtf_df["gene_id"] = gtf_df["original_gene_id"].apply(lambda x: x.split(".")[0])


def stat_and_save_duplicated_gene_id(df, save_path):
    num_duplicated_gene_id = len(df) - len(df["gene_id"].unique())
    print(f"Number of genes: {len(df)}, duplicated gene id: {num_duplicated_gene_id}")
    df[df["gene_id"].duplicated(keep=False)].sort_values("gene_id").reset_index().to_csv(save_path)


# XXX xk: save duplicated gene_id
Path("misc").mkdir(exist_ok=True, parents=True)
stat_and_save_duplicated_gene_id(gtf_df[gtf_df["Feature"] == "gene"], "misc/duplicated_gene_id-before_filtering.csv")


# NOTE xk: keep exprsessed protein coding genes
# FIXME xk: What does `protain_id`.notnull() mean?
expr_protain_cond = (gtf_df["gene_type"] == "protein_coding") & gtf_df["protein_id"].notnull()
expr_protain_uniq_gene_id = gtf_df[expr_protain_cond]["gene_id"].unique()
gene_filter_cond = (gtf_df["Feature"] == "gene") & gtf_df["gene_id"].isin(expr_protain_uniq_gene_id)

gtf_df = gtf_df[gene_filter_cond]  # NOTE xk: there are duplicated gene_id
# XXX xk: save duplicated gene_id
Path("misc").mkdir(exist_ok=True, parents=True)
stat_and_save_duplicated_gene_id(gtf_df, "misc/duplicated_gene_id-after_filtering.csv")


def get_gene_pos(row):
    if row["Strand"] == "+":
        return row["Start"]
    elif row["Strand"] == "-":
        return row["End"]
    else:
        raise ValueError("Strand is not + or -")


gtf_df["gene_pos"] = gtf_df.apply(get_gene_pos, axis=1)
gtf_df.set_index("gene_id", inplace=True)


# Read cpg background
if len(sys.argv) != 3:
    raise ValueError("Please provide data_dir.")
data_dir = sys.argv[1]
save_dir = sys.argv[2]

data_dir = Path(data_dir)
save_dir = Path(save_dir)

dnam_file_name = "me_rownamesloc.csv"  # 393309 cpg sites X 8578 samples, 57GB
gene_expr_file_name = "ge.csv"  # 58560 genes X 8578 samples, 4.5GB
cancer_type_file_name = "project.csv"  # 8578 samples X 2 {sample name, cancer type} pair, 0.2MB
cpg_bg_file_name = (
    "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB
)

cpg_bg_dna_seq_path = data_dir / cpg_bg_file_name

cpg_bg_df = pd.read_csv(cpg_bg_dna_seq_path)

# NOTE xk: be aware of the column names, e.g., "sequence" -> "DNA_sequence"
cpg_bg_df.rename(
    columns={"CpG_name": "cpg_id", "CpG_location": "cpg_chr_pos", "sequence": "DNA_sequence"}, inplace=True
)
cpg_bg_df.set_index("cpg_id", inplace=True)


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

save_dir.mkdir(parents=True, exist_ok=True)

save_gene_chr_pos_path = save_dir / "gene_cpg_chr_bg.tdb"
with ThunderDB.open(str(save_gene_chr_pos_path), "c") as db:
    db["gene_bg"] = gtf_df
    db["cpg_bg"] = cpg_bg_df
    db["chr_bg"] = chr_bg_df
