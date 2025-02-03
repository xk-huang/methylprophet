"""
We first filter the gene, then write them to tdb.

Structure of gene_expr.tdb:
```
gene_expr.tdb:
    - sample_name_keys: List of sample names
    - gene_name_keys: List of gene names
    - sample_idx: gene expr numpy array
```

Example of me_cpg_dataset.parquet:
```
            cpg_chr_pos  cpg_idx                                           sequence                                        sample_name  methylation  sample_idx
0        chrX_101407287        0  ACACAGAAATGTACTGTGACTGCCACCTTTTACCAACATCTATcag...    Homo sapiens aorta tissue male adult (34 years)     0.010375           0
1          chrX_4726400        1  ggtgcatttacaaacctttagctagacagaaaagttctccaagtcc...    Homo sapiens aorta tissue male adult (34 years)     0.905137           0
2         chr7_56116409        2  CCTAGCCCCTTCTACCCACCTATGAAGTTCCGCAGGCTTTAATGCT...    Homo sapiens aorta tissue male adult (34 years)     0.055036           0
...
TCCCCCTTGCTGGTGTTCTGAACAAGCAGGTTTTGCAGCAGTGGCT...  Homo sapiens upper lobe of left lung tissue ma...     0.774020          94
949999    chr8_51767743     9999  tctagttagtccaacatctgagttttcttatggatagtttctattg...  Homo sapiens upper lobe of left lung tissue ma...     0.871823          94
```
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging

flags.DEFINE_string("data_dir", "data/parquet/encode_wgbs-240802", "Directory containing the parquet files")
flags.DEFINE_alias("d", "data_dir")
flags.DEFINE_string("save_dir", "data/processed/encode_wgbs-240802", "Directory to save the processed data")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite the existing files")

# flags.DEFINE_string("cpg_bg_file_name", "cpg_bg.shuffled.parquet", "Name of the CpG background file")
# flags.DEFINE_string("me_file_name", "me.shuffled.parquet", "Name of the methylation file")
flags.DEFINE_string("gene_expr_file_name", "gene_expr.parquet", "Name of the gene expression file")
flags.DEFINE_string("grch38_parquet_file", "data/parquet/grch38_hg38/grch38.v41.parquet", "grch38 parquet file path")

# flags.DEFINE_integer("num_cpgs", -1, help="Number of CpGs to process")
# flags.DEFINE_integer("num_samples", -1, help="Number of samples to process")
# flags.DEFINE_integer("sample_chunk_size", 100, "Number of samples to process in each chunk")

FLAGS = flags.FLAGS


def get_gene_position(row):
    if row["Strand"] == "+":
        return row["Start"]
    else:
        return row["End"]


chromosome_order = {f"chr{i}": i for i in range(1, 23)}
chromosome_order.update({"chrM": 23, "chrX": 24, "chrY": 25})


def get_gene_id_chr_pos_df(grch38_df):
    # NOTE: strip "ENSG00000223972.5" to "ENSG00000223972" in "gene_id"
    gene_id_chr_pos_df = grch38_df[["gene_id", "Chromosome", "Start", "End", "Strand"]].copy()
    gene_id = gene_id_chr_pos_df["gene_id"].str.split(".").str[0]
    chr = grch38_df["Chromosome"].map(chromosome_order)
    pos = gene_id_chr_pos_df.apply(get_gene_position, axis=1)
    return pd.DataFrame({"gene_id": gene_id, "chr": chr, "pos": pos})


def preprocess_gene_expr(gene_expr_df, grch38_df):
    # NOTE xk: direct condition is the same to complex condition
    # gene_condition = grch38_df["Feature"] == "gene"
    # protain_coding_condition = (grch38_df["gene_type"] == "protein_coding") & grch38_df["protein_id"].notnull()
    # gene_grch38_df = grch38_df[gene_condition]
    # protain_coding_grch38_df = grch38_df[protain_coding_condition]
    # condition = gene_grch38_df["gene_id"].isin(protain_coding_grch38_df["gene_id"])
    # filtered_grch38_df_complex = gene_grch38_df[condition]

    # condition = (grch38_df["Feature"] == "gene") & (grch38_df["gene_type"] == "protein_coding")
    # filtered_grch38_df = grch38_df[condition]
    # if not (filtered_grch38_df["gene_id"] == filtered_grch38_df_complex["gene_id"]).all():
    #     raise ValueError("filtered_grch38_df and filtered_grch38_df_complex are not the same")

    # Select protein coding genes in grch38_df
    condition = (grch38_df["Feature"] == "gene") & (grch38_df["gene_type"] == "protein_coding")
    filtered_gene_id = grch38_df[condition]["gene_id"]

    # Remove version number from gene_id in grch38_df
    # NOTE: from `ENSG00000186092.7` to `ENSG00000186092`
    filtered_gene_id = filtered_gene_id.apply(lambda x: x.split(".")[0])

    # Remove duplicated gene_id in grch38_df
    duplicated_gene_id_condition = filtered_gene_id.duplicated(keep=False)
    logging.info(f"Number of duplicated gene_id: {len(filtered_gene_id[duplicated_gene_id_condition])}")
    logging.info(grch38_df[condition]["gene_id"][duplicated_gene_id_condition])

    logging.info(
        f"Number of gene_id: {len(filtered_gene_id)} -> {len(filtered_gene_id[~duplicated_gene_id_condition])}"
    )
    filtered_gene_id = filtered_gene_id[~duplicated_gene_id_condition]
    if filtered_gene_id.duplicated().any():
        raise ValueError("filtered_gene_id has duplicated gene_id")

    # Remove null gene_id in gene_expr_df
    gene_expr_null_condition = gene_expr_df.index.isnull()
    logging.info(f"Number of null gene_id: {gene_expr_null_condition.sum()}")
    logging.info(f"Number of gene_expr_df: {len(gene_expr_df)} -> {len(gene_expr_df[~gene_expr_null_condition])}")
    gene_expr_df = gene_expr_df[~gene_expr_null_condition]

    # Remove gene name from gene_id in gene_expr_df (Maybe)
    # NOTE: from `TSPAN6;ENSG00000000003` to `ENSG00000000003`
    gene_expr_df.index = gene_expr_df.index.str.split(";").str[-1]
    # Remove duplicated gene_id in gene_expr_df
    if gene_expr_df.index.duplicated().any():
        raise ValueError("gene_expr_df has duplicated gene_id")

    gene_expr_filtered_gene_id_condition = gene_expr_df.index.isin(filtered_gene_id)
    logging.info(f"Number of gene_expr_df: {len(gene_expr_df)} -> {gene_expr_filtered_gene_id_condition.sum()}")
    gene_expr_df = gene_expr_df[gene_expr_filtered_gene_id_condition]

    # NOTE xk: filter by value, HVG
    rna_mean = gene_expr_df.mean(axis=1)
    rna_std = gene_expr_df.std(axis=1)
    mean_threshold = 0.1
    std_threshold = 0.1
    gene_expr_df = gene_expr_df[(rna_mean >= mean_threshold) & (rna_std >= std_threshold)]

    # NOTE xk: reorder gene by chr and pos
    # get chr pos from grch38_df
    gene_id_chr_pos_df = get_gene_id_chr_pos_df(grch38_df[condition])
    gene_id_chr_pos_df = gene_id_chr_pos_df.drop_duplicates(subset=["gene_id"])
    merged_gene_expr_df = gene_expr_df.copy()
    merged_gene_expr_df["gene_id"] = merged_gene_expr_df.index

    merged_gene_expr_df = merged_gene_expr_df.merge(gene_id_chr_pos_df, how="left", on="gene_id")
    merged_gene_expr_df = merged_gene_expr_df.sort_values(by=["chr", "pos"])
    gene_expr_df = merged_gene_expr_df.drop(columns=["chr", "pos"])
    gene_expr_df.set_index("gene_id", inplace=True)

    return gene_expr_df


def main(_):
    data_dir = Path(FLAGS.data_dir)
    save_dir = Path(FLAGS.save_dir)

    gene_expr_file_name = FLAGS.gene_expr_file_name
    gene_expr_file = data_dir / gene_expr_file_name

    logging.info(f"Loading gene expr and cpg bg from {data_dir}")
    gene_expr_df = pd.read_parquet(str(gene_expr_file))
    grch38_df = pd.read_parquet(str(FLAGS.grch38_parquet_file))
    logging.info(f"Gene expression dataframe shape: {gene_expr_df.shape}")

    # Set gene_name as index for gene expression dataframe
    gene_expr_df.rename(columns={"Unnamed: 0": "gene_name"}, inplace=True)
    gene_expr_df.set_index("gene_name", inplace=True)

    # Preprocess gene expression dataframe
    gene_expr_df = preprocess_gene_expr(gene_expr_df, grch38_df)

    # Get gene expression keys to index mappings
    gene_name_keys = gene_expr_df.index
    # gene_name_keys_to_idx = pd.Series(np.arange(len(gene_name_keys)), index=gene_name_keys)
    sample_name_keys = gene_expr_df.columns
    # sample_name_keys_to_idx = pd.Series(np.arange(len(sample_name_keys)), index=sample_name_keys)
    logging.info(f"Number of genes: {len(gene_name_keys)}")
    logging.info(f"Number of samples: {len(sample_name_keys)}")

    save_filtered_gene_expr_file = save_dir / gene_expr_file.with_suffix(".filtered.parquet").name
    if save_filtered_gene_expr_file.exists():
        if FLAGS.overwrite:
            logging.info(f"Overwriting {save_filtered_gene_expr_file}")
            save_filtered_gene_expr_file.unlink()
        else:
            logging.info(f"File {save_filtered_gene_expr_file} already exists. Skipping...")
            return

    logging.info(f"Saving filtered gene expression dataframe to {save_filtered_gene_expr_file}")
    gene_expr_df.to_parquet(save_filtered_gene_expr_file)


if __name__ == "__main__":
    app.run(main)
