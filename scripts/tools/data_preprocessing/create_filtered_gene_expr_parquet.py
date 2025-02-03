"""
E.g., After merging all the genes from TCGA array, epic, and wgbs, we filter genes by mean and std.

python scripts/tools/data_preprocessing/create_filtered_gene_expr_parquet.py \
    --input_gene_expr_parquet_file data/parquet/241231-tcga/gene_expr.parquet \
    --input_grch38_parquet_file data/parquet/grch38_hg38/grch38.v41.parquet \
    --output_dir data/parquet/241231-tcga/gene_stats \
    --output_file_name gene_stats \
    --overwrite --nofilter_non_protein_coding_gene \
    --plot_only
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags, logging


flags.DEFINE_string("input_gene_expr_parquet_file", None, "Directory containing the parquet files")
flags.mark_flag_as_required("input_gene_expr_parquet_file")
flags.DEFINE_string("input_grch38_parquet_file", None, "grch38 parquet file path")
flags.mark_flag_as_required("input_grch38_parquet_file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output directory")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_boolean("debug", False, "Enable debug mode")
flags.DEFINE_boolean("filter_non_protein_coding_gene", True, "Remove non-protein coding gene")
flags.DEFINE_float("mean_threshold", 0.1, "Mean threshold for filtering gene")
flags.DEFINE_float("std_threshold", 0.1, "Std threshold for filtering gene")
flags.DEFINE_boolean("plot_only", False, "Plot only")
flags.DEFINE_bool("ignore_duplicated_gene_id", False, "Ignore duplicated gene_id")


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
    condition = grch38_df["Feature"] == "gene"
    gene_id_chr_pos_df = grch38_df[condition][["gene_id", "Chromosome", "Start", "End", "Strand"]].copy()
    gene_id = gene_id_chr_pos_df["gene_id"].str.split(".").str[0]
    chr = grch38_df["Chromosome"].map(chromosome_order)
    pos = gene_id_chr_pos_df.apply(get_gene_position, axis=1)
    return pd.DataFrame({"gene_id": gene_id, "chr": chr, "pos": pos})


def preprocess_gene_expr(
    gene_expr_df,
    grch38_df,
    filter_non_protein_coding_gene=True,
    ignore_duplicated_gene_id=True,
):
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
    # NOTE: Only use grch (version) 38. The version number matters.
    if filter_non_protein_coding_gene:
        condition = (grch38_df["Feature"] == "gene") & (grch38_df["gene_type"] == "protein_coding")
    else:
        condition = grch38_df["Feature"] == "gene"
    filtered_gene_id = grch38_df[condition]["gene_id"]
    is_protein_coding = grch38_df[condition]["gene_type"] == "protein_coding"
    is_protein_coding.name = "is_protein_coding"
    # XXX xk: the gene_name can be gene_id
    gene_name = grch38_df[condition]["gene_name"]

    # Remove version number from gene_id in grch38_df
    # NOTE: from `ENSG00000186092.7` to `ENSG00000186092`
    filtered_gene_id = filtered_gene_id.apply(lambda x: x.split(".")[0])

    # Remove duplicated gene_id in grch38_df
    duplicated_gene_id_condition = filtered_gene_id.duplicated(keep=False)
    duplicated_gene_id = grch38_df[condition]["gene_id"][duplicated_gene_id_condition]
    logging.info(f"Number of duplicated gene_id in grch38: {len(duplicated_gene_id)}")
    logging.info(duplicated_gene_id)

    logging.info(
        f"Number of gene_id after deduplicate gene_id due to version number: {len(filtered_gene_id)} -> {len(filtered_gene_id[~duplicated_gene_id_condition])}"
    )
    filtered_gene_id = filtered_gene_id[~duplicated_gene_id_condition]
    is_protein_coding = is_protein_coding[~duplicated_gene_id_condition]
    gene_name = gene_name[~duplicated_gene_id_condition]
    if filtered_gene_id.duplicated().any():
        raise ValueError("filtered_gene_id has duplicated gene_id")

    # Remove null gene_id in gene_expr_df
    gene_expr_null_condition = gene_expr_df.index.isnull()
    logging.info(f"Number of null gene_id: {gene_expr_null_condition.sum()}")
    logging.info(
        f"Number of gene_expr_df after remove null gene_id: {len(gene_expr_df)} -> {len(gene_expr_df[~gene_expr_null_condition])}"
    )
    gene_expr_df = gene_expr_df[~gene_expr_null_condition]

    gene_is_protein_coding = pd.concat([filtered_gene_id, is_protein_coding, gene_name], axis=1)
    if not gene_is_protein_coding.index.equals(filtered_gene_id.index):
        raise ValueError("gene_is_protein_coding index is not the same as filtered_gene_id index")
    if not gene_is_protein_coding.index.equals(is_protein_coding.index):
        raise ValueError("gene_is_protein_coding index is not the same as is_protein_coding index")
    gene_is_protein_coding = gene_is_protein_coding.set_index("gene_id")

    # Remove gene name from gene_id in gene_expr_df (Maybe)
    gene_name_id_from_df = gene_expr_df.index.copy()
    # NOTE: from `TSPAN6;ENSG00000000003` to `ENSG00000000003`
    gene_id_from_df = gene_expr_df.index.str.split(";").str[-1]
    # NOTE: from `TSPAN6;ENSG00000000003.1` to `ENSG00000000003`
    gene_id_from_df = gene_id_from_df.str.split(".").str[0]

    # Remove duplicated gene_id in gene_expr_df
    gene_expr_filtered_gene_id_condition = gene_id_from_df.isin(filtered_gene_id)
    logging.info(
        f"Number of gene_expr_df after filtering: {len(gene_expr_df)} -> {gene_expr_filtered_gene_id_condition.sum()}"
    )
    gene_expr_df = gene_expr_df[gene_expr_filtered_gene_id_condition]
    filtered_gene_id_from_df = gene_id_from_df[gene_expr_filtered_gene_id_condition]
    filtered_gene_name_id_from_df = gene_name_id_from_df[gene_expr_filtered_gene_id_condition]

    if filtered_gene_id_from_df.duplicated().any():
        duplicated_gene_name_id_in_df = gene_name_id_from_df[filtered_gene_id_from_df.duplicated(keep=False)]
        raise ValueError(f"gene_expr_df has {len(duplicated_gene_name_id_in_df)}: {duplicated_gene_name_id_in_df}")

    filtered_gene_name_id_from_df = rebuild_gene_name_id_without_version(filtered_gene_name_id_from_df)
    gene_is_protein_coding = gene_is_protein_coding.loc[filtered_gene_id_from_df]
    if not gene_is_protein_coding.index.equals(filtered_gene_id_from_df):
        raise ValueError("gene_is_protein_coding index is not the same as filtered gene_expr_df index")

    # Filter MT, PR, RPS, RPL genes
    gene_expr_df, gene_is_protein_coding = filter_unwanted_genes(gene_expr_df, gene_is_protein_coding)

    return gene_expr_df, duplicated_gene_id, gene_is_protein_coding


def rebuild_gene_name_id_without_version(gene_id_index):
    gene_name = gene_id_index.str.split(";").str[0]
    gene_name = gene_name.str.split(".").str[0]
    gene_id_version = gene_id_index.str.split(";").str[1]
    gene_id = gene_id_version.str.split(".").str[0]
    return gene_name + ";" + gene_id


def sort_gene_by_chr_pos(gene_expr_df, grch38_df):
    gene_id_chr_pos_df = get_gene_id_chr_pos_df(grch38_df)
    gene_id_chr_pos_df = gene_id_chr_pos_df.drop_duplicates(subset=["gene_id"])
    merged_gene_expr_df = gene_expr_df.copy()

    merged_gene_expr_df = merged_gene_expr_df.merge(gene_id_chr_pos_df, how="left", on="gene_id")
    merged_gene_expr_df = merged_gene_expr_df.sort_values(by=["chr", "pos"])
    gene_expr_df = merged_gene_expr_df.drop(columns=["chr", "pos"])
    gene_expr_df.set_index("gene_id", inplace=True)
    return gene_expr_df


def filter_by_value_hvg(gene_expr_df, gene_is_protein_coding, mean_threshold=0.1, std_threshold=0.1):
    old_num_genes = gene_expr_df.shape[0]
    rna_mean = gene_expr_df.mean(axis=1)
    rna_std = gene_expr_df.std(axis=1)

    condition_gene_name_id_version = (rna_mean >= mean_threshold) & (rna_std >= std_threshold)
    # NOTE: from `TSPAN6;ENSG00000000003.1` to `ENSG00000000003`
    condition_gene_id = condition_gene_name_id_version.copy()
    condition_gene_id.index = condition_gene_name_id_version.index.str.split(";").str[-1].str.split(".").str[0]

    gene_expr_df = gene_expr_df[condition_gene_name_id_version]
    gene_is_protein_coding = gene_is_protein_coding[condition_gene_id]
    new_num_genes = gene_expr_df.shape[0]
    logging.info(
        f"Number of genes after filtering by mean ({mean_threshold}) and std ({std_threshold}): {old_num_genes} -> {new_num_genes}"
    )
    return gene_expr_df, gene_is_protein_coding


REMOVE_GENE_PREFIX_LIST = ["MT", "PR", "RPS", "RPL"]


def filter_unwanted_genes(gene_expr_df, gene_is_protein_coding):
    """
    Remove genes
    - MT- (Mitochondrial Genes)
    - PR- (Proline-Rich Genes)
    - RPS- / RPL- (Ribosomal Protein Genes)
    """
    for gene_prefix in REMOVE_GENE_PREFIX_LIST:
        gene_expr_df, gene_is_protein_coding = _remove_gene(gene_expr_df, gene_is_protein_coding, gene_prefix)
    return gene_expr_df, gene_is_protein_coding


def _remove_gene(gene_expr_df, gene_is_protein_coding, gene_prefix):
    prev_len = len(gene_expr_df)
    remove_gene_cond = gene_expr_df.index.str.startswith(gene_prefix)

    gene_expr_df = gene_expr_df[~remove_gene_cond]
    gene_is_protein_coding = gene_is_protein_coding[~remove_gene_cond]
    current_len = len(gene_expr_df)
    logging.info(f"Number of genes removed with prefix {gene_prefix}: {prev_len} -> {current_len}")
    return gene_expr_df, gene_is_protein_coding


def main(_):
    input_gene_expr_parquet_file = Path(FLAGS.input_gene_expr_parquet_file)
    input_grch38_parquet_file = Path(FLAGS.input_grch38_parquet_file)

    output_dir = Path(FLAGS.output_dir)
    output_file_path = output_dir / FLAGS.output_file_name

    if output_file_path.exists():
        if not FLAGS.overwrite:
            logging.info(f"{output_file_path} already exists. Skipping...")
            return
        else:
            logging.info(f"Overwriting {output_file_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading gene expr from {input_gene_expr_parquet_file}")
    gene_expr_df = pd.read_parquet(input_gene_expr_parquet_file)
    logging.info(f"Gene expression dataframe shape: {gene_expr_df.shape}")

    logging.info(f"Loading grch38 from {input_grch38_parquet_file}")
    grch38_df = pd.read_parquet(input_grch38_parquet_file)

    # Set gene_id as index for gene expression dataframe
    gene_expr_df.rename(columns={"Unnamed: 0": "gene_id"}, inplace=True)
    gene_expr_df.set_index("gene_id", inplace=True)

    # Preprocess gene expression dataframe
    logging.info("Preprocessing gene expression dataframe")
    gene_expr_df, duplicated_gene_id, gene_is_protein_coding = preprocess_gene_expr(
        gene_expr_df, grch38_df, FLAGS.filter_non_protein_coding_gene, FLAGS.ignore_duplicated_gene_id
    )

    logging.info("Plotting gene stats")
    if FLAGS.filter_non_protein_coding_gene:
        file_name = f"{input_gene_expr_parquet_file.parent.name}-protein_coding_gene"
    else:
        file_name = f"{input_gene_expr_parquet_file.parent.name}-full_gene"
    plot_gene_stats(input_gene_expr_parquet_file, output_dir, gene_expr_df, file_name)
    if FLAGS.plot_only:
        logging.info("Plot only mode. Skipping...")
        return

    # NOTE xk: filter by value, HVG
    mean_threshold = FLAGS.mean_threshold
    std_threshold = FLAGS.std_threshold
    logging.info("Filtering gene by mean and std")
    gene_expr_df, gene_is_protein_coding = filter_by_value_hvg(
        gene_expr_df, gene_is_protein_coding, mean_threshold, std_threshold
    )

    logging.info("Save gene stats after filtering")
    save_gene_stats_after_filtering(
        output_dir,
        gene_is_protein_coding,
        file_name,
        mean_threshold,
        std_threshold,
    )

    # NOTE xk: reorder gene by chr and pos
    # get chr pos from grch38_df
    logging.info("Reordering gene by chr and pos")
    gene_expr_df = sort_gene_by_chr_pos(gene_expr_df, grch38_df)

    # Get gene expression keys to index mappings
    gene_id_keys = gene_expr_df.index
    # gene_id_keys_to_idx = pd.Series(np.arange(len(gene_id_keys)), index=gene_id_keys)
    sample_name_keys = gene_expr_df.columns
    # sample_name_keys_to_idx = pd.Series(np.arange(len(sample_name_keys)), index=sample_name_keys)
    logging.info(f"Number of genes: {len(gene_id_keys)}")
    logging.info(f"Number of samples: {len(sample_name_keys)}")

    logging.info(f"Saving filtered gene expression dataframe to {output_file_path}")
    gene_expr_df.to_parquet(output_file_path)

    logging.info(f"Saving duplicated gene_id to {output_dir / 'duplicated_gene_id.csv'}")
    duplicated_gene_id.to_csv(output_dir / "duplicated_gene_id.csv")


def plot_gene_stats(
    input_gene_expr_parquet_file,
    output_dir,
    gene_expr_df,
    file_name,
):
    gene_expr_mean = gene_expr_df.mean(axis=1)
    gene_expr_std = gene_expr_df.std(axis=1)

    sns.histplot(gene_expr_mean)
    sns.histplot(gene_expr_std)
    if FLAGS.filter_non_protein_coding_gene:
        histplot_file_path = output_dir / "gene_expr_mean_std-protein_coding_gene-histplot.png"
        histplot_title = (
            f"Gene expression mean and std - {input_gene_expr_parquet_file.parent.name} (protein coding gene)"
        )
    else:
        histplot_file_path = output_dir / "gene_expr_mean_std-full_gene-histplot.png"
        histplot_title = f"Gene expression mean and std - {input_gene_expr_parquet_file.parent.name} (full gene)"
    plt.title(histplot_title)
    plt.savefig(histplot_file_path)
    logging.info(f"Saving gene expression mean and std histplot to {histplot_file_path}")
    plt.cla()

    sns.boxenplot(gene_expr_mean)
    sns.boxenplot(gene_expr_std)
    if FLAGS.filter_non_protein_coding_gene:
        histplot_file_path = output_dir / "gene_expr_mean_std-protein_coding_gene-boxenplot.png"
        histplot_title = (
            f"Gene expression mean and std - {input_gene_expr_parquet_file.parent.name} (protein coding gene)"
        )
    else:
        histplot_file_path = output_dir / "gene_expr_mean_std-full_gene-boxenplot.png"
        histplot_title = f"Gene expression mean and std - {input_gene_expr_parquet_file.parent.name} (full gene)"
    plt.title(histplot_title)
    plt.savefig(histplot_file_path)
    logging.info(f"Saving gene expression mean and std boxenplot to {histplot_file_path}")
    plt.cla()

    num_genes = gene_expr_df.shape[0]
    qauntiles = np.linspace(0, 1, 21)

    gene_expr_mean_quantiles = gene_expr_mean.quantile(qauntiles)
    gene_expr_mean_quantiles = pd.DataFrame(gene_expr_mean_quantiles, columns=["mean-quantile_value"])

    gene_expr_mean_quantiles["mean-quantile_to_index"] = num_genes * pd.Series(
        qauntiles, index=gene_expr_mean_quantiles.index
    )
    gene_expr_mean_quantiles["mean-quantile_to_index"] = gene_expr_mean_quantiles["mean-quantile_to_index"].astype(int)
    gene_expr_mean_quantiles["num_remaining_genes"] = num_genes - gene_expr_mean_quantiles["mean-quantile_to_index"]

    gene_expr_std_quantiles = gene_expr_std.quantile(qauntiles)
    gene_expr_std_quantiles = pd.DataFrame(gene_expr_std_quantiles, columns=["std-quantile_value"])
    gene_expr_std_quantiles["std-quantile_to_index"] = num_genes * pd.Series(
        qauntiles, index=gene_expr_mean_quantiles.index
    )
    gene_expr_std_quantiles["std-quantile_to_index"] = gene_expr_std_quantiles["std-quantile_to_index"].astype(int)
    gene_expr_std_quantiles["num_remaining_genes"] = num_genes - gene_expr_std_quantiles["std-quantile_to_index"]

    gene_expr_quantiles = pd.concat([gene_expr_mean_quantiles, gene_expr_std_quantiles], axis=1)
    gene_expr_quantiles.index.name = "filtering_ratio_by_num_genes_quantile"

    output_gene_expr_quantiles_file_path = output_dir / f"{file_name}-gene_expr_quantiles.csv"
    gene_expr_quantiles.to_csv(output_gene_expr_quantiles_file_path)
    logging.info(f"Saving gene expression quantiles to: {output_gene_expr_quantiles_file_path}")


def save_gene_stats_after_filtering(
    output_dir,
    gene_is_protein_coding,
    file_name,
    mean_threshold,
    std_threshold,
):
    gene_is_protein_coding = gene_is_protein_coding.reset_index()

    gene_is_protein_coding_file_path = output_dir / f"{file_name}-gene_is_protein_coding.csv"
    gene_is_protein_coding.to_csv(gene_is_protein_coding_file_path)
    logging.info(f"Saving gene_is_protein_coding to {gene_is_protein_coding_file_path}")

    gene_is_protein_coding_ratio_file_path = output_dir / f"{file_name}-gene_is_protein_coding_ratio.log"
    with open(gene_is_protein_coding_ratio_file_path, "w") as f:
        f.write(f"num genes: {len(gene_is_protein_coding)}\n")
        f.write(f"mean_threshold: {mean_threshold}\n")
        f.write(f"std_threshold: {std_threshold}\n")
        gene_is_protein_coding_ratio = gene_is_protein_coding["is_protein_coding"].value_counts(normalize=False)
        f.write(f"gene_is_protein_coding_ratio:\n{gene_is_protein_coding_ratio}\n")
        gene_is_protein_coding_ratio = gene_is_protein_coding["is_protein_coding"].value_counts(normalize=True)
        f.write(f"gene_is_protein_coding_ratio after normalized:\n{gene_is_protein_coding_ratio}\n")
    logging.info(f"Saving gene_is_protein_coding to {gene_is_protein_coding_file_path}")


if __name__ == "__main__":
    app.run(main)
