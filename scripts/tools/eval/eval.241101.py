"""
pip install absl-py

python misc/eval.241101.py \
    --input_result_csv outputs/241025-predictive_model-new_encode_wgbs/eval-241023-encode_wgbs-train_0_01_val_0_01-ind_tissue-pred_model-cu_insomnia-1xh100_80gb-num_nbase_2400-train_num_shards_2831/eval/version_1/eval_results-test-val_cpg_val_sample.csv \
    --input_sample_name_to_idx_csv data/parquet/241023-encode_wgbs/metadata/name_to_idx/sample_name_to_idx.csv \
    --input_cpg_chr_pos_to_idx_csv data/parquet/241023-encode_wgbs/metadata/name_to_idx/cpg_chr_pos_to_idx.csv \
    --output_dir misc/eval/
"""

import multiprocessing as mp
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags, logging
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


flags.DEFINE_string("input_result_csv", None, "Input result csv file")
flags.mark_flag_as_required("input_result_csv")

flags.DEFINE_string("input_sample_name_to_idx_csv", None, "Input sample name to idx csv file")
flags.mark_flag_as_required("input_sample_name_to_idx_csv")

flags.DEFINE_string("input_cpg_chr_pos_to_idx_csv", None, "Input cpg chr pos to idx csv file")
flags.mark_flag_as_required("input_cpg_chr_pos_to_idx_csv")


flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")
flags.DEFINE_bool("only_plot", False, "Only plot the figures")

FLAGS = flags.FLAGS


def read_data(input_result_csv, input_sample_name_to_idx_csv, input_cpg_chr_pos_to_idx_csv):
    combined_df = pd.read_csv(input_result_csv)
    df_sid = pd.read_csv(input_sample_name_to_idx_csv)
    df_sid.columns = ["tissue", "sample_id"]
    df_cid = pd.read_csv(input_cpg_chr_pos_to_idx_csv)
    df_cid.columns = ["cpg", "cpg_id"]
    combined_df = combined_df.iloc[:, 1:]
    combined_df = combined_df.merge(df_sid, on="sample_id", how="left")
    combined_df = combined_df.merge(df_cid, on="cpg_id", how="left")
    combined_df = combined_df[["pred_methyl", "gt_methyl", "tissue", "cpg"]]
    combined_df = combined_df.rename(columns={"pred_methyl": "pred", "gt_methyl": "gs"})

    logging.info("Data join completed")

    ## pivot into gs and pred separately
    df_pred = combined_df.pivot(index="cpg", columns="tissue", values="pred")
    df_gs = combined_df.pivot(index="cpg", columns="tissue", values="gs")
    df_pred.columns = [col.split("Homo sapiens ")[1].split(" tissue")[0] for col in df_pred.columns]
    df_gs.columns = [col.split("Homo sapiens ")[1].split(" tissue")[0] for col in df_gs.columns]
    df_pred = df_pred.reset_index()
    df_gs = df_gs.reset_index()
    df_pred[["Chromosome", "Start"]] = df_pred.iloc[:, 0].str.split("_", expand=True)
    df_pred["Start"] = df_pred["Start"].astype(float).astype(int)
    df_pred["End"] = df_pred["Start"] + 1
    df_gs[["Chromosome", "Start"]] = df_gs.iloc[:, 0].str.split("_", expand=True)
    df_gs["Start"] = df_gs["Start"].astype(float).astype(int)
    df_gs["End"] = df_gs["Start"] + 1
    df_pred = df_pred.sort_values(by="Start", ascending=True).reset_index(drop=True)
    df_gs = df_gs.sort_values(by="Start", ascending=True).reset_index(drop=True)

    def rename_duplicate_columns(df):
        cols = pd.Series(df.columns)
        counts = {}
        for i, col in enumerate(cols):
            if col in counts:
                counts[col] += 1
                cols[i] = f"{col}_sample{counts[col]}"
            else:
                counts[col] = 1

        df.columns = cols
        return df

    df_pred = rename_duplicate_columns(df_pred)
    df_gs = rename_duplicate_columns(df_gs)
    return df_pred, df_gs


def save_bedfile(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with mp.Pool(mp.cpu_count()) as pool:
        results = []
        for tissue in df.columns[4:]:
            # wirte_bedfile_one_tissue(df, output_dir, tissue)
            result = pool.apply_async(wirte_bedfile_one_tissue, args=(df, output_dir, tissue))
            results.append(result)

        for result in results:
            tissue = result.get()
            print(f"bedGraph file created for {tissue} in {output_dir}")


def wirte_bedfile_one_tissue(df, output_dir, tissue):
    bedgraph_lines = []

    for idx, row in df.iterrows():
        chr_name = row["Chromosome"]
        start = int(row["Start"])
        end = int(row["End"])
        methylation_value = row[tissue]

        if isinstance(methylation_value, (int, float)):
            bedgraph_lines.append(f"{chr_name}\t{start}\t{end}\t{methylation_value}\n")

    filename = f"{output_dir}/{tissue}.bedGraph"
    with open(filename, "w") as f:
        f.writelines(bedgraph_lines)
    return tissue


def plot_something(df, output_dir, output_name):
    genome_length = 248956422  # chr length as example
    bin_size = 3000  # due to local memory limit
    df_methylation = df.copy()
    df_methylation = df_methylation.iloc[:, 1:]
    df_methylation["Bin"] = pd.cut(df_methylation["Start"], bins=np.arange(0, genome_length, bin_size))
    percent_mCG = df_methylation.groupby(["Chromosome", "Bin"]).mean()
    percent_mCG = percent_mCG.drop(columns=["Start", "End"])
    percent_mCG = percent_mCG.reset_index()
    percent_mCG = percent_mCG.iloc[1:, :]
    percent_mCG_long = pd.melt(percent_mCG, id_vars=["Chromosome", "Bin"], var_name="Tissue", value_name="mCG")
    percent_mCG_long = percent_mCG_long.dropna(subset=["mCG"])

    plt.figure(figsize=(12, 12))
    sns.boxplot(x="Tissue", y="mCG", data=percent_mCG_long, showfliers=False)
    plt.ylim(0, 1)
    plt.axhline(0.80, linestyle="--", color="red", label="80% mCG")
    plt.axhline(0.90, linestyle="--", color="blue", label="90% mCG")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Length of 3000 bp Bin Level Distribution Across Samples {output_name}")
    plt.tight_layout()
    save_fig_path = output_dir / f"boxplotbin_1031_3000bp_{output_name}.png"
    plt.savefig(save_fig_path, format="png", bbox_inches="tight")
    logging.info(f"Saving boxplot to {save_fig_path}")

    ## sample level bin correlation clustering
    percent_mCG_matrix = percent_mCG.drop(columns=["Chromosome", "Bin"])
    correlation_matrix = percent_mCG_matrix.corr(method="pearson")
    Z = linkage(correlation_matrix, method="ward")
    max_clusters = 7
    clusters = fcluster(Z, max_clusters, criterion="maxclust")
    clustered_tissues = pd.DataFrame({"Tissue": correlation_matrix.columns, "Cluster": clusters})

    def leaf_label_func(id):
        tissue_name = correlation_matrix.columns[id]
        cluster_label = clustered_tissues[clustered_tissues["Tissue"] == tissue_name]["Cluster"].values[0]
        return f"{tissue_name}"

    plt.figure(figsize=(8, 8))
    dendrogram(
        Z,
        labels=correlation_matrix.columns,
        leaf_rotation=90,
        leaf_label_func=leaf_label_func,
        color_threshold=Z[-max_clusters, 2],
    )
    plt.title(f"Length of 3000 bp Bin Level Dendrogram Across Samples {output_name}")
    plt.tight_layout()
    output_fig_path = output_dir / f"dendrogram_1031_3000bp_{output_name}.png"
    plt.savefig(output_fig_path, format="png", bbox_inches="tight")
    logging.info(f"Saving dendrogram to {output_fig_path}")


def main(_):
    input_result_csv = FLAGS.input_result_csv
    input_sample_name_to_idx_csv = FLAGS.input_sample_name_to_idx_csv
    input_cpg_chr_pos_to_idx_csv = FLAGS.input_cpg_chr_pos_to_idx_csv
    output_dir = Path(FLAGS.output_dir)

    overwrite = FLAGS.overwrite
    only_plot = FLAGS.only_plot

    ## data join
    df_pred, df_gs = read_data(input_result_csv, input_sample_name_to_idx_csv, input_cpg_chr_pos_to_idx_csv)

    ## tissue level bedfile generation
    def rename_and_aggregate_columns(df):
        metadata_columns = ["cpg", "Chromosome", "Start", "End"]
        tissue_columns = [col for col in df.columns if col not in metadata_columns]
        tissue_groups = {}
        for col in tissue_columns:
            base_name = col.split("_")[0]
            if base_name in tissue_groups:
                tissue_groups[base_name].append(col)
            else:
                tissue_groups[base_name] = [col]

        df_aggregated = df[metadata_columns].copy()

        for tissue, cols in tissue_groups.items():
            if len(cols) > 1:
                df_aggregated[tissue] = df[cols].mean(axis=1)
            else:
                df_aggregated[tissue] = df[cols[0]]

        return df_aggregated

    if not only_plot:
        logging.info("Start generating bedGraph files for tissue level data")

        df_pred_t = df_pred.copy()
        df_pred_t = rename_and_aggregate_columns(df_pred_t)
        df_gs_t = df_gs.copy()
        df_gs_t = rename_and_aggregate_columns(df_gs_t)
        save_bedfile(df_pred_t, output_dir / "bedgraphfiles_1031_pred")
        save_bedfile(df_gs_t, output_dir / "bedgraphfiles_1031_gs")

        ## sample level bedfile generation
        df_pred_s = df_pred.copy()
        df_gs_s = df_gs.copy()
        save_bedfile(df_pred_s, output_dir / "bedgraphfiles_1031_pred_s")
        save_bedfile(df_gs_s, output_dir / "bedgraphfiles_1031_gs_s")

    ## bin/region level plot

    plot_something(df_pred.copy(), output_dir, "pred")
    plot_something(df_gs.copy(), output_dir, "gs")


if __name__ == "__main__":
    app.run(main)
