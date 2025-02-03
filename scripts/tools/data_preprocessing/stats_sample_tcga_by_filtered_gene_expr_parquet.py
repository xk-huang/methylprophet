"""
set -e

python scripts/tools/compare_two_df.py --a data/parquet/241231-tcga_array/cancer_type.parquet --b data/parquet/241231-tcga_epic/cancer_type.parquet

python scripts/tools/compare_two_df.py --a data/parquet/241231-tcga_array/cancer_type.parquet --b data/parquet/241231-tcga_wgbs/cancer_type.parquet

python scripts/tools/data_preprocessing/stats_sample_tcga_by_filtered_gene_expr_parquet.py \
    --input_parquet_file data/processed/241231-tcga/gene_expr.filtered.parquet \
    --input_cancer_type_file data/parquet/241231-tcga_array/cancer_type.parquet \
    --output_dir data/parquet/241231-tcga/metadata \
    --output_file_name sample_split \
    --assign_na_tissue_to_unknown_samples

python scripts/tools/data_preprocessing/merge_sample_tissue_count_with_idx.py \
    --input_sample_tissue_counts_iwth_idx_files data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_path data/processed/241231-tcga/sample_tissue_count_with_idx.csv
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")
flags.DEFINE_string("input_cancer_type_file", None, "Path to the cancer type parquet file")
flags.mark_flag_as_required("input_cancer_type_file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output file name")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_integer("sample_idx_offset", 0, "Sample index offset")
flags.DEFINE_integer("tissue_idx_offset", 0, "Tissue index offset")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("assign_na_tissue_to_unknown_samples", False, "Assign unknown samples to NA")

FLAGS = flags.FLAGS


def main(_):
    input_parquet_file = Path(FLAGS.input_parquet_file)
    input_cancer_type_file = Path(FLAGS.input_cancer_type_file)

    output_dir = Path(FLAGS.output_dir)
    output_file_name = FLAGS.output_file_name
    output_dir = output_dir / output_file_name

    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_dir} already exists. Skipping...")
            return
        else:
            logging.warning(f"Overwriting {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Reading {input_parquet_file}")
    df = pd.read_parquet(input_parquet_file)
    cancer_type_df = pd.read_parquet(input_cancer_type_file)

    # Get the sample names
    columns = df.columns  # NOTE: in gene_expr, the first column is not "Unnamed: 0"

    # Sort the sample names
    columns = columns.sort_values()
    gene_expr_sample_names = pd.Series(columns)
    sorted_cancer_type_df = cancer_type_df.sort_values("Sample").reset_index(drop=True)
    if not gene_expr_sample_names.isin(sorted_cancer_type_df["Sample"]).all():
        # NOTE: samples in gene expr has no corresponding tissue name
        unknown_samples = gene_expr_sample_names[~gene_expr_sample_names.isin(sorted_cancer_type_df["Sample"])]
        unknown_samples_path = output_dir / "unknown_samples_from_cancer_type.csv"
        unknown_samples.to_csv(unknown_samples_path)
        logging.info(f"Saved unknown samples to {unknown_samples_path}")
        if FLAGS.assign_na_tissue_to_unknown_samples:
            logging.warning(f"Assign NA tissue to unknown {len(unknown_samples)} samples. See {unknown_samples_path}")
            na_cancer_type_df = pd.DataFrame(
                {
                    "Sample": unknown_samples,
                    "Cancer": "NA",
                }
            )
            sorted_cancer_type_df = pd.concat([sorted_cancer_type_df, na_cancer_type_df], ignore_index=True)
        else:
            raise ValueError(f"Unknown {len(unknown_samples)} samples. See {unknown_samples_path}")

    # Get cancer names
    select_condition = sorted_cancer_type_df["Sample"].isin(gene_expr_sample_names)
    selected_sorted_cancer_type_df = sorted_cancer_type_df[select_condition].reset_index()
    if (
        not selected_sorted_cancer_type_df["Sample"]
        .sort_values()
        .reset_index(drop=True)
        .equals(gene_expr_sample_names.sort_values().reset_index(drop=True))
    ):
        raise ValueError(
            "Sample names are not equal between the input parquet file and the cancer type file, after use samples for indexing."
        )
    cancer_names = selected_sorted_cancer_type_df["Cancer"]
    sample_names = selected_sorted_cancer_type_df["Sample"]

    # NOTE: sample_names is not the sample_names in cancer_type_df, is the sorted sample_names from the gene expression data
    # Unlike the ENCODE, where the cancer_names is from samples in gene_expr data, here the cancer_names is from the cancer_type_df
    sample_tissue_with_sample_idx_df = pd.DataFrame(
        {
            "sample_name": sample_names,
            "tissue_name": cancer_names,
        }
    )
    sample_tissue_with_sample_idx_df.reset_index(names="sample_idx", inplace=True)
    sample_idx_offset = FLAGS.sample_idx_offset
    logging.info(f"Adding sample_idx offset: {sample_idx_offset}")
    sample_tissue_with_sample_idx_df["sample_idx"] += sample_idx_offset
    sample_tissue_with_sample_idx_df_path = output_dir / "sample_with_idx.csv"
    sample_tissue_with_sample_idx_df.to_csv(sample_tissue_with_sample_idx_df_path)
    logging.info(f"Saved {sample_tissue_with_sample_idx_df_path}")

    # NOTE: save sample name, tissue name, count, and index
    sample_tissue_with_sample_idx_df["sample_tissue_same"] = (
        sample_tissue_with_sample_idx_df["sample_name"] == sample_tissue_with_sample_idx_df["tissue_name"]
    )
    tissue_name_series = cancer_names
    tissue_names_value_counts_with_idx = pd.Series(tissue_name_series).value_counts(sort=False)
    tissue_names_value_counts_with_idx.index.name = "tissue_name"
    tissue_names_value_counts_with_idx = tissue_names_value_counts_with_idx.reset_index()
    tissue_names_value_counts_with_idx.reset_index(names="tissue_idx", inplace=True)
    tissue_idx_offset = FLAGS.tissue_idx_offset
    logging.info(f"Adding tissue_idx offset: {tissue_idx_offset}")
    tissue_names_value_counts_with_idx["tissue_idx"] += tissue_idx_offset
    tissue_names_value_counts_with_idx.set_index(keys="tissue_name", inplace=True)

    sample_tissue_with_sample_idx_df = sample_tissue_with_sample_idx_df.set_index("tissue_name")
    sample_tissue_count_with_idx_df = pd.merge(
        sample_tissue_with_sample_idx_df,
        tissue_names_value_counts_with_idx,
        how="left",
        left_index=True,
        right_index=True,
    )
    sample_tissue_count_with_idx_df.reset_index(inplace=True)

    sample_tissue_count_with_idx_df_path = output_dir / "sample_tissue_count_with_idx.csv"
    sample_tissue_count_with_idx_df.to_csv(sample_tissue_count_with_idx_df_path)
    logging.info(f"Saved {sample_tissue_count_with_idx_df_path}")

    sample_tissue_count_with_idx_df_sort_by_count_path = (
        output_dir / "sample_tissue_count_with_idx-sorted_by_count.csv"
    )
    sample_tissue_count_with_idx_df.sort_values(by="count").to_csv(sample_tissue_count_with_idx_df_sort_by_count_path)
    tissue_names_path = output_dir / "tissue_names.json"
    with open(tissue_names_path, "w") as f:
        json.dump(sample_tissue_count_with_idx_df["tissue_name"].unique().tolist(), f, indent=4)

    # Plot the tissue_name value counts
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
    sns.barplot(tissue_names_value_counts_with_idx["count"].sort_values(ascending=False), ax=ax)
    tissue_with_samples_more_than_one = tissue_names_value_counts_with_idx[
        tissue_names_value_counts_with_idx["count"] > 1
    ].shape[0]
    ax.axvline(tissue_with_samples_more_than_one - 0.5, color="red", ls="--")  # Add a vertical line at x=22
    ax.text(
        tissue_with_samples_more_than_one - 0.5, 2, f"# tissue with > 1 sample: {tissue_with_samples_more_than_one}"
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right"
    )  # Rotate labels by 45 degrees and align them to the right
    plt.tight_layout()  # Adjust the layout

    fig.savefig(output_dir / "tissue_name_value_counts.png")
    print(f"Save the figure to {output_dir / 'tissue_name_value_counts.png'}")

    # Validate saved results
    sample_tissue_with_sample_idx_df = pd.read_csv(sample_tissue_with_sample_idx_df_path, index_col=0)
    sample_tissue_count_with_idx_df = pd.read_csv(sample_tissue_count_with_idx_df_path, index_col=0)
    if not sample_tissue_with_sample_idx_df["sample_name"].equals(sample_tissue_count_with_idx_df["sample_name"]):
        raise ValueError("sample_name is not equal between original and processed dataframes")
    if not sample_tissue_with_sample_idx_df["tissue_name"].equals(sample_tissue_count_with_idx_df["tissue_name"]):
        raise ValueError("tissue_name is not equal between original and processed dataframes")
    if not sample_tissue_with_sample_idx_df.index.equals(sample_tissue_count_with_idx_df.index):
        raise ValueError("index is not equal between original and processed dataframes")

    # Again, check with input_cancer_type_file
    cancer_type_df = pd.read_parquet(input_cancer_type_file)
    cancer_type_df = cancer_type_df.sort_values("Sample").reset_index(drop=True)
    cancer_type_df = cancer_type_df.set_index("Sample")
    cancer_type_series = cancer_type_df["Cancer"]
    # NOTE xk: 'NA' is still string 'NA' in the cancer_type_series, not interpreted as np.nan
    sample_tissue_count_with_idx_df = pd.read_csv(sample_tissue_count_with_idx_df_path, index_col=0, na_filter=False)
    sample_tissue_count_with_idx_df = sample_tissue_count_with_idx_df.sort_values("sample_name").reset_index(drop=True)
    sample_tissue_count_with_idx_df = sample_tissue_count_with_idx_df.set_index("sample_name")

    joint_index = cancer_type_series.index.intersection(sample_tissue_count_with_idx_df.index)
    cancer_type_series = cancer_type_series.loc[joint_index]
    sample_tissue_count_with_idx_df = sample_tissue_count_with_idx_df.loc[joint_index]
    cancer_type_series.index.name = "sample_name"
    cancer_type_series.name = "tissue_name"
    if not cancer_type_series.equals(sample_tissue_count_with_idx_df["tissue_name"]):
        raise ValueError("tissue_name is not equal between cancer_type_series and processed dataframes")

    if not sample_tissue_count_with_idx_df["tissue_name"].isin(cancer_type_series).all():
        raise ValueError("some `sample_name:tissue_name` pairs are not in the cancer_type_series")


if __name__ == "__main__":
    app.run(main)
