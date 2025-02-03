import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output file name")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")

FLAGS = flags.FLAGS


def main(_):
    input_parquet_file = Path(FLAGS.input_parquet_file)

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

    # Get the tissue names
    columns = df.columns[1:]  # NOTE: Remove the first column, "Unnamed: 0", which is the cpg_chr_pos
    # Sort the sample names
    sample_names = sorted(columns)
    trimmed_sample_names = [x.split("Homo sapiens ")[-1] for x in columns]
    tissue_names = [x.split(" tissue")[0] for x in trimmed_sample_names]
    tissue_names = sorted(tissue_names)

    sample_tissue_with_sample_idx_df = pd.DataFrame(
        {
            "sample_name": sample_names,
            "tissue_name": tissue_names,
        }
    )
    sample_tissue_with_sample_idx_df.reset_index(names="sample_idx", inplace=True)
    sample_tissue_with_sample_idx_df_path = output_dir / "sample_with_idx.csv"
    sample_tissue_with_sample_idx_df.to_csv(sample_tissue_with_sample_idx_df_path)
    logging.info(f"Saved {sample_tissue_with_sample_idx_df_path}")

    # NOTE: save sample name, tissue name, count, and index
    sample_tissue_with_sample_idx_df["sample_tissue_same"] = (
        sample_tissue_with_sample_idx_df["sample_name"] == sample_tissue_with_sample_idx_df["tissue_name"]
    )
    tissue_name_series = pd.Series(tissue_names)
    tissue_names_value_counts_with_idx = pd.Series(tissue_name_series).value_counts(sort=False)
    tissue_names_value_counts_with_idx.index.name = "tissue_name"
    tissue_names_value_counts_with_idx = tissue_names_value_counts_with_idx.reset_index()
    tissue_names_value_counts_with_idx.reset_index(names="tissue_idx", inplace=True)
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


if __name__ == "__main__":
    app.run(main)
