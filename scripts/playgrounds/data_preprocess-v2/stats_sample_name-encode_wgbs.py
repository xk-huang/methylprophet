import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from absl import app, flags

flags.DEFINE_string(
    "input_parquet_file", "data/parquet/encode_wgbs-240802/me.parquet/00000.parquet", "Path to the parquet file"
)
flags.DEFINE_alias("i", "input_parquet_file")
flags.DEFINE_string("output_dir", "data/stats_sample_name-encode_wgbs", "Output directory")
flags.DEFINE_alias("o", "output_dir")
FLAGS = flags.FLAGS
flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")


def main(_):
    input_parquet_file = Path(FLAGS.input_parquet_file)
    df = pd.read_parquet(input_parquet_file)
    print(df)

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if "sample_name" in df.columns:
        print(f"Number of sample_name: {df['sample_name'].unique().shape[0]}")
        print(df["sample_name"].unique())

    if "cpg_chr_pos" in df.columns:
        print(f"Number of cpg_chr_pos: {df['cpg_chr_pos'].unique().shape[0]}")
        print(df["cpg_chr_pos"].unique())

    columns = df.columns[1:]  # Remove the first column, "Unnamed: 0", which is the cpg_chr_pos
    # NOTE: Sort the sample names
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
    sample_tissue_with_sample_idx_df.to_csv(output_dir / "sample_with_idx.csv")

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

    sample_tissue_count_with_idx_df.to_csv(output_dir / "sample_tissue_count_with_idx.csv")
    sample_tissue_count_with_idx_df.sort_values(by="count").to_csv(
        output_dir / "sample_tissue_count_with_idx-sorted_by_count.csv"
    )
    with open(output_dir / "tissue_names.json", "w") as f:
        json.dump(sample_tissue_count_with_idx_df["tissue_name"].unique().tolist(), f, indent=4)

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

    # Validate sample_name
    sample_tissue_with_sample_idx_df = pd.read_csv(output_dir / "sample_with_idx.csv", index_col=0)
    sample_tissue_count_with_idx_df = pd.read_csv(output_dir / "sample_tissue_count_with_idx.csv", index_col=0)
    if not sample_tissue_with_sample_idx_df["sample_name"].equals(sample_tissue_count_with_idx_df["sample_name"]):
        raise ValueError("sample_name is not equal between original and processed dataframes")
    if not sample_tissue_with_sample_idx_df["tissue_name"].equals(sample_tissue_count_with_idx_df["tissue_name"]):
        raise ValueError("tissue_name is not equal between original and processed dataframes")
    if not sample_tissue_with_sample_idx_df.index.equals(sample_tissue_count_with_idx_df.index):
        raise ValueError("index is not equal between original and processed dataframes")

    if FLAGS.ipython:
        from IPython import embed

        embed()


if __name__ == "__main__":
    app.run(main)
