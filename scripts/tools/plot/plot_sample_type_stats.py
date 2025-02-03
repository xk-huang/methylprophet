"""
python scripts/tools/plot/plot_sample_type_stats.py \
    --input_sample_tissue_count_csv data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_dir data/parquet/241231-tcga/metadata/sample_split/ \
    --output_plot_name sample_tissue_count-tcga \
    --overwrite
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from absl import app, flags, logging


flags.DEFINE_string(
    "input_sample_tissue_count_csv",
    "data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv",
    "Path to the sample tissue count csv",
)

flags.DEFINE_string("output_dir", "data/parquet/241231-tcga/metadata/sample_split/", "Output directory")
flags.DEFINE_string("output_plot_name", "sample_tissue_count", "Output plot name")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("rename_encode_tissue", False, "Rename the tissue names to match the ENCODE tissue names")

FLAGS = flags.FLAGS


sns.set_style("white")
sns.set_context("paper", font_scale=2)
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["axes.labelsize"] = 20


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_plot_path = output_dir / (FLAGS.output_plot_name + ".pdf")

    if output_plot_path.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_plot_path} already exists. Skipping...")
            return
        if FLAGS.overwrite:
            logging.warning(f"Overwriting {output_plot_path}")

    df = pd.read_csv(FLAGS.input_sample_tissue_count_csv)
    df = df[["tissue_name", "count"]]
    # get unique
    df = df.drop_duplicates()
    # Rename "NaN" Tissue to "Unknown"
    df["tissue_name"] = df["tissue_name"].fillna("Unknown")

    if FLAGS.rename_encode_tissue:
        df["tissue_name"] = df["tissue_name"].str.replace(" (37 years)", "")
        df["tissue_name"] = df["tissue_name"].str.replace(" (33 years)", "")
        df["tissue_name"] = df["tissue_name"].str.replace(" originated from", "")
        df["tissue_name"] = df["tissue_name"].str.replace(" female adult", "")
        df["tissue_name"] = df["tissue_name"].str.replace(" male adult", "")
        df["tissue_name"] = df["tissue_name"].str.replace("common myeloid progenitor,", "CPM")

    df.sort_values("tissue_name", ascending=True, inplace=True)
    df.sort_values("count", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"count": "# Samples", "tissue_name": "Tissue"}, inplace=True)

    # Plot the tissue_name value counts
    fig, ax = plt.subplots(figsize=(15, 12))  # Increase the figure size
    sns.barplot(
        df,
        x="# Samples",
        y="Tissue",
        ax=ax,
        # palette=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
        palette="Spectral",
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right"
    )  # Rotate labels by 45 degrees and align them to the right
    plt.tight_layout()  # Adjust the layout

    fig.savefig(output_plot_path)
    logging.info(f"Saved {output_plot_path}")


if __name__ == "__main__":
    app.run(main)
