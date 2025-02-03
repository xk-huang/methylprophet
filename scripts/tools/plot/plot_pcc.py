"""
python scripts/tools/plot/plot_pcc.py \
    --input_eval_pcc_dir outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/pcc_and_me_std \
    --output_dir outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/plots/pcc/ \
    --output_plot_name eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
from absl import app, flags, logging


plt.style.use(["science", "no-latex"])

flags.DEFINE_string(
    "input_eval_pcc_dir",
    "outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/pcc_and_me_std",
    "Path to the sample tissue count csv",
)

flags.DEFINE_string("output_dir", "data/parquet/241231-tcga/metadata/sample_split/", "Output directory")
flags.DEFINE_string("output_plot_name", "sample_tissue_count", "Output plot name")
flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("rename_encode_tissue", False, "Rename the tissue names to match the ENCODE tissue names")

FLAGS = flags.FLAGS


# sns.set_style("white")
# sns.set_context("paper", font_scale=2)
# plt.rcParams["ytick.labelsize"] = 15
# plt.rcParams["xtick.labelsize"] = 20
# plt.rcParams["axes.labelsize"] = 20

PCC_BY_CPG_ID_FILES = {
    "Train CpG\nVal Sample": "pcc_by_cpg_id_me_std-train_cpg-val_sample.parquet.csv",
    "Val CpG\nTrain Sample": "pcc_by_cpg_id_me_std-val_cpg-train_sample.parquet.csv",
    "Val CpG\nVal Sample": "pcc_by_cpg_id_me_std-val_cpg-val_sample.parquet.csv",
}

PCC_BY_SAMPLE_ID_FILES = {
    "Train CpG\nVal Sample": "pcc_by_sample_id-train_cpg-val_sample.parquet.csv",
    "Val CpG\nTrain Sample": "pcc_by_sample_id-val_cpg-train_sample.parquet.csv",
    "Val CpG\nVal Sample": "pcc_by_sample_id-val_cpg-val_sample.parquet.csv",
}


def read_pcc_files(pcc_by_cpg_id_dir, pcc_files, index_name):
    pcc_by_cpg_id_dir = Path(pcc_by_cpg_id_dir)
    pcc_by_cpg_id_dfs = {}
    for name, file in pcc_files.items():
        pcc_df = pd.read_csv(pcc_by_cpg_id_dir / file)
        pcc_df = pcc_df[[index_name, "pcc"]]

        pcc_df.set_index(index_name, inplace=True)
        pcc_df.rename(columns={"pcc": name}, inplace=True)
        pcc_by_cpg_id_dfs[name] = pcc_df
    return pd.concat(pcc_by_cpg_id_dfs.values(), axis=1)


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pcc_by_cpg_idx_plot_path = output_dir / f"pcc_by_cpg_id-{FLAGS.output_plot_name}.pdf"
    output_pcc_by_sample_idx_plot_path = output_dir / f"pcc_by_sample_id-{FLAGS.output_plot_name}.pdf"
    for output_plot_path in (output_pcc_by_cpg_idx_plot_path, output_pcc_by_sample_idx_plot_path):
        if output_plot_path.exists():
            if not FLAGS.overwrite:
                logging.warning(f"{output_plot_path} already exists. Skipping...")
                return
            if FLAGS.overwrite:
                logging.warning(f"Overwriting {output_plot_path}")

    pcc_by_cpg_idx_df = read_pcc_files(FLAGS.input_eval_pcc_dir, PCC_BY_CPG_ID_FILES, "cpg_idx")
    pcc_by_sample_idx_df = read_pcc_files(FLAGS.input_eval_pcc_dir, PCC_BY_SAMPLE_ID_FILES, "sample_idx")

    plot_violin(output_pcc_by_cpg_idx_plot_path, pcc_by_cpg_idx_df, ylim_bottom=-0.5, ylim_top=1)
    plot_violin(output_pcc_by_sample_idx_plot_path, pcc_by_sample_idx_df, ylim_bottom=0, ylim_top=1)


def plot_violin(output_plot_path, pcc_df, ylim_bottom=-1, ylim_top=1):
    fig, ax = plt.subplots()  # Increase the figure size
    sns.violinplot(pcc_df, ax=ax, palette="pastel")
    ax.set_ylim(ylim_bottom, ylim_top)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
    plt.tight_layout()  # Adjust the layout

    fig.savefig(output_plot_path)
    fig.savefig(output_plot_path.with_suffix(".png"))
    logging.info(f"Saved {output_plot_path}")
    logging.info(f"Saved {output_plot_path.with_suffix('.png')}")


if __name__ == "__main__":
    app.run(main)
