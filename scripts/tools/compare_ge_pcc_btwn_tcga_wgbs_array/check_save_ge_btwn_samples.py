"""
python scripts/tools/compare_ge_pcc_btwn_tcga_wgbs_array/check_save_ge_btwn_samples.py \
    --ge_csv_1 /insomnia001/depts/houlab/users/wh2526/methylformer/data/tcga/hg38/450k/ge.csv \
    --ge_csv_2 /insomnia001/depts/houlab/users/wh2526/methylformer/data/tcga/hg38/wgbs/ge.csv \
    --output_dir misc/241204-comp_ge_tcga_wgbs_array
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("ge_csv_1", None, "Path to the first CSV file")
flags.DEFINE_string("ge_csv_2", None, "Path to the second CSV file")
flags.DEFINE_string("output_dir", None, "Path to the output directory")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output directory")

flags.mark_flag_as_required("ge_csv_1")
flags.mark_flag_as_required("ge_csv_2")
flags.mark_flag_as_required("output_dir")

FLAGS = flags.FLAGS

TARGET_SUBJECT_NAMES = [
    "A13J",
    "A1AA",
    "A1AG",
    "A20V",
    "A2HQ",
    "A2LA",
    "A0CE",
    "A0YG",
    "6177",
    "6452",
    "6519",
    "A1CI",
    "A1CK",
]


def main(_):
    output_dir = Path(FLAGS.output_dir)
    if output_dir.exists():
        if FLAGS.overwrite:
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ge_df_1_path = Path(FLAGS.ge_csv_1)
    ge_df_1 = pd.read_csv(FLAGS.ge_csv_1, nrows=None, index_col=0)
    logging.info(f"Loaded GE CSV 1 from {ge_df_1_path}: {ge_df_1.shape}")

    ge_df_2_path = Path(FLAGS.ge_csv_2)
    ge_df_2 = pd.read_csv(FLAGS.ge_csv_2, nrows=None, index_col=0)
    logging.info(f"Loaded GE CSV 2 from {ge_df_2_path}: {ge_df_2.shape}")

    # Convert index from "TSPAN6;ENSG00000000003" to "TSPAN6"
    ge_df_1.index = ge_df_1.index.str.split(";").str[0]
    ge_df_2.index = ge_df_2.index.str.split(";").str[0]
    if ge_df_1.index.equals(ge_df_2.index):
        logging.info("Index matched for gene names")
    else:
        logging.warning(f"Index mismatch for gene names: {len(ge_df_1)}, {len(ge_df_2)}")
        mismatched_index_1_diff_2 = ge_df_1.index.difference(ge_df_2.index)
        mismatched_index_2_to_1 = ge_df_2.index.difference(ge_df_1.index)

        mismatched_index_1_diff_2_path = output_dir / "mismatched_gene_1_diff_2.csv"
        mismatched_index_2_diff_1_path = output_dir / "mismatched_gene_2_diff_1.csv"

        mismatched_index_df = pd.DataFrame(mismatched_index_1_diff_2, columns=["gene_names"])
        mismatched_index_df.to_csv(mismatched_index_1_diff_2_path, index=True)
        mismatched_index_df = pd.DataFrame(mismatched_index_2_to_1, columns=["gene_names"])
        mismatched_index_df.to_csv(mismatched_index_2_diff_1_path, index=True)

        joint_index = ge_df_1.index.intersection(ge_df_2.index)
        joint_index_df = pd.DataFrame(joint_index, columns=["gene_names"])
        joint_index_path = output_dir / "joint_gene_names.csv"
        joint_index_df.to_csv(joint_index_path, index=True)

    # Convert column namefrom "TCGA-BK-A0CA-01" to "A0CA" TCGA Array
    ge_df_1_columns = ge_df_1.columns.str.split("-").str[2]
    # Convert column namefrom "TCGA_BLCA_A13J" to "A13J" in TCGA WGBS
    ge_df_2_columns = ge_df_2.columns.str.split("_").str[2]

    # Check joint columns
    num_samples_df_1 = len(ge_df_1_columns)
    num_unique_samples_df_1 = len(ge_df_1_columns.unique())
    logging.info(f"Number of samples vs unique samples in GE CSV 1: {num_samples_df_1} vs. {num_unique_samples_df_1}")
    num_samples_df_2 = len(ge_df_2_columns)
    num_unique_samples_df_2 = len(ge_df_2_columns.unique())
    logging.info(f"Number of samples vs unique samples in GE CSV 2: {num_samples_df_2} vs. {num_unique_samples_df_2}")

    target_subject_names_set_ = set(ge_df_1_columns) & set(ge_df_2_columns)
    if target_subject_names_set_ != set(TARGET_SUBJECT_NAMES):
        logging.warning(
            f"diff A-B: {set(TARGET_SUBJECT_NAMES) - target_subject_names_set_} and B-A: {target_subject_names_set_ - set(TARGET_SUBJECT_NAMES)}"
        )
        raise ValueError("Target subject names mismatch from pre-defined list")
    logging.info("Target subject names matched with online computation")

    target_subject_names_set = set(TARGET_SUBJECT_NAMES)
    if (target_subject_names_set & set(ge_df_1_columns)) == (target_subject_names_set):
        logging.info("GE CSV 1 columns matched")
    else:
        raise ValueError("GE CSV 1 columns mismatch")
    if (target_subject_names_set & set(ge_df_2_columns)) == (target_subject_names_set):
        logging.info("GE CSV 2 columns matched")
    else:
        raise ValueError("GE CSV 2 columns mismatch")

    # ge_df_1_columns.value_counts().loc[TARGET_SUBJECT_NAMES]
    # ge_df_2_columns.value_counts().loc[TARGET_SUBJECT_NAMES]

    ge_df_1_columns = pd.DataFrame(
        {
            "sample_name": ge_df_1.columns,
            "subject_name": ge_df_1.columns.str.split("-").str[2],
            "type_name": ge_df_1.columns.str.split("-").str[1],
        }
    )
    ge_df_2_columns = pd.DataFrame(
        {
            "sample_name": ge_df_2.columns,
            "subject_name": ge_df_2.columns.str.split("_").str[2],
            "type_name": ge_df_2.columns.str.split("_").str[1],
        }
    )
    ge_df_1_columns_selected = ge_df_1_columns[ge_df_1_columns["subject_name"].isin(TARGET_SUBJECT_NAMES)]
    ge_df_2_columns_selected = ge_df_2_columns[ge_df_2_columns["subject_name"].isin(TARGET_SUBJECT_NAMES)]

    ge_df_1_columns_selected = ge_df_1_columns_selected.sort_values("sample_name").sort_values("subject_name")
    ge_df_2_columns_selected = ge_df_2_columns_selected.sort_values("sample_name").sort_values("subject_name")

    ge_df_1_columns_selected_path = output_dir / "ge_df_1_columns_selected.csv"
    ge_df_1_columns_selected.to_csv(ge_df_1_columns_selected_path, index=True)
    ge_df_2_columns_selected_path = output_dir / "ge_df_2_columns_selected.csv"
    ge_df_2_columns_selected.to_csv(ge_df_2_columns_selected_path, index=True)
    logging.info(f"Saved selected columns to: {ge_df_1_columns_selected_path}")
    logging.info(f"Saved selected columns to: {ge_df_2_columns_selected_path}")

    ge_df_1_selected = ge_df_1.loc[:, ge_df_1_columns_selected["sample_name"]]
    ge_df_2_selected = ge_df_2.loc[:, ge_df_2_columns_selected["sample_name"]]

    ge_df_1_selected_path = output_dir / "ge_df_1_selected.csv"
    ge_df_1_selected.to_csv(ge_df_1_selected_path, index=True)
    ge_df_2_selected_path = output_dir / "ge_df_2_selected.csv"
    ge_df_2_selected.to_csv(ge_df_2_selected_path, index=True)


if __name__ == "__main__":
    app.run(main)
