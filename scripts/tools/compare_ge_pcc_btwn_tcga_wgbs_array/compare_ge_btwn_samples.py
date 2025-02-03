"""
This script compares two Gene Expression (GE) CSV files and calculates the Pearson Correlation Coefficient (PCC) between them.

python scripts/tools/compare_ge_pcc_btwn_tcga_wgbs_array/compare_ge_btwn_samples.py \
    --ge_csv_1 misc/241204-comp_ge_tcga_wgbs_array/ge_df_1_selected.csv \
    --ge_csv_2 misc/241204-comp_ge_tcga_wgbs_array/ge_df_2_selected.csv \
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


def main(_):
    ge_csv_1 = Path(FLAGS.ge_csv_1)
    ge_csv_2 = Path(FLAGS.ge_csv_2)

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=FLAGS.overwrite)

    logging.info(f"Reading {ge_csv_1}")
    ge_df_1 = pd.read_csv(ge_csv_1, index_col=0)
    logging.info(f"Reading {ge_csv_2}")
    ge_df_2 = pd.read_csv(ge_csv_2, index_col=0)

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

    pcc_dict_list = []
    joint_genes = ge_df_1.index.intersection(ge_df_2.index)

    for df_1_row in ge_df_1_columns.itertuples():
        for df_2_row in ge_df_2_columns.itertuples():
            if df_1_row.subject_name == df_2_row.subject_name:
                index_df_1 = df_1_row.Index
                index_df_2 = df_2_row.Index
                sample_name_df_1 = df_1_row.sample_name
                sample_name_df_2 = df_2_row.sample_name

                gene_df_1_sample = ge_df_1.loc[joint_genes, sample_name_df_1]
                gene_df_2_sample = ge_df_2.loc[joint_genes, sample_name_df_2]
                gene_df_1_sample = gene_df_1_sample.sort_index()
                gene_df_2_sample = gene_df_2_sample.sort_index()

                # compute pcc
                pcc = gene_df_1_sample.corr(gene_df_2_sample)
                pcc_dict = {
                    "sample_name_df_1": sample_name_df_1,
                    "sample_name_df_2": sample_name_df_2,
                    "pcc": pcc,
                }
                pcc_dict_list.append(pcc_dict)

                logging.info(
                    f"index_df_1: {index_df_1}, index_df_2: {index_df_2}, "
                    f"sample_name_df_1: {sample_name_df_1}, "
                    f"sample_name_df_2: {sample_name_df_2}, "
                    f"pcc: {pcc}"
                )
    pcc_df = pd.DataFrame(pcc_dict_list)
    pcc_df_path = output_dir / "pcc_df.csv"
    pcc_df.to_csv(pcc_df_path)


if __name__ == "__main__":
    app.run(main)
