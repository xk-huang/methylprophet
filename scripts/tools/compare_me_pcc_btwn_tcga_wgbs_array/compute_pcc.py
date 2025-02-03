import datetime
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string(
    "tcga_array_me_csv",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/me/joint_tcga_array_me.csv",
    "Path to the TCGA array ME CSV file",
)

flags.DEFINE_string(
    "tcga_wgbs_me_csv",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/me/joint_tcga_wgbs_me.csv",
    "Path to the TCGA WGBS ME CSV file",
)


flags.DEFINE_string(
    "tcga_array_joint_sample_name_csv",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/sample_name/joint_tcga_array_sample_subject.csv",
    "Path to the TCGA array joint sample names CSV file",
)
flags.DEFINE_string(
    "tcga_wgbs_joint_sample_name_csv",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/sample_name/joint_tcga_wgbs_sample_subject.csv",
    "Path to the TCGA WGBS joint sample names CSV file",
)


flags.DEFINE_string(
    "output_dir", "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/me_pcc", "Path to the output directory"
)
flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output directory")

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
    output_dir = prepare_output_dir(FLAGS.output_dir, overwrite=FLAGS.overwrite)

    tcga_array_me = pd.read_csv(FLAGS.tcga_array_me_csv, index_col=0)
    tcga_wgbs_me = pd.read_csv(FLAGS.tcga_wgbs_me_csv, index_col=0)

    # Check cpg chr pos
    if not tcga_array_me.index.sort_values().equals(tcga_wgbs_me.index.sort_values()):
        raise ValueError("CpG chr pos between two ME dataframes are not the same")

    # Check subject names
    tcga_array_sample_name = tcga_array_me.columns
    tcga_wgbs_sample_name = tcga_wgbs_me.columns

    tcga_array_subject_name = tcga_array_sample_name.str.split("-").str[2]
    tcga_wgbs_subject_name = tcga_wgbs_sample_name.str.split("_").str[2]

    if not set(tcga_array_subject_name) == set(tcga_wgbs_subject_name):
        raise ValueError("Subject names between two ME dataframes are not the same")
    if not set(tcga_array_subject_name) == set(TARGET_SUBJECT_NAMES):
        raise ValueError("Subject names do not match target subject names")

    tcga_array_sample_subject = pd.DataFrame(
        {
            "sample_name": tcga_array_sample_name,
            "subject_name": tcga_array_subject_name,
        }
    )
    tcga_wgbs_sample_subject = pd.DataFrame(
        {
            "sample_name": tcga_wgbs_sample_name,
            "subject_name": tcga_wgbs_subject_name,
        }
    )

    pcc_compare_dict_list = []
    for subject_name in TARGET_SUBJECT_NAMES:
        tcga_array_me_by_subject = tcga_array_me.loc[:, tcga_array_subject_name == subject_name]
        tcga_wgbs_me_by_subject = tcga_wgbs_me.loc[:, tcga_wgbs_subject_name == subject_name]
        for tcga_array_sample_name in tcga_array_me_by_subject.columns:
            for tcga_wgbs_sample_name in tcga_wgbs_me_by_subject.columns:
                tcga_array_me_by_sample = tcga_array_me_by_subject.loc[:, tcga_array_sample_name]
                tcga_wgbs_me_by_sample = tcga_wgbs_me_by_subject.loc[:, tcga_wgbs_sample_name]

                pcc = tcga_array_me_by_sample.corr(tcga_wgbs_me_by_sample, method="pearson")
                mae = np.abs(tcga_array_me_by_sample - tcga_wgbs_me_by_sample).mean()
                mse = ((tcga_array_me_by_sample - tcga_wgbs_me_by_sample) ** 2).mean()
                pcc_compare_dict = {
                    "subject_name": subject_name,
                    "tcga_array_sample_name": tcga_array_sample_name,
                    "tcga_wgbs_sample_name": tcga_wgbs_sample_name,
                    "pcc": pcc,
                    "mae": mae,
                    "mse": mse,
                }

                pcc_compare_dict_list.append(pcc_compare_dict)

    pcc_compare_df = pd.DataFrame(pcc_compare_dict_list)

    pcc_compare_df_path = output_dir / "pcc_compare.csv"
    pcc_compare_df.to_csv(pcc_compare_df_path, index=True)
    logging.info(f"Saved PCC compare to {pcc_compare_df_path}")


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    app.run(main)
