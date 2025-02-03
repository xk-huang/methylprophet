import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string(
    "tcga_array_me_csv", "data/extracted/241213-tcga_array/me_rownamesloc.csv", "Path to the TCGA array ME CSV file"
)

flags.DEFINE_string(
    "tcga_wgbs_me_csv", "data/extracted/241213-tcga_wgbs/me_rownamesloc.csv", "Path to the TCGA WGBS ME CSV file"
)
flags.DEFINE_string("output_dir", "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/sample_name", "Output directory")
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")


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
    output_dir = prepare_output_dir(FLAGS.output_dir, FLAGS.overwrite)

    logging.info(f"Reading TCGA array ME chr pos from {FLAGS.tcga_array_me_csv}")
    tcga_array_sample_names = read_sample_names_from_me(FLAGS.tcga_array_me_csv)
    logging.info(f"Reading TCGA WGBS ME chr pos from {FLAGS.tcga_wgbs_me_csv}")
    tcga_wgbs_sample_names = read_sample_names_from_me(FLAGS.tcga_wgbs_me_csv)

    tcga_array_subject_name = tcga_array_sample_names.str.split("-").str[2]
    tcga_wgbs_subject_name = tcga_wgbs_sample_names.str.split("_").str[2]

    joint_subject_names = set(tcga_array_subject_name).intersection(set(tcga_wgbs_subject_name))
    if len(joint_subject_names) != len(TARGET_SUBJECT_NAMES):
        set_a = set(TARGET_SUBJECT_NAMES)
        set_b = set(joint_subject_names)
        logging.warning(
            f"Joint sample names do not match target subject names: A-B {set_a - set_b}, B-A {set_b - set_a}"
        )
        raise ValueError("Joint subject names do not match target subject names")
    logging.info("Joint subject match target subject names")

    tcga_array_sample_subject_df = pd.DataFrame(
        {
            "sample_name": tcga_array_sample_names,
            "subject_name": tcga_array_subject_name,
        }
    )
    tcga_wgbs_sample_subject_df = pd.DataFrame(
        {
            "sample_name": tcga_wgbs_sample_names,
            "subject_name": tcga_wgbs_subject_name,
        }
    )

    joint_tcga_array_sample_subject_df = tcga_array_sample_subject_df[
        tcga_array_sample_subject_df["subject_name"].isin(joint_subject_names)
    ]
    joint_tcga_wgbs_sample_subject_df = tcga_wgbs_sample_subject_df[
        tcga_wgbs_sample_subject_df["subject_name"].isin(joint_subject_names)
    ]

    write_csv(joint_tcga_array_sample_subject_df, output_dir, "joint_tcga_array_sample_subject.csv")
    write_csv(joint_tcga_wgbs_sample_subject_df, output_dir, "joint_tcga_wgbs_sample_subject.csv")


def write_csv(df, output_dir, file_name):
    output_path = output_dir / file_name
    df.to_csv(output_path)
    logging.info(f"Saved {file_name} to {output_path}")


def read_sample_names_from_me(me_csv):
    df = pd.read_csv(me_csv, nrows=1, index_col=0)
    return df.columns


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
