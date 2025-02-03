import datetime
import shutil
import time
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

flags.DEFINE_string(
    "tcga_array_joint_cpg_chr_pos_csv",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/cpg_chr_pos/joint_array_cpg_chr_pos.csv",
    "Path to the TCGA array joint CpG chr pos CSV file",
)
flags.DEFINE_string(
    "tcga_wgbs_joint_cpg_chr_pos_csv",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/cpg_chr_pos/joint_wgbs_cpg_chr_pos.csv",
    "Path to the TCGA WGBS joint CpG chr pos CSV file",
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


flags.DEFINE_string("output_dir", "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/me", "Path to the output directory")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output directory")


FLAGS = flags.FLAGS


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, overwrite=FLAGS.overwrite)

    tcga_array_row_index = get_csv_index(FLAGS.tcga_array_joint_cpg_chr_pos_csv)
    tcga_array_col_index = get_csv_index(FLAGS.tcga_array_joint_sample_name_csv)
    joint_tcga_array_me = read_me(FLAGS.tcga_array_me_csv, tcga_array_row_index, tcga_array_col_index)

    tcga_wgbs_row_index = get_csv_index(FLAGS.tcga_wgbs_joint_cpg_chr_pos_csv)
    tcga_wgbs_col_index = get_csv_index(FLAGS.tcga_wgbs_joint_sample_name_csv)
    joint_tcga_wgbs_me = read_me(FLAGS.tcga_wgbs_me_csv, tcga_wgbs_row_index, tcga_wgbs_col_index)

    write_csv(joint_tcga_array_me, output_dir, "joint_tcga_array_me.csv")
    write_csv(joint_tcga_wgbs_me, output_dir, "joint_tcga_wgbs_me.csv")


def write_csv(df, output_dir, filename):
    output_path = output_dir / filename
    df.to_csv(output_path)
    logging.info(f"Saved to {output_path}")


def get_csv_index(index_csv):
    df = pd.read_csv(index_csv, index_col=0)
    return df.index


def read_me(me_csv, row_index, col_index):
    df = pd.read_csv(me_csv, index_col=0)
    return df.iloc[row_index, col_index]


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
