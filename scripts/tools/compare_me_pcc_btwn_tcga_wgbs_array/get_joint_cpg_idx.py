import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string(
    "tcga_array_cpg_chr_pos_path",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/tcga_array_cpg_chr_pos.csv",
    "Path to TCGA array ME chr pos file",
)
flags.DEFINE_string(
    "tcga_wgbs_cpg_chr_pos_path",
    "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/tcga_wgbs_cpg_chr_pos.csv",
    "Path to TCGA WGBS ME chr pos file",
)

flags.DEFINE_string("output_dir", "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/cpg_chr_pos", "Output directory")
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")


FLAGS = flags.FLAGS


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, FLAGS.overwrite)
    array_cpg_chr_pos_df = pd.read_csv(FLAGS.tcga_array_cpg_chr_pos_path)
    wgbs_cpg_chr_pos_df = pd.read_csv(FLAGS.tcga_wgbs_cpg_chr_pos_path)

    array_cpg_chr_pos_df.rename(columns={"Unnamed: 0": "chr_pos"}, inplace=True)
    wgbs_cpg_chr_pos_df.rename(columns={"Unnamed: 0": "chr_pos"}, inplace=True)

    joint_chr_pos = set(array_cpg_chr_pos_df["chr_pos"]).intersection(set(wgbs_cpg_chr_pos_df["chr_pos"]))
    logging.info(f"Number of joint chr pos: {len(joint_chr_pos)}")
    logging.info(f"Number of array chr pos: {len(array_cpg_chr_pos_df)}")
    logging.info(f"Number of WGBS chr pos: {len(wgbs_cpg_chr_pos_df)}")

    joint_array_cpg_chr_pos_df = array_cpg_chr_pos_df[array_cpg_chr_pos_df["chr_pos"].isin(joint_chr_pos)]
    non_joint_array_cpg_chr_pos_df = array_cpg_chr_pos_df[~array_cpg_chr_pos_df["chr_pos"].isin(joint_chr_pos)]
    joint_wgbs_cpg_chr_pos_df = wgbs_cpg_chr_pos_df[wgbs_cpg_chr_pos_df["chr_pos"].isin(joint_chr_pos)]
    non_joint_array_cpg_chr_pos_df = wgbs_cpg_chr_pos_df[~wgbs_cpg_chr_pos_df["chr_pos"].isin(joint_chr_pos)]

    write_to_csv(joint_array_cpg_chr_pos_df, output_dir, "joint_array_cpg_chr_pos.csv")
    write_to_csv(non_joint_array_cpg_chr_pos_df, output_dir, "non_joint_array_cpg_chr_pos.csv")
    write_to_csv(joint_wgbs_cpg_chr_pos_df, output_dir, "joint_wgbs_cpg_chr_pos.csv")
    write_to_csv(non_joint_array_cpg_chr_pos_df, output_dir, "non_joint_wgbs_cpg_chr_pos.csv")


def write_to_csv(df, output_dir, file_name):
    output_path = output_dir / file_name
    df.to_csv(output_path)
    logging.info(f"Saved {file_name} to {output_path}")


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
