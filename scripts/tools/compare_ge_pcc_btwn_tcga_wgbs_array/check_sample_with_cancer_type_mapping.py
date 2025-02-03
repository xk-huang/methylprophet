from pathlib import Path

import pandas as pd
from absl import app, flags, logging

flags.DEFINE_string(
    "cancer_type_csv", "data/extracted/241213-tcga_array/project.csv", "Path to the TCGA cancer type csv file"
)
flags.DEFINE_string("gene_csv", "data/extracted/241213-tcga_wgbs/ge.csv", "Path to the gene csv file")

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
    cancer_type_csv = Path(FLAGS.cancer_type_csv)
    gene_csv = Path(FLAGS.gene_csv)

    logging.info(f"Reading {cancer_type_csv}")
    cancer_type_df = pd.read_csv(cancer_type_csv)
    logging.info(f"Reading {gene_csv}")
    gene_df = pd.read_csv(gene_csv, nrows=None, index_col=0)

    array_subject_list = cancer_type_df["Sample"].str.split("-").str[2].to_list()
    wgbs_subject_list = gene_df.columns.str.split("_").str[2].to_list()
    joint_subject_list = set(array_subject_list) & set(wgbs_subject_list)

    if set(joint_subject_list) != set(TARGET_SUBJECT_NAMES):
        logging.warning(
            f"Mismatched subjects, A-B: {set(TARGET_SUBJECT_NAMES) - set(joint_subject_list)}, B-A: {set(joint_subject_list) - set(TARGET_SUBJECT_NAMES)}"
        )
        raise ValueError(f"Expected {TARGET_SUBJECT_NAMES}, got {joint_subject_list}")
    logging.info("Subjects matched.")

    sample_subject_df = pd.DataFrame(
        {
            "sample_name": gene_df.columns,
            "subject_name": gene_df.columns.str.split("_").str[2],
        }
    )
    joint_sample_subject_df = sample_subject_df[sample_subject_df["subject_name"].isin(joint_subject_list)]
    non_joint_sample_subject_df = sample_subject_df[~sample_subject_df["subject_name"].isin(joint_subject_list)]

    logging.info(f"Number of samples: {len(sample_subject_df)}")
    logging.info(f"Number of joint samples: {len(joint_sample_subject_df)}")
    logging.info(f"Number of non-joint samples: {len(non_joint_sample_subject_df)}")

    logging.warning(f"Non-joint samples: {non_joint_sample_subject_df}")
    logging.warning("We do not know the cancer type of the non-joint samples.")


if __name__ == "__main__":
    app.run(main)
