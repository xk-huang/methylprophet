from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


# "data/stats_sample_name-encode_wgbs/sample_tissue_count_with_idx.csv"
flags.DEFINE_string("input_sample_tissue_count_with_idx_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_sample_tissue_count_with_idx_file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
# "ind_tissue"
flags.DEFINE_string("output_sample_split_type", None, "Output directory")
flags.mark_flag_as_required("output_sample_split_type")

flags.DEFINE_float("val_split_ratio", 0.1, "Val split ratio")

flags.DEFINE_integer("seed", 42, "Random seed for split")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_list("exclude_sample_name_files", None, "List of sample names to exclude")

FLAGS = flags.FLAGS


def main(_):
    input_sample_tissue_count_with_idx_file = Path(FLAGS.input_sample_tissue_count_with_idx_file)

    output_dir = Path(FLAGS.output_dir)
    output_sample_split_type = FLAGS.output_sample_split_type
    output_dir = output_dir / output_sample_split_type

    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_dir} already exists. Skipping...")
            return
        else:
            logging.warning(f"Overwriting {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Reading {input_sample_tissue_count_with_idx_file}")
    sample_tissue_count_with_idx_df = pd.read_csv(input_sample_tissue_count_with_idx_file, index_col=0)

    # filter by exclude_sample_name_files
    exclude_sample_name_files = FLAGS.exclude_sample_name_files
    sample_tissue_count_with_idx_df = filter_by_exclude_sample_name_files(
        sample_tissue_count_with_idx_df, exclude_sample_name_files
    )

    seed = FLAGS.seed
    rng = np.random.default_rng(seed)

    # Create train and val split
    val_split_ratio = FLAGS.val_split_ratio
    logging.info(f"Creating {output_sample_split_type} split, val split ratio: {val_split_ratio}")
    if output_sample_split_type == "ind_cancer":
        train_sample_tissue_count_with_idx_df, val_sample_tissue_count_with_idx_df = create_ind_cancer_split(
            sample_tissue_count_with_idx_df, val_split_ratio, rng
        )
    else:
        raise ValueError(f"Unknown sample split type: {output_sample_split_type}")

    # Check if there is any overlap between train and val
    logging.info("Checking overlap between train and val")

    if not (train_sample_tissue_count_with_idx_df.index == train_sample_tissue_count_with_idx_df["sample_idx"]).all():
        raise ValueError("Train sample_idx is not equal to index")

    # NOTE xk: No need to check this since the sample idx is merged from all tissues, so it's not necessary to be continuous
    # if not (
    #     val_sample_tissue_count_with_idx_df.index
    #     == (
    #         val_sample_tissue_count_with_idx_df["sample_idx"] - val_sample_tissue_count_with_idx_df["sample_idx"].min()
    #     )
    # ).all():
    #     raise ValueError("Val sample_idx is not equal to index")

    index_overlap = train_sample_tissue_count_with_idx_df.index.intersection(val_sample_tissue_count_with_idx_df.index)
    non_index_overlap = len(index_overlap) == 0
    if not non_index_overlap:
        raise ValueError("Overlap between train and val")

    # Check if the union of train and val is equal to the original samples
    logging.info("Checking union of train and val is equal to the original samples")
    index_union = train_sample_tissue_count_with_idx_df.index.union(val_sample_tissue_count_with_idx_df.index)
    is_index_union_equal = index_union.sort_values().equals(sample_tissue_count_with_idx_df.index)
    if not is_index_union_equal:
        raise ValueError("Union of train and val is not equal to the original samples")

    logging.info(
        f"Train samples: {len(train_sample_tissue_count_with_idx_df)}, num of tissues: {len(train_sample_tissue_count_with_idx_df['tissue_idx'].unique())}"
    )
    logging.info(
        f"Val samples: {len(val_sample_tissue_count_with_idx_df)}, num of tissues: {len(val_sample_tissue_count_with_idx_df['tissue_idx'].unique())}"
    )

    train_sample_tissue_count_with_idx_df.to_csv(output_dir / "train_sample_tissue_count_with_idx.csv")
    val_sample_tissue_count_with_idx_df.to_csv(output_dir / "val_sample_tissue_count_with_idx.csv")


def create_ind_cancer_split(sample_tissue_count_with_idx_df, val_split_ratio, rng):
    val_split_ratio = FLAGS.val_split_ratio
    if val_split_ratio < 0 or val_split_ratio >= 1:
        raise ValueError("Val split ratio should be in the range [0, 1)")
    train_split_ratio = 1 - val_split_ratio
    logging.info(f"Train split ratio: {train_split_ratio}, Val split ratio: {val_split_ratio}")

    train_sample_tissue_count_with_idx_df_ls = []
    val_sample_tissue_count_with_idx_df_ls = []
    for group_idx, group_df in sample_tissue_count_with_idx_df.groupby("tissue_idx"):
        # print(group_idx, len(group_df))
        if len(group_df) == 1:
            logging.warning(f"group idx: {group_idx}, num of samples: {len(group_df)}, train only")
            train_sample_tissue_count_with_idx_df_ls.append(group_df)
        else:
            # print(f"group idx: {group_idx}, num of samples: {len(group_df)}, train/val")
            train_sample_tissue_count_with_idx_df = group_df.sample(frac=train_split_ratio, random_state=rng)
            val_sample_tissue_count_with_idx_df = group_df.drop(train_sample_tissue_count_with_idx_df.index)
            if len(val_sample_tissue_count_with_idx_df) == 0:
                logging.warning(f"No validation samples for group idx: {group_idx}")

            train_sample_tissue_count_with_idx_df_ls.append(train_sample_tissue_count_with_idx_df)
            val_sample_tissue_count_with_idx_df_ls.append(val_sample_tissue_count_with_idx_df)

    train_sample_tissue_count_with_idx_df = pd.concat(train_sample_tissue_count_with_idx_df_ls)
    val_sample_tissue_count_with_idx_df = pd.concat(val_sample_tissue_count_with_idx_df_ls)
    return train_sample_tissue_count_with_idx_df, val_sample_tissue_count_with_idx_df


def filter_by_exclude_sample_name_files(sample_tissue_count_with_idx_df, exclude_sample_name_files):
    if exclude_sample_name_files is None:
        return sample_tissue_count_with_idx_df

    exclude_sample_names = []
    for exclude_sample_name_file in exclude_sample_name_files:
        with open(exclude_sample_name_file, "r") as f:
            exclude_sample_names.extend(f.read().splitlines())

    logging.info(f"Excluding {len(exclude_sample_names)} samples: {exclude_sample_names}")
    exclude_condition = sample_tissue_count_with_idx_df["sample_name"].isin(exclude_sample_names)
    sample_names_to_be_excluded = sample_tissue_count_with_idx_df["sample_name"][exclude_condition]
    if len(exclude_sample_names) != len(set(sample_names_to_be_excluded)):
        # find the missing samples
        missing_samples = set(exclude_sample_names) - set(sample_names_to_be_excluded)
        raise ValueError(f"Some samples are not found in the sample_tissue_count_with_idx_df: {missing_samples}")

    filtered_sample_tissue_count_with_idx_df = sample_tissue_count_with_idx_df[~exclude_condition]

    # double check
    if filtered_sample_tissue_count_with_idx_df["sample_name"].isin(exclude_sample_names).any():
        raise ValueError("Failed to exclude samples, which should not happened.")

    return sample_tissue_count_with_idx_df[~exclude_condition]


if __name__ == "__main__":
    app.run(main)
