import json
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags


flags.DEFINE_string(
    "input_sample_tissue_count_with_idx_file",
    "data/stats_sample_name-encode_wgbs/sample_tissue_count_with_idx.csv",
    "Path to the parquet file",
)
flags.DEFINE_alias("i", "input_sample_tissue_count_with_idx_file")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("sample_split_type", "val_ind_tissue", "Output directory")
flags.DEFINE_integer("num_val_ood_tissues", 10, "Number of tissues for val_ood_tissue split")
flags.DEFINE_integer("seed", 42, "Random seed for split")
flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")

FLAGS = flags.FLAGS


def main(_):
    input_sample_tissue_count_with_idx_file = Path(FLAGS.input_sample_tissue_count_with_idx_file)
    sample_tissue_count_with_idx_df = pd.read_csv(input_sample_tissue_count_with_idx_file, index_col=0)
    print(sample_tissue_count_with_idx_df)

    seed = FLAGS.seed
    rng = np.random.default_rng(seed)

    # Create train and val split
    sample_split_type = FLAGS.sample_split_type
    if sample_split_type == "val_ind_tissue":
        train_sample_tissue_count_with_idx_df, val_sample_tissue_count_with_idx_df = create_val_ind_tissue_split(
            sample_tissue_count_with_idx_df, rng
        )
    elif sample_split_type == "val_ood_tissue":
        num_val_ood_tissues = FLAGS.num_val_ood_tissues
        train_sample_tissue_count_with_idx_df, val_sample_tissue_count_with_idx_df = create_val_ood_tissue_split(
            sample_tissue_count_with_idx_df, rng, num_val_ood_tissues
        )
    else:
        raise ValueError(f"Unknown sample split type: {sample_split_type}")

    # Check if there is any overlap between train and val
    if not (train_sample_tissue_count_with_idx_df.index == train_sample_tissue_count_with_idx_df["sample_idx"]).all():
        raise ValueError("Train sample_idx is not equal to index")
    if not (val_sample_tissue_count_with_idx_df.index == val_sample_tissue_count_with_idx_df["sample_idx"]).all():
        raise ValueError("Val sample_idx is not equal to index")
    index_overlap = train_sample_tissue_count_with_idx_df.index.intersection(val_sample_tissue_count_with_idx_df.index)
    non_index_overlap = len(index_overlap) == 0
    if not non_index_overlap:
        raise ValueError("Overlap between train and val")

    # Check if the union of train and val is equal to the original samples
    index_union = train_sample_tissue_count_with_idx_df.index.union(val_sample_tissue_count_with_idx_df.index)
    is_index_union_equal = index_union.equals(sample_tissue_count_with_idx_df.index)
    if not is_index_union_equal:
        raise ValueError("Union of train and val is not equal to the original samples")

    print(
        f"Train samples: {len(train_sample_tissue_count_with_idx_df)}, num of tissues: {len(train_sample_tissue_count_with_idx_df['tissue_idx'].unique())}"
    )
    print(
        f"Val samples: {len(val_sample_tissue_count_with_idx_df)}, num of tissues: {len(val_sample_tissue_count_with_idx_df['tissue_idx'].unique())}"
    )

    output_dir = FLAGS.output_dir
    if output_dir is None:
        output_dir = input_sample_tissue_count_with_idx_file.parent
    else:
        output_dir = Path(output_dir) / sample_split_type
        output_dir.mkdir(exist_ok=True, parents=True)

    train_sample_tissue_count_with_idx_df.to_csv(output_dir / "train_sample_tissue_count_with_idx.csv")
    val_sample_tissue_count_with_idx_df.to_csv(output_dir / "val_sample_tissue_count_with_idx.csv")

    if FLAGS.ipython:
        from IPython import embed

        embed()


def create_val_ind_tissue_split(sample_tissue_count_with_idx_df, rng):
    train_sample_tissue_count_with_idx_df_ls = []
    val_sample_tissue_count_with_idx_df_ls = []
    for group_idx, group_df in sample_tissue_count_with_idx_df.groupby("tissue_idx"):
        # print(group_idx, len(group_df))
        if len(group_df) == 1:
            # print(f"group idx: {group_idx}, num of samples: {len(group_df)}, train")
            train_sample_tissue_count_with_idx_df_ls.append(group_df)
        else:
            # print(f"group idx: {group_idx}, num of samples: {len(group_df)}, train/val")
            train_sample_tissue_count_with_idx_df = group_df.sample(frac=0.5, random_state=rng)
            val_sample_tissue_count_with_idx_df = group_df.drop(train_sample_tissue_count_with_idx_df.index)
            if len(val_sample_tissue_count_with_idx_df) == 0:
                raise ValueError("No validation samples")

            train_sample_tissue_count_with_idx_df_ls.append(train_sample_tissue_count_with_idx_df)
            val_sample_tissue_count_with_idx_df_ls.append(val_sample_tissue_count_with_idx_df)

    train_sample_tissue_count_with_idx_df = pd.concat(train_sample_tissue_count_with_idx_df_ls)
    val_sample_tissue_count_with_idx_df = pd.concat(val_sample_tissue_count_with_idx_df_ls)
    return train_sample_tissue_count_with_idx_df, val_sample_tissue_count_with_idx_df


def create_val_ood_tissue_split(sample_tissue_count_with_idx_df, rng, num_val_ood_tissues=10):
    """
    For those tissue types having # Sample == 1, we attribute num_val_ood_tissues of them to the validation fold. While the rest sample types are used for training
    """
    tissue_idx = sample_tissue_count_with_idx_df["tissue_idx"].copy()
    tissue_idx_with_one_sample = tissue_idx[sample_tissue_count_with_idx_df["count"] == 1]

    tissue_idx = pd.Series(tissue_idx.unique())
    tissue_idx_with_one_sample = pd.Series(tissue_idx_with_one_sample.unique())

    val_tissue_idx = rng.choice(tissue_idx_with_one_sample, num_val_ood_tissues, replace=False)
    train_tissue_idx = tissue_idx[~tissue_idx.isin(val_tissue_idx)]

    # Check no overlap between val and train tissue types
    if len(set(val_tissue_idx).intersection(set(train_tissue_idx))) > 0:
        raise ValueError("Overlap between val and train tissue types")
    # Check all tissue types are used
    if len(set(val_tissue_idx).union(set(train_tissue_idx))) != len(tissue_idx):
        raise ValueError("Not all tissue types")

    train_sample_tissue_count_with_idx_df_ls = []
    val_sample_tissue_count_with_idx_df_ls = []
    for group_idx, group_df in sample_tissue_count_with_idx_df.groupby("tissue_idx"):
        if group_df["tissue_idx"].iloc[0] != group_idx:
            raise ValueError("Group idx is not equal to tissue_idx")

        if group_idx in train_tissue_idx:
            train_sample_tissue_count_with_idx_df_ls.append(group_df)
        elif group_idx in val_tissue_idx:
            # print(f"Val tissue idx: {group_idx}, num of samples: {len(group_df)}")
            val_sample_tissue_count_with_idx_df_ls.append(group_df)
        else:
            raise ValueError("Unknown group idx")

    train_sample_tissue_count_with_idx_df = pd.concat(train_sample_tissue_count_with_idx_df_ls)
    val_sample_tissue_count_with_idx_df = pd.concat(val_sample_tissue_count_with_idx_df_ls)
    return train_sample_tissue_count_with_idx_df, val_sample_tissue_count_with_idx_df


if __name__ == "__main__":
    app.run(main)
