"""
python scripts/tools/data_preprocessing/merge_cpg_and_stats.py \
    --cpg_chr_pos_files data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_array/cpg_chr_pos_df.parquet,data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_epic/cpg_chr_pos_df.parquet,data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_wgbs/cpg_chr_pos_df.parquet \
    --output_dir data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix \
    --overwrite
"""

import json
import pprint
import shutil
from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_list("cpg_chr_pos_files", None, "List of CpG chr pos files")
flags.mark_flag_as_required("cpg_chr_pos_files")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_bool("overwrite", False, "Overwrite output directory")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS


def prepare_output_dir(output_dir: str, overwrite: bool):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Overwriting output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory already exists: {output_dir}")
            exit()

    logging.info(f"Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, FLAGS.overwrite)

    cpg_chr_pos_files = FLAGS.cpg_chr_pos_files
    cpg_chr_pos_source_to_idx_mapping = {
        cpg_chr_pos_file: idx for idx, cpg_chr_pos_file in enumerate(cpg_chr_pos_files)
    }
    cpg_chr_pos_df_list = []
    for cpg_chr_pos_file in cpg_chr_pos_files:
        cpg_chr_pos_df = pd.read_parquet(cpg_chr_pos_file)
        cpg_chr_pos_df["group_idx"] = cpg_chr_pos_source_to_idx_mapping[cpg_chr_pos_file]
        cpg_chr_pos_df_list.append(cpg_chr_pos_df)
    cpg_chr_pos_df = pd.concat(cpg_chr_pos_df_list, ignore_index=True)
    num_cpg_before_unique = len(cpg_chr_pos_df)

    duplicated_cpg_chr_pos_cond = cpg_chr_pos_df.duplicated(subset=["chr_pos"], keep=False)
    duplicated_cpg_chr_pos_df = cpg_chr_pos_df[duplicated_cpg_chr_pos_cond]

    # Group by chr_pos and get the combinations
    duplicated_combinations = (
        duplicated_cpg_chr_pos_df.groupby("chr_pos")["group_idx"].agg(list).apply(get_combination_type)
    )
    duplicated_combinations_file = output_dir / "duplicated_combinations.csv"
    logging.info(f"Writing output file: {duplicated_combinations_file}")
    duplicated_combinations.to_csv(duplicated_combinations_file)

    # Count the frequency of each combination type
    duplicated_combination_counts = duplicated_combinations.value_counts()
    # Print results
    logging.info(f"Combination counts:\n{pprint.pformat(duplicated_combination_counts)}")

    # Get uniqe CpG chr pos
    duplicated_cpg_chr_pos_cond_left_one = cpg_chr_pos_df.duplicated(subset=["chr_pos"])
    unique_cpg_chr_pos_df = cpg_chr_pos_df[~duplicated_cpg_chr_pos_cond_left_one]
    if unique_cpg_chr_pos_df.duplicated(subset=["chr_pos"]).any():
        raise ValueError("Unique CpG chr pos df still has duplicates")
    num_cpg_after_unique = len(unique_cpg_chr_pos_df)
    unique_cpg_chr_pos_df.sort_values("chr_pos", inplace=True)
    unique_cpg_chr_pos_df.reset_index(drop=True, inplace=True)
    logging.info(f"Number of CpG before unique: {num_cpg_before_unique:,}")
    logging.info(f"Number of CpG after unique: {num_cpg_after_unique:,}")

    output_file = output_dir / "cpg_chr_pos_df.parquet"
    logging.info(f"Writing output file: {output_file}")
    unique_cpg_chr_pos_df.to_parquet(output_file)

    output_duplicates_file = output_dir / "duplicated_cpg_chr_pos_df.parquet"
    logging.info(f"Writing output file: {output_duplicates_file}")
    duplicated_cpg_chr_pos_df.to_parquet(output_duplicates_file)

    output_mapping_file = output_dir / "cpg_chr_pos_source_to_idx_mapping.json"
    logging.info(f"Writing output file: {output_mapping_file}")
    with open(output_mapping_file, "w") as f:
        json.dump(cpg_chr_pos_source_to_idx_mapping, f, indent=4)

    output_log_json_file = output_dir / "merge_cpg_and_stats.log"
    with open(output_log_json_file, "w") as f:
        f.write(f"Combination counts:\n{pprint.pformat(duplicated_combination_counts)}\n")
        f.write(f"Number of CpG before unique: {num_cpg_before_unique:,}\n")
        f.write(f"Number of CpG after unique: {num_cpg_after_unique:,}\n")


def get_combination_type(group_indices):
    # Convert list to sorted tuple of unique values
    unique_groups = tuple(sorted(set(group_indices)))

    # Map the combinations to descriptive strings
    combination_map = {
        (0,): "0",
        (1,): "1",
        (2,): "2",
        (0, 1): "0+1",
        (0, 2): "0+2",
        (1, 2): "1+2",
        (0, 1, 2): "0+1+2",
    }

    return combination_map.get(unique_groups, "other")


if __name__ == "__main__":
    app.run(main)
