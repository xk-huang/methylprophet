"""
python scripts/tools/eval/stats_plot_cpg_nbase_intersection.py \
    --input_parquet_dir data/parquet/241213-encode_wgbs/metadata/cpg_split/train_0_9_val_0_1 \
    --num_nbase 1000 \
    --output_dir data/parquet/241213-encode_wgbs/metadata/cpg_split/train_0_9_val_0_1/cpg_nbase_intersected \
    --num_workers 120


# rm -rf data/parquet/241231-tcga/metadata/cpg_stats
mkdir -p data/parquet/241231-tcga/metadata/cpg_stats/train.parquet

cp data/parquet/241231-tcga_array/metadata/cpg_split/index_files/val.parquet data/parquet/241231-tcga/metadata/cpg_stats/

cp data/parquet/241231-tcga_array/metadata/cpg_split/index_files/train.parquet data/parquet/241231-tcga/metadata/cpg_stats/train.parquet/00000.parquet
cp data/parquet/241231-tcga_epic/metadata/cpg_split/index_files/train.parquet data/parquet/241231-tcga/metadata/cpg_stats/train.parquet/00001.parquet
cp data/parquet/241231-tcga_wgbs/metadata/cpg_split/index_files/train.parquet data/parquet/241231-tcga/metadata/cpg_stats/train.parquet/00002.parquet

python scripts/tools/eval/stats_plot_cpg_nbase_intersection.py \
    --input_parquet_dir data/parquet/241231-tcga/metadata/cpg_stats/ \
    --num_nbase 1000 \
    --output_dir data/parquet/241231-tcga/metadata/cpg_stats/cpg_nbase_intersected \
    --num_workers 120
"""

import json
import multiprocessing as mp
import shutil
from pathlib import Path
from pprint import pformat

import colorcet
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_dir", None, "Path to the input processed dataset dir")
flags.mark_flag_as_required("input_parquet_dir")
flags.DEFINE_alias("i", "input_parquet_dir")


flags.DEFINE_integer("num_nbase", 1000, "Number of N-base")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("output_name", "cpg_nbase_intersected", "Output name")

flags.DEFINE_integer("num_workers", 8, "Number of worker processes for parallel processing")
flags.DEFINE_bool("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS


# def get_cpg_chr_pos_from_processed_parquet(parquet_file):
#     breakpoint()
#     cpg_chr_pos_df = pd.read_parquet(parquet_file, columns=["cpg_idx", "sample_idx", "cpg_chr_pos"])
#     # NOTE: cpg_chr_pos has string like "chr5_126673519"
#     # We need to split it into chromosome and position
#     cpg_chr_pos_df["chr"] = cpg_chr_pos_df["cpg_chr_pos"].str.split("_").str[0]
#     cpg_chr_pos_df["pos"] = cpg_chr_pos_df["cpg_chr_pos"].str.split("_").str[1].astype(int)
#     return cpg_chr_pos_df


def get_cpg_chr_pos_from_train_val_split_parquet(parquet_file):
    return pd.read_parquet(parquet_file)


def get_cpg_chr_pos_from_parquet(input_path: Path, num_workers=8, debug=False):
    if input_path.is_file():
        df = get_cpg_chr_pos_from_train_val_split_parquet(input_path)

        # deduplicate
        num_rows = len(df)
        df = df.drop_duplicates(subset=["chr_pos"])
        num_rows_after_dedup = len(df)
        logging.info(f"Loaded {num_rows} rows from {input_path}, deduplicated to {num_rows_after_dedup}")
        return df

    parquet_files = sorted(input_path.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {input_path}")

    if debug:
        cpg_id_sample_id_list = []
        for parquet_file in tqdm.tqdm(parquet_files, desc="Processing files"):
            cpg_id_sample_id = get_cpg_chr_pos_from_train_val_split_parquet(parquet_file)
            cpg_id_sample_id_list.append(cpg_id_sample_id)
            break

        df = pd.concat(cpg_id_sample_id_list)
        # deduplicate
        num_rows = len(df)
        df = df.drop_duplicates(subset=["chr_pos"])
        num_rows_after_dedup = len(df)
        logging.info(f"Loaded {num_rows} rows from {input_path}, deduplicated to {num_rows_after_dedup}")
        return pd.concat(cpg_id_sample_id_list)

    with mp.Pool(processes=num_workers) as pool:
        cpg_id_sample_id_list = list(
            tqdm.tqdm(
                pool.imap(get_cpg_chr_pos_from_train_val_split_parquet, parquet_files),
                total=len(parquet_files),
                desc="Getting cpg_idx_sample_idx",
            )
        )

    df = pd.concat(cpg_id_sample_id_list)
    # deduplicate
    num_rows = len(df)
    df = df.drop_duplicates(subset=["chr_pos"])
    num_rows_after_dedup = len(df)
    logging.info(f"Loaded {num_rows} rows from {input_path}, deduplicated to {num_rows_after_dedup}")
    return df


def get_chr_max_pos(cpg_chr_pos_df):
    chr_max_pos = cpg_chr_pos_df.groupby("chr")["pos"].max()
    return chr_max_pos


def get_nbase_count_one_chr(
    cpg_chr_pos_df: pd.DataFrame,
    chr_name: str,
    chr_max_pos: int,
    num_nbase: int,
):
    nbase_count = np.zeros(chr_max_pos + num_nbase)
    chr_cpg_pos_df_ = cpg_chr_pos_df[cpg_chr_pos_df["chr"] == chr_name]
    for pos in tqdm.tqdm(
        chr_cpg_pos_df_["pos"],
        desc=f"Counting N-base in {chr_name} with {len(chr_cpg_pos_df_)} CpGs, num_nbase={num_nbase}",
    ):
        pos = pos + num_nbase // 2  # NOTE: Shift by num_nbase // 2 due to the window size
        left_pos = max(0, pos - num_nbase // 2)
        right_pos = min(chr_max_pos, pos + num_nbase // 2)
        nbase_count[left_pos:right_pos] += 1
    return nbase_count


def plot_nbase_count(nbase_count_one_chr_dict, colormaps, output_file_path):
    # Create figure with subplots
    if len(nbase_count_one_chr_dict) != len(colormaps):
        raise ValueError("The number of datasets and colormaps must be the same")

    fig, axes = plt.subplots(1, len(nbase_count_one_chr_dict), figsize=(20, 6))
    axes = axes.flatten()

    # Different colormaps for each subplot
    titles = nbase_count_one_chr_dict.keys()
    datasets = nbase_count_one_chr_dict.values()

    for idx, (arr, cmap, title, ax) in enumerate(zip(datasets, colormaps, titles, axes)):
        # Create a DataFrame with indices and values
        data = pd.DataFrame({"Index": np.arange(len(arr)), "Value": arr})

        x_min = data["Index"].min()
        x_max = data["Index"].max()
        y_min = data["Value"].min()
        y_max = data["Value"].max()
        # y_max = y_max + 0.1 * (y_max - y_min)

        # Define canvas size and data ranges
        height = 400
        width = 1000
        canvas = ds.Canvas(plot_width=width, plot_height=height, x_range=(x_min, x_max), y_range=(y_min, y_max))

        # Aggregate data using points
        agg = canvas.points(data, "Index", "Value")
        agg = tf.spread(agg, px=1)

        # Convert the aggregation to an image
        img = tf.shade(agg, cmap=cmap)

        # Display the image
        extent = [data["Index"].min(), data["Index"].max(), data["Value"].min(), data["Value"].max()]

        ax.imshow(img.to_pil(), aspect="auto", extent=extent, origin="lower")

        # Add title and axis labels
        if title == "train.parquet":
            title = "Train CpG NBase Count"
        elif title == "val.parquet":
            title = "Val CpG NBase Count"
        ax.set_title(title)

        ax.set_xlabel("NBase Index")
        if idx == 0 or idx == 1:
            ax.set_ylabel("# CpGs Having the NBase")
        elif idx == 2:
            ax.set_ylabel("Train-Val CpG Intersected or Not")
        else:
            ax.set_ylabel

        # Customize ticks
        ax.set_xticks(np.linspace(x_min, x_max, num=10, dtype=int))
        ax.set_yticks(np.linspace(y_min, y_max, num=10, dtype=int))

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved {output_file_path}")


def main(_):
    input_parquet_dir = Path(FLAGS.input_parquet_dir)
    logging.info(f"Reading {input_parquet_dir}")

    split_dir_list = [input_parquet_dir / "train.parquet", input_parquet_dir / "val.parquet"]

    logging.info(f"Found {len(split_dir_list)} split directories:\n{pformat(split_dir_list)}")

    num_workers = FLAGS.num_workers
    debug = FLAGS.debug

    output_dir = FLAGS.output_dir
    if output_dir is None:
        output_dir = input_parquet_dir
    else:
        output_dir = Path(FLAGS.output_dir)
    output_dir = output_dir / FLAGS.output_name
    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_dir} already exists. Skipping...")
            return
        else:
            shutil.rmtree(output_dir)
            logging.warning(f"Overwriting {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    cpg_chr_pos_df_dict = {}
    chr_max_pos_dict = {}
    for split_dir in split_dir_list:
        split_name = split_dir.name
        logging.info(f"Processing {split_name}")

        cpg_chr_pos_df = get_cpg_chr_pos_from_parquet(split_dir, num_workers, debug)
        chr_max_pos = get_chr_max_pos(cpg_chr_pos_df)

        cpg_chr_pos_df_dict[split_name] = cpg_chr_pos_df
        chr_max_pos_dict[split_name] = chr_max_pos

    chr_max_pos = pd.concat(chr_max_pos_dict)
    chr_max_pos = chr_max_pos.groupby("chr").max()

    intersected_info_dict_list = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm.tqdm(total=len(chr_max_pos))
        pool_results = []
        for chr_name in chr_max_pos.index:
            result = pool.apply_async(
                stats_intersected_one_chr,
                args=(output_dir, cpg_chr_pos_df_dict, chr_name, chr_max_pos[chr_name]),
            )
            pool_results.append(result)

        for result in pool_results:
            intersected_info_dict = result.get()
            intersected_info_dict_list.append(intersected_info_dict)
            pbar.set_postfix_str(f"Finished {chr_name}")
            pbar.update(1)

    intersected_info_json_path = output_dir / "intersected_info.json"
    with open(intersected_info_json_path, "w") as f:
        json.dump(intersected_info_dict_list, f, indent=4)
    intersected_info_df = pd.DataFrame(intersected_info_dict_list)
    intersected_info_csv_path = output_dir / "intersected_info.csv"
    intersected_info_df.to_csv(intersected_info_csv_path)
    logging.info(f"Saved {intersected_info_json_path}")


def stats_intersected_one_chr(output_dir, cpg_chr_pos_df_dict, chr_name, max_pos):
    nbase_count_one_chr_dict = {}
    for split_name, cpg_chr_pos_df in cpg_chr_pos_df_dict.items():
        nbase_count_one_chr = get_nbase_count_one_chr(
            cpg_chr_pos_df,
            chr_name,
            max_pos,
            FLAGS.num_nbase,
        )
        nbase_count_one_chr_dict[split_name] = nbase_count_one_chr

    nbase_count_sum = np.sum(list(nbase_count_one_chr_dict.values()), axis=0)
    intersected_dict = {}
    for split_name, nbase_count_one_chr in nbase_count_one_chr_dict.items():
        # NOTE: if the sum of N-base count is not equal to the sum of N-base count of one chromosome
        # we mark it as potential intersected
        intersected = nbase_count_sum != nbase_count_one_chr
        intersected_dict[split_name] = intersected

        # NOTE: if all the N-base count intersected, it means that the N-base count is the same
    nbase_count_intersected = np.all(list(intersected_dict.values()), axis=0)
    num_intersected = nbase_count_intersected.sum()
    num_nbase = len(nbase_count_intersected)
    num_used_nbase = np.count_nonzero(np.sum(list(intersected_dict.values()), axis=0))

    intersected_ratio_for_all_nbase = num_intersected / num_nbase
    intersected_ratio_for_used_nbase = num_intersected / num_used_nbase

    intersected_info_dict = {
        "chr_name": str(chr_name),
        "intersected_ratio_for_all_nbase": float(intersected_ratio_for_all_nbase),
        "intersected_ratio_for_used_nbase": float(intersected_ratio_for_used_nbase),
        "num_intersected": int(num_intersected),
        "num_nbase": int(num_nbase),
        "num_used_nbase": int(num_used_nbase),
        "num_train_cpg": get_num_cpg_per_chr(cpg_chr_pos_df_dict["train.parquet"], chr_name),
        "num_val_cpg": get_num_cpg_per_chr(cpg_chr_pos_df_dict["val.parquet"], chr_name),
    }
    intersected_info = (
        f"Train-Val CpG Intersection in {chr_name}: All Nbase {num_intersected:,} / {num_nbase:,} ({intersected_ratio_for_all_nbase*100:.2f} %);\n"
        f"Train-Val CpG Intersection in in {chr_name}: Used Nbase {num_intersected:,} / {num_used_nbase:,} ({intersected_ratio_for_used_nbase * 100:.2f} %)"
    )
    logging.info(intersected_info)

    # Plot N-base count
    plot_dict = nbase_count_one_chr_dict.copy()
    plot_dict[intersected_info] = nbase_count_intersected.astype(np.float32)
    colormaps = [colorcet.fire, colorcet.kbc, colorcet.kbc]

    plot_nbase_count(plot_dict, colormaps, output_dir / f"cpg_intersected-{chr_name}.png")

    return intersected_info_dict


def get_num_cpg_per_chr(cpg_chr_pos_df, chr_name):
    num_cpg_by_chr = len(cpg_chr_pos_df[cpg_chr_pos_df["chr"] == chr_name])
    return num_cpg_by_chr


if __name__ == "__main__":
    app.run(main)
