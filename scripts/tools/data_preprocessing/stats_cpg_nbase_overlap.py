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


flags.DEFINE_integer("num_nbase", 1200, "Number of N-base")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("output_name", "cpg_nbase_overlap", "Output name")

flags.DEFINE_integer("num_workers", 8, "Number of worker processes for parallel processing")
flags.DEFINE_bool("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS


# def get_cpg_chr_pos_from_processed_parquet(parquet_file):
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
        return get_cpg_chr_pos_from_train_val_split_parquet(input_path)

    parquet_files = sorted(input_path.glob("*.parquet"))
    logging.info(f"Found {len(parquet_files)} parquet files in {input_path}")

    if debug:
        cpg_id_sample_id_list = []
        for parquet_file in tqdm.tqdm(parquet_files, desc="Processing files"):
            cpg_id_sample_id = get_cpg_chr_pos_from_train_val_split_parquet(parquet_file)
            cpg_id_sample_id_list.append(cpg_id_sample_id)
            break
        return pd.concat(cpg_id_sample_id_list)

    with mp.Pool(processes=num_workers) as pool:
        cpg_id_sample_id_list = list(
            tqdm.tqdm(
                pool.imap(get_cpg_chr_pos_from_train_val_split_parquet, parquet_files),
                total=len(parquet_files),
                desc="Getting cpg_idx_sample_idx",
            )
        )

    return pd.concat(cpg_id_sample_id_list)


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
        ax.set_title(title)
        ax.set_xlabel("NBase Index")
        ax.set_ylabel("Count")

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

    overlap_info_dict_list = []
    with mp.Pool(processes=num_workers) as pool:
        pbar = tqdm.tqdm(total=len(chr_max_pos))
        pool_results = []
        for chr_name in chr_max_pos.index:
            result = pool.apply_async(
                stats_overlap_one_chr,
                args=(output_dir, cpg_chr_pos_df_dict, chr_name, chr_max_pos[chr_name]),
            )
            pool_results.append(result)

        for result in pool_results:
            overlap_info_dict = result.get()
            overlap_info_dict_list.append(overlap_info_dict)
            pbar.set_postfix_str(f"Finished {chr_name}")
            pbar.update(1)

    overlap_info_json_path = output_dir / "overlap_info.json"
    with open(overlap_info_json_path, "w") as f:
        json.dump(overlap_info_dict_list, f, indent=4)
    overlap_info_df = pd.DataFrame(overlap_info_dict_list)
    overlap_info_csv_path = output_dir / "overlap_info.csv"
    overlap_info_df.to_csv(overlap_info_csv_path)
    logging.info(f"Saved {overlap_info_json_path}")


def stats_overlap_one_chr(output_dir, cpg_chr_pos_df_dict, chr_name, max_pos):
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
    overlap_dict = {}
    for split_name, nbase_count_one_chr in nbase_count_one_chr_dict.items():
        # NOTE: if the sum of N-base count is not equal to the sum of N-base count of one chromosome
        # we mark it as potential overlap
        overlap = nbase_count_sum != nbase_count_one_chr
        overlap_dict[split_name] = overlap

        # NOTE: if all the N-base count overlap, it means that the N-base count is the same
    nbase_count_overlap = np.all(list(overlap_dict.values()), axis=0)
    num_overlap = nbase_count_overlap.sum()
    num_nbase = len(nbase_count_overlap)
    num_used_nbase = np.count_nonzero(np.sum(list(overlap_dict.values()), axis=0))

    overlap_ratio_for_all_nbase = num_overlap / num_nbase
    overlap_ratio_for_used_nbase = num_overlap / num_used_nbase

    overlap_info_dict = {
        "chr_name": str(chr_name),
        "overlap_ratio_for_all_nbase": float(overlap_ratio_for_all_nbase),
        "overlap_ratio_for_used_nbase": float(overlap_ratio_for_used_nbase),
        "num_overlap": int(num_overlap),
        "num_nbase": int(num_nbase),
        "num_used_nbase": int(num_used_nbase),
    }
    overlap_info = (
        f"Overlap in {chr_name}: all nbase {num_overlap:,}/{num_nbase:,} ({overlap_ratio_for_all_nbase*100:.2f} %);\n"
        f"Overlap in {chr_name}: used nbase {num_overlap:,}/{num_used_nbase:,} ({overlap_ratio_for_used_nbase * 100:.2f} %)"
    )
    logging.info(overlap_info)

    # Plot N-base count
    plot_dict = nbase_count_one_chr_dict.copy()
    plot_dict[overlap_info] = nbase_count_overlap.astype(np.float32)
    colormaps = [colorcet.fire, colorcet.kbc, colorcet.kbc]

    plot_nbase_count(plot_dict, colormaps, output_dir / f"cpg_overlap-{chr_name}.png")

    return overlap_info_dict


if __name__ == "__main__":
    app.run(main)
