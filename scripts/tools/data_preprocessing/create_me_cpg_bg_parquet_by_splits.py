"""
python scripts/tools/data_preprocessing/create_me_cpg_bg_parquet_by_splits.py \
    --input_me_parquet_file data/parquet/241231-tcga_wgbs/me.parquet \
    --input_cpg_bg_parquet_file data/parquet/241231-tcga_wgbs/cpg_bg.parquet \
    --input_cpg_island_tuple_parquet_file data/parquet/241231-tcga_wgbs/cpg_island_tuple.parquet \
    --input_cpg_split_file data/parquet/241231-tcga_wgbs/metadata/cpg_split/index_files/train.parquet \
    --input_sample_split_file data/parquet/241231-tcga_wgbs/metadata/subset_sample_split/sample_tissue_count_with_idx.csv \
    --output_dir misc/test_processed_parquet/tcga_wgbs \
    --output_file_name 241231-tcga_wgbs \
    --output_chunk_size 10000 \
    --debug --overwrite
"""

import math
import multiprocessing
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_me_parquet_file", None, "Path to the parquet file")
flags.DEFINE_string("input_cpg_bg_parquet_file", None, "Path to the parquet file")
flags.DEFINE_string("input_cpg_island_tuple_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_me_parquet_file")
flags.mark_flag_as_required("input_cpg_bg_parquet_file")
flags.mark_flag_as_required("input_cpg_island_tuple_parquet_file")

flags.DEFINE_string("input_cpg_split_file", None, "Path to the split file")
flags.DEFINE_string("input_sample_split_file", None, "Path to the split file")
flags.mark_flag_as_required("input_cpg_split_file")
flags.mark_flag_as_required("input_sample_split_file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output directory")
flags.mark_flag_as_required("output_file_name")
flags.DEFINE_integer("output_chunk_size", 100, "Output chunk size")
flags.DEFINE_integer("seed", 42, "Random seed")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_boolean("debug", False, "Enable debug mode")
flags.DEFINE_integer("num_workers", 20, "Number of worker processes to use")

FLAGS = flags.FLAGS


def read_csv_or_parquet(input_file):
    input_file = Path(input_file)
    if input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    elif input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file extension: {input_file.suffix}")
    return df


def init_read_process(input_cpg_split_file, input_sample_split_file, output_dir):
    init_read_process.input_cpg_split_file = input_cpg_split_file
    init_read_process.input_sample_split_file = input_sample_split_file
    init_read_process.output_dir = output_dir

    logging.info(f"Reading CpG split file: {input_cpg_split_file}")
    cpg_split_df = read_csv_or_parquet(input_cpg_split_file)
    logging.info(f"Reading sample split file: {input_sample_split_file}")
    sample_split_df = read_csv_or_parquet(input_sample_split_file)

    init_read_process.cpg_split_df = cpg_split_df
    init_read_process.sample_split_df = sample_split_df


def create_me_cpg_bg_by_cpg_sample_split(args):
    """
    create me cpg bg by cpg & sample split.
    We also need to reindex the all_me_cpg_bg_df to match the cpg_split_df index
    """
    cpg_split_df = getattr(init_read_process, "cpg_split_df", None)
    sample_split_df = getattr(init_read_process, "sample_split_df", None)
    if cpg_split_df is None or sample_split_df is None:
        logging.error("Dataframes are not initialized in init_process")
        raise ValueError("Dataframes are not initialized in init_process")

    me_parquet_file, cpg_bg_parquet_file, cpg_island_tuple_file = args
    me_df = pd.read_parquet(me_parquet_file)
    cpg_bg_df = pd.read_parquet(cpg_bg_parquet_file)
    cpg_island_tuple_df = pd.read_parquet(cpg_island_tuple_file)
    if not (me_df.index == cpg_bg_df.index).all():
        raise ValueError(f"Index mismatch: {me_parquet_file} vs. {cpg_bg_parquet_file}")
    if not (me_df.index == cpg_island_tuple_df.index).all():
        raise ValueError(f"Index mismatch: {me_parquet_file} vs. {cpg_island_tuple_file}")

    # NOTE xk: the me_df index is not the same as the cpg_split_df index
    # We merge all tcga array, epic, wgbs cpg, and get their unique cpg idx
    # we need to reindex the all_me_cpg_bg_df to match the cpg_split_df index
    # NOTE xk: the index between `me_df`, `cpg_bg_df`, and `cpg_island_tuple_df` should be the same
    me_df_in_cpg_split_binary = me_df["Unnamed: 0"].isin(cpg_split_df["chr_pos"])
    me_df_in_cpg_split = me_df[me_df_in_cpg_split_binary]
    cpg_bg_df_in_cpg_split = cpg_bg_df[me_df_in_cpg_split_binary]
    cpg_island_tuple_df_in_cpg_split = cpg_island_tuple_df[me_df_in_cpg_split_binary]

    # NOTE xk: cpg_chr_pos is the index of the ME dataframe, so it does not count as a column in `sample_idx`
    me_df_in_cpg_split = me_df_in_cpg_split.rename(columns={"Unnamed: 0": "cpg_chr_pos"})
    me_df_cpg_chr_pos = me_df_in_cpg_split["cpg_chr_pos"]
    me_df_in_cpg_split = me_df_in_cpg_split.drop(columns=["cpg_chr_pos"])

    selected_columns = me_df_in_cpg_split.columns.isin(sample_split_df["sample_name"])
    me_cpg_bg_df_in_cpg_sample_splits = me_df_in_cpg_split.iloc[:, selected_columns]
    me_cpg_bg_df_in_cpg_sample_splits = pd.concat(
        [
            me_cpg_bg_df_in_cpg_sample_splits,
            cpg_bg_df_in_cpg_split["sequence"],
            cpg_island_tuple_df_in_cpg_split["cpg_island_tuple"],
        ],
        axis=1,
    )
    me_cpg_bg_df_in_cpg_sample_splits["cpg_chr_pos"] = me_df_cpg_chr_pos

    # NOTE xk: reindex the all_me_cpg_bg_df (its cpg index) to match the cpg_split_df index
    used_cpg_split_df = cpg_split_df[cpg_split_df["chr_pos"].isin(me_df_cpg_chr_pos)]
    me_cpg_bg_df_in_cpg_sample_splits.index = me_cpg_bg_df_in_cpg_sample_splits["cpg_chr_pos"].map(
        dict(zip(used_cpg_split_df["chr_pos"], used_cpg_split_df.index))
    )

    return me_cpg_bg_df_in_cpg_sample_splits


def init_save_process(input_sample_split_file, seed, all_me_cpg_bg_df, output_chunk_size, output_dir):
    logging.info(f"Reading sample split file: {input_sample_split_file}")
    sample_split_df = read_csv_or_parquet(input_sample_split_file)
    init_save_process.sample_split_df = sample_split_df
    init_save_process.rng = np.random.default_rng(seed)

    init_save_process.all_me_cpg_bg_df = all_me_cpg_bg_df
    init_save_process.output_chunk_size = output_chunk_size
    init_save_process.output_dir = output_dir

    init_save_process.sample_name_to_sample_idx = pd.Series(
        sample_split_df["sample_idx"].values, sample_split_df["sample_name"].values
    )
    init_save_process.sample_name_to_tissue_idx = pd.Series(
        sample_split_df["tissue_idx"].values, sample_split_df["sample_name"].values
    )


def save_me_cpg_bg_by_cpg_sample_splits(args):
    chunk_idx, start_idx, end_idx = args

    all_me_cpg_bg_df = getattr(init_save_process, "all_me_cpg_bg_df", None)
    output_chunk_size = getattr(init_save_process, "output_chunk_size", None)
    output_dir = getattr(init_save_process, "output_dir", None)
    rng = getattr(init_save_process, "rng", None)
    sample_name_to_sample_idx = getattr(init_save_process, "sample_name_to_sample_idx", None)
    sample_name_to_tissue_idx = getattr(init_save_process, "sample_name_to_tissue_idx", None)

    if (
        all_me_cpg_bg_df is None
        or output_chunk_size is None
        or output_dir is None
        or rng is None
        or sample_name_to_sample_idx is None
        or sample_name_to_tissue_idx is None
    ):
        logging.error("Variables are not initialized in init_save_process")
        raise ValueError("Variables are not initialized in init_save_process")

    output_me_cpg_bg_df = all_me_cpg_bg_df.iloc[start_idx:end_idx]
    output_me_cpg_bg_df_melt = output_me_cpg_bg_df.melt(
        id_vars=["cpg_chr_pos", "sequence", "cpg_island_tuple"],
        var_name="sample_name",
        value_name="methylation",
        ignore_index=False,
    )

    output_me_cpg_bg_df_melt["sample_idx"] = output_me_cpg_bg_df_melt["sample_name"].map(sample_name_to_sample_idx)
    output_me_cpg_bg_df_melt["tissue_idx"] = output_me_cpg_bg_df_melt["sample_name"].map(sample_name_to_tissue_idx)
    output_me_cpg_bg_df_melt.reset_index(names="cpg_idx", inplace=True)

    if output_me_cpg_bg_df_melt["sample_idx"].isna().any():
        raise ValueError("Sample name not found in sample split file")
    if output_me_cpg_bg_df_melt["sample_idx"].isnull().any():
        raise ValueError("Sample name is null")

    # Shuffle the data
    output_me_cpg_bg_df_melt = output_me_cpg_bg_df_melt.sample(frac=1, random_state=rng)

    output_file = output_dir / f"{chunk_idx:05d}.parquet"
    # NOTE xk: index=False, as we do not need it in huggingface datasets ("__index_level_0__").
    output_me_cpg_bg_df_melt.to_parquet(output_file, index=False)

    return output_file


def main(argv):
    output_dir = Path(FLAGS.output_dir) / FLAGS.output_file_name

    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.info(f"{output_dir} already exists. Skipping...")
            return
        else:
            logging.info(f"Overwriting {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    input_me_parquet_file = Path(FLAGS.input_me_parquet_file)
    input_cpg_bg_parquet_file = Path(FLAGS.input_cpg_bg_parquet_file)
    input_cpg_island_tuple_parquet_file = Path(FLAGS.input_cpg_island_tuple_parquet_file)

    me_parquet_files = sorted(input_me_parquet_file.glob("*.parquet"))
    cpg_bg_parquet_files = sorted(input_cpg_bg_parquet_file.glob("*.parquet"))
    input_cpg_island_tuple_parquet_files = sorted(input_cpg_island_tuple_parquet_file.glob("*.parquet"))
    if len(me_parquet_files) == 0:
        logging.error(f"No parquet files found in {input_me_parquet_file}")
    if len(cpg_bg_parquet_files) == 0:
        logging.error(f"No parquet files found in {input_cpg_bg_parquet_file}")
    if len(input_cpg_island_tuple_parquet_files) == 0:
        logging.error(f"No parquet files found in {input_cpg_island_tuple_parquet_file}")
    if len(me_parquet_files) != len(cpg_bg_parquet_files):
        logging.error(
            f"Number of parquet files in {input_me_parquet_file} and {input_cpg_bg_parquet_file} do not match: {len(me_parquet_files)} vs {len(cpg_bg_parquet_files)}"
        )
        return
    if len(me_parquet_files) != len(input_cpg_island_tuple_parquet_files):
        logging.error(
            f"Number of parquet files in {input_me_parquet_file} and {input_cpg_island_tuple_parquet_file} do not match: {len(me_parquet_files)} vs {len(input_cpg_island_tuple_parquet_files)}"
        )
        return

    # check if the file names are the same
    for me_parquet_file, cpg_bg_parquet_file in zip(me_parquet_files, cpg_bg_parquet_files):
        if me_parquet_file.stem != cpg_bg_parquet_file.stem:
            logging.error(f"File names do not match: {me_parquet_file.stem} vs {cpg_bg_parquet_file.stem}")
            return
    for me_parquet_file, cpg_island_tuple_parquet_file in zip(me_parquet_files, input_cpg_island_tuple_parquet_files):
        if me_parquet_file.stem != cpg_island_tuple_parquet_file.stem:
            logging.error(f"File names do not match: {me_parquet_file.stem} vs {cpg_island_tuple_parquet_file.stem}")
            return

    logging.info(f"Reading ME parquet file: {input_me_parquet_file}: {len(me_parquet_files)} files")
    logging.info(f"Reading CpG BG parquet file: {input_cpg_bg_parquet_file}: {len(cpg_bg_parquet_files)} files")
    logging.info(
        f"Reading CpG island tuple parquet file: {input_cpg_island_tuple_parquet_file}: {len(input_cpg_island_tuple_parquet_files)} files"
    )

    input_cpg_split_file = Path(FLAGS.input_cpg_split_file)
    input_sample_split_file = Path(FLAGS.input_sample_split_file)

    if FLAGS.debug:
        logging.info("Debug mode enabled")
        init_read_process(input_cpg_split_file, input_sample_split_file, output_dir)
        all_me_cpg_bg_df = create_me_cpg_bg_by_cpg_sample_split(
            (me_parquet_files[0], cpg_bg_parquet_files[0], input_cpg_island_tuple_parquet_files[0])
        )

        output_chunk_size = FLAGS.output_chunk_size
        init_save_process(input_sample_split_file, 0, all_me_cpg_bg_df, output_chunk_size, output_dir)
        save_me_cpg_bg_by_cpg_sample_splits((0, 0, output_chunk_size))
        return

    # Read me and cpg bg
    all_me_cpg_bg_df = []
    with multiprocessing.Pool(
        FLAGS.num_workers,
        initializer=init_read_process,
        initargs=(input_cpg_split_file, input_sample_split_file, output_dir),
    ) as pool:
        with tqdm.tqdm(total=len(me_parquet_files)) as pbar:
            for me_cpg_bg_df in pool.imap_unordered(
                create_me_cpg_bg_by_cpg_sample_split,
                zip(me_parquet_files, cpg_bg_parquet_files, input_cpg_island_tuple_parquet_files),
            ):
                all_me_cpg_bg_df.append(me_cpg_bg_df)
                pbar.update()

    # Save me cpg bg
    all_me_cpg_bg_df = pd.concat(all_me_cpg_bg_df)
    output_chunk_size = FLAGS.output_chunk_size
    seed = FLAGS.seed
    cpg_split_df = read_csv_or_parquet(input_cpg_split_file)
    # NOTE xk: We **have already reindex** the all_me_cpg_bg_df
    # to match the cpg_split_df index in `create_me_cpg_bg_by_cpg_sample_split`
    # the original me_df index is not the same as the cpg_split_df index,
    # as we merge all tcga array, epic, wgbs cpg, and get their unique cpg idx
    if not (all_me_cpg_bg_df.index.isin(cpg_split_df.index)).all():
        raise ValueError("Index mismatch for all ME CpG BG dataframes")
    all_me_cpg_bg_df = all_me_cpg_bg_df.reindex(cpg_split_df.index)

    with multiprocessing.Pool(
        FLAGS.num_workers,
        initializer=init_save_process,
        initargs=(input_sample_split_file, seed, all_me_cpg_bg_df, output_chunk_size, output_dir),
    ) as pool:
        # Create a list to store all async results
        async_results = []

        num_chunks = math.ceil(len(all_me_cpg_bg_df) / output_chunk_size)
        with tqdm.tqdm(total=num_chunks, desc="Saving") as pbar:
            for chunk_idx, start_idx in enumerate(range(0, len(all_me_cpg_bg_df), output_chunk_size)):
                end_idx = min(start_idx + output_chunk_size, len(all_me_cpg_bg_df))
                # Store the async result
                result = pool.apply_async(save_me_cpg_bg_by_cpg_sample_splits, args=((chunk_idx, start_idx, end_idx),))
                async_results.append(result)

            # Wait for all tasks to complete
            for result in async_results:
                result_ = result.get()  # This will raise any exceptions that occurred in the worker processes
                pbar.set_postfix_str(f"Processed {result_}")
                pbar.update()


if __name__ == "__main__":
    app.run(main)
