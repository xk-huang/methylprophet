"""
PARQUET_DIR_NAME=241231-tcga_array
INPUT_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_bg.parquet"
INPUT_COLUMN_NAME="CpG_location"
python scripts/tools/data_preprocessing/stats_cpg_per_chr.py \
    --input_parquet_file "${INPUT_PARQUET_FILE}" \
    --input_column_name "${INPUT_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/cpg_per_chr_stats" \
    --input_reindex_cpg_idx_file data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix/cpg_chr_pos_df.parquet \
    --overwrite
"""

import multiprocessing
import shutil
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")
flags.DEFINE_string("input_column_name", None, "Column name to check for cpg per chr values")
flags.mark_flag_as_required("input_column_name")
flags.DEFINE_string("input_reindex_cpg_idx_file", None, "Path to the cpg index file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_integer("num_workers", 20, "Number of worker processes to use")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_boolean("debug", False, "Overwrite the output directory if it exists")


FLAGS = flags.FLAGS


def init_process(input_column_name, output_dir):
    init_process.input_column_name = input_column_name
    init_process.output_dir = output_dir


def stats_cpg_per_chr_in_parquet(input_parquet_file):
    input_column_name = getattr(init_process, "input_column_name", None)
    output_dir = getattr(init_process, "output_dir", None)
    if input_column_name is None:
        logging.error("Column name is not initialized in init_process")
        raise ValueError("Column name is not initialized in init_process")
    if output_dir is None:
        logging.error("Output directory is not initialized in init_process")
        raise ValueError("Output directory is not initialized in init_process")

    df = pd.read_parquet(input_parquet_file, columns=[input_column_name])
    chr_df = df.apply(lambda x: x[input_column_name].split("_")[0], axis=1)
    pos_df = df.apply(lambda x: int(eval(x[input_column_name].split("_")[1])), axis=1)
    chr_pos_df = pd.concat([chr_df, pos_df, df], axis=1)
    chr_pos_df.columns = ["chr", "pos", "chr_pos"]

    return chr_pos_df


def main(argv):
    input_parquet_file = Path(FLAGS.input_parquet_file)
    if input_parquet_file.is_file():
        input_parquet_files = [input_parquet_file]
    else:
        input_parquet_files = sorted(input_parquet_file.glob("*.parquet"))

    output_dir = Path(FLAGS.output_dir)
    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"Output directory already exists: {output_dir}")
            return
        else:
            logging.warning(f"Overwriting the output directory: {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if FLAGS.debug:
        logging.info("Debug mode enabled")
        init_process(FLAGS.input_column_name, output_dir)
        stats_cpg_per_chr_in_parquet(input_parquet_files[0])
        return

    results = []
    with multiprocessing.Pool(
        processes=FLAGS.num_workers,
        initializer=init_process,
        initargs=(FLAGS.input_column_name, output_dir),
    ) as pool:
        with tqdm.tqdm(total=len(input_parquet_files)) as pbar:
            for result in pool.imap_unordered(stats_cpg_per_chr_in_parquet, input_parquet_files):
                results.append(result)
                pbar.update()

    chr_pos_df = pd.concat(results)
    if FLAGS.input_reindex_cpg_idx_file is not None:
        logging.info(f"Loading cpg index file: {FLAGS.input_reindex_cpg_idx_file}, update cpg idx")

        cpg_idx_df = pd.read_parquet(FLAGS.input_reindex_cpg_idx_file)
        old_index_head = chr_pos_df.index[0:10]

        chr_pos_index_mapping = dict(zip(cpg_idx_df["chr_pos"], cpg_idx_df.index))
        chr_pos_df.index = chr_pos_df["chr_pos"].map(chr_pos_index_mapping)

        new_index_head = chr_pos_df.index[0:10]
        logging.info(f"index head: {old_index_head} -> {new_index_head}")

        del old_index_head, new_index_head, cpg_idx_df, chr_pos_index_mapping
    else:
        logging.info("No cpg index file provided, keep the original cpg idx")
    chr_pos_df.sort_index(inplace=True)

    chr_pos_df_path = output_dir / "cpg_chr_pos_df.parquet"
    chr_pos_df.to_parquet(chr_pos_df_path)
    logging.info(f"Saved chr df to: {chr_pos_df_path}")

    cpg_per_chr_counts = chr_pos_df["chr"].value_counts()
    cpg_per_chr_counts.sort_values(ascending=False)
    cpg_per_chr_counts_path = output_dir / "cpg_per_chr_counts-sorted_by_counts.csv"
    cpg_per_chr_counts.to_csv(cpg_per_chr_counts_path)
    logging.info(f"Saved cpg per chr counts to: {cpg_per_chr_counts_path}")

    cpg_per_chr_counts_path = output_dir / "cpg_per_chr_counts-sorted_by_chr.csv"

    def _sort_index_func(x):
        def _chr_to_int(chr_label):
            chr_label = chr_label.strip("chr")
            if chr_label.isdigit():
                return int(chr_label)
            elif chr_label.upper() == "X":
                return 23
            elif chr_label.upper() == "Y":
                return 24
            else:
                return 25  # for other non-numeric labels

        return x.map(_chr_to_int)

    cpg_per_chr_counts = cpg_per_chr_counts.sort_index(key=_sort_index_func)
    cpg_per_chr_counts.to_csv(cpg_per_chr_counts_path)
    logging.info(f"Saved cpg per chr counts to: {cpg_per_chr_counts_path}")


if __name__ == "__main__":
    app.run(main)
