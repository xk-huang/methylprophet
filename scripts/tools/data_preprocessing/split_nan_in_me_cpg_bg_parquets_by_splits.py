"""
python scripts/tools/data_preprocessing/split_nan_in_me_cpg_bg_parquets_by_splits.py \
    --i data/processed/241023-tcga_array-train_0_9_val_0_1-ind_cancer \
    --num_workers 10

Note:
num_workers=10, memory: ~360GB
"""

import multiprocessing as mp
import shutil
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_me_cpg_bg_dataset_dir", None, "Path to the directory with the ME CpG BG dataset")
flags.mark_flag_as_required("input_me_cpg_bg_dataset_dir")
flags.DEFINE_alias("i", "input_me_cpg_bg_dataset_dir")
flags.DEFINE_string("output_dir", None, "Path to the output directory")
flags.DEFINE_alias("o", "output_dir")

flags.DEFINE_integer("num_workers", 20, "Number of workers for parallel processing")

flags.DEFINE_bool("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_bool("debug", False, "Debug mode")


FLAGS = flags.FLAGS

DUPLICATED_GENE_ID_CSV_FILE_NAME = "duplicated_gene_id.csv"
GENE_EXPR_FILTERED_PARQUET_FILE_NAME = "gene_expr.filtered.parquet"
ME_CPG_BG_SPLIT_NAMES = [
    "train_cpg-train_sample.parquet",
    "train_cpg-val_sample.parquet",
    "val_cpg-train_sample.parquet",
    "val_cpg-val_sample.parquet",
]


def split_nan_in_me_cpg_bg_split_parquet(
    input_parquet: Path,
    output_nan_dir=None,
    output_non_nan_dir=None,
):
    if output_nan_dir is None:
        output_nan_dir = getattr(init_mp_process, "output_nan_dir", None)
        output_non_nan_dir = getattr(init_mp_process, "output_non_nan_dir", None)
    if output_nan_dir is None or output_non_nan_dir is None:
        raise ValueError("Output directories are not provided, run init_mp_process first or provide them as arguments")

    df = pd.read_parquet(input_parquet)
    is_nan_condition = df["methylation"].isna()
    nan_df = df[is_nan_condition]
    non_nan_df = df[~is_nan_condition]

    parquet_name = input_parquet.name
    output_nan_parquet = output_nan_dir / parquet_name
    output_non_nan_paruqet = output_non_nan_dir / parquet_name

    # NOTE xk: index=False, as we do not need it in huggingface datasets ("__index_level_0__").
    nan_df.to_parquet(output_nan_parquet, index=False)
    non_nan_df.to_parquet(output_non_nan_paruqet, index=False)
    return parquet_name


def init_mp_process(
    output_nan_dir,
    output_non_nan_dir,
):
    init_mp_process.output_nan_dir = output_nan_dir
    init_mp_process.output_non_nan_dir = output_non_nan_dir
    logging.info(
        f"Initialized multiprocessing process with output directories: {output_nan_dir}, {output_non_nan_dir}"
    )


def main(_):
    input_me_cpg_bg_dataset_dir = Path(FLAGS.input_me_cpg_bg_dataset_dir)
    if not input_me_cpg_bg_dataset_dir.exists():
        logging.fatal(f"Input ME CpG BG dataset directory {input_me_cpg_bg_dataset_dir} does not exist")
    logging.info(f"Reading ME CpG BG dataset from {input_me_cpg_bg_dataset_dir}")

    # Prepare output_dir
    output_dir = FLAGS.output_dir
    if output_dir is None:
        output_dir = input_me_cpg_bg_dataset_dir.parent
    else:
        output_dir = Path(output_dir)
    logging.info(f"Output directory: {output_dir}")

    output_nan_me_cpg_bg_dataset_dir = output_dir / (input_me_cpg_bg_dataset_dir.name + "-nan")
    output_non_nan_me_cpg_bg_dataset_dir = output_dir / (input_me_cpg_bg_dataset_dir.name + "-non_nan")
    if output_nan_me_cpg_bg_dataset_dir.exists() or output_non_nan_me_cpg_bg_dataset_dir.exists():
        if FLAGS.overwrite:
            logging.warning("Output directory exists and will be overwritten")
            shutil.rmtree(output_nan_me_cpg_bg_dataset_dir)
            shutil.rmtree(output_non_nan_me_cpg_bg_dataset_dir)
        else:
            logging.warning("Output directory exists and will not be overwritten")
            return

    output_nan_me_cpg_bg_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_non_nan_me_cpg_bg_dataset_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory for NaN ME CpG BG dataset: {output_nan_me_cpg_bg_dataset_dir}")
    logging.info(f"Output directory for non-NaN ME CpG BG dataset: {output_non_nan_me_cpg_bg_dataset_dir}")

    # Copy gene related files to the output directories
    for file_name in [DUPLICATED_GENE_ID_CSV_FILE_NAME, GENE_EXPR_FILTERED_PARQUET_FILE_NAME]:
        shutil.copy(input_me_cpg_bg_dataset_dir / file_name, output_nan_me_cpg_bg_dataset_dir / file_name)
        shutil.copy(input_me_cpg_bg_dataset_dir / file_name, output_non_nan_me_cpg_bg_dataset_dir / file_name)

    for me_cpg_bg_split_name in ME_CPG_BG_SPLIT_NAMES:
        logging.info(f"Processing split {me_cpg_bg_split_name}")
        input_me_cpg_bg_split = input_me_cpg_bg_dataset_dir / "me_cpg_bg" / me_cpg_bg_split_name
        if not input_me_cpg_bg_split.exists():
            logging.warning(f"ME CpG BG split {input_me_cpg_bg_split} does not exist")
            continue

        output_nan_me_cpg_bg_split = output_nan_me_cpg_bg_dataset_dir / "me_cpg_bg" / me_cpg_bg_split_name
        output_nan_me_cpg_bg_split.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory for NaN ME CpG BG split: {output_nan_me_cpg_bg_split}")
        output_non_nan_me_cpg_bg_split = output_non_nan_me_cpg_bg_dataset_dir / "me_cpg_bg" / me_cpg_bg_split_name
        output_non_nan_me_cpg_bg_split.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory for non-NaN ME CpG BG split: {output_non_nan_me_cpg_bg_split}")

        input_me_cpg_bg_split_parquets = sorted(input_me_cpg_bg_split.glob("*.parquet"))

        if FLAGS.debug:
            with tqdm.tqdm(total=len(input_me_cpg_bg_split_parquets), desc="Saving nan and non-nan") as pbar:
                for input_me_cpg_bg_split_parquet in input_me_cpg_bg_split_parquets:
                    output_name = split_nan_in_me_cpg_bg_split_parquet(
                        input_me_cpg_bg_split_parquet,
                        output_nan_me_cpg_bg_split,
                        output_non_nan_me_cpg_bg_split,
                    )
                    pbar.update(1)
                    pbar.set_postfix_str(f"Processed {output_name}")
            return

        num_workers = FLAGS.num_workers
        with mp.Pool(
            num_workers,
            initializer=init_mp_process,
            initargs=(output_nan_me_cpg_bg_split, output_non_nan_me_cpg_bg_split),
        ) as pool:
            with tqdm.tqdm(total=len(input_me_cpg_bg_split_parquets), desc="Saving nan and non-nan") as pbar:
                for output_name in pool.imap_unordered(
                    split_nan_in_me_cpg_bg_split_parquet,
                    input_me_cpg_bg_split_parquets,
                ):
                    pbar.update(1)
                    pbar.set_postfix_str(f"Processed {output_name}")


if __name__ == "__main__":
    app.run(main)
