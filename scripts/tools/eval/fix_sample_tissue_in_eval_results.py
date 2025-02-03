"""
This fix is temporary, when process data from scratch, no need to use this.

input_dir=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/eval_results-test.parquet
correct_sample_idx_file=data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv
wrong_sample_idx_file=data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv
output_dir=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/fix_tissue_map-eval_results-test.parquet

python scripts/tools/eval/fix_sample_tissue_in_eval_results.py \
    --input_eval_result_dir=$input_dir \
    --output_eval_result_dir=$output_dir \
    --correct_sample_idx_csv=$correct_sample_idx_file \
    --wrong_sample_idx_csv=$wrong_sample_idx_file


input_dir=data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/train_cpg-train_sample.parquet
correct_sample_idx_file=data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv
wrong_sample_idx_file=data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv
output_dir=data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/fix_tissue_map-train_cpg-train_sample.parquet

python scripts/tools/eval/fix_sample_tissue_in_eval_results.py \
    --input_eval_result_dir=$input_dir \
    --output_eval_result_dir=$output_dir \
    --correct_sample_idx_csv=$correct_sample_idx_file \
    --wrong_sample_idx_csv=$wrong_sample_idx_file --num_processes=20
"""

import multiprocessing as mp
import shutil
from functools import partial
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_eval_result_dir", None, "Directory containing eval results")
flags.DEFINE_string("output_eval_result_dir", None, "Directory to write fixed eval results")

flags.DEFINE_string("correct_sample_idx_csv", None, "CSV file containing correct sample indices")
flags.DEFINE_string("wrong_sample_idx_csv", None, "CSV file containing wrong sample indices")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite existing files")
flags.DEFINE_integer("num_processes", 1, "Number of processes to use for multiprocessing")


def process_file(eval_result_file, output_eval_result_dir, merged_sample_idx_mapping):
    # logging.info(f"Processing {eval_result_file}")

    eval_result = pd.read_parquet(eval_result_file)
    # change wrong "sample_idx" to the correct ones using merged_sample_idx
    eval_result["correct_sample_idx"] = eval_result["sample_idx"].map(merged_sample_idx_mapping["sample_idx_correct"])
    eval_result["correct_tissue_idx"] = eval_result["sample_idx"].map(merged_sample_idx_mapping["tissue_idx_correct"])
    eval_result["correct_tissue_name"] = eval_result["sample_idx"].map(
        merged_sample_idx_mapping["tissue_name_correct"]
    )
    eval_result["correct_sample_name"] = eval_result["sample_idx"].map(merged_sample_idx_mapping["sample_name"])

    output_eval_result_file = output_eval_result_dir / eval_result_file.name
    eval_result.to_parquet(output_eval_result_file)

    # logging.info(f"Saved to {output_eval_result_file}")
    return str(eval_result_file)


def main(_):
    input_eval_result_dir = Path(flags.FLAGS.input_eval_result_dir)
    output_eval_result_dir = Path(flags.FLAGS.output_eval_result_dir)
    if output_eval_result_dir.exists():
        if flags.FLAGS.overwrite:
            shutil.rmtree(output_eval_result_dir)
            logging.info(f"Removed existing directory {output_eval_result_dir}")
        else:
            logging.warning(f"Directory {output_eval_result_dir} already exists")
            return
    output_eval_result_dir.mkdir(parents=True)

    correct_sample_idx_csv = Path(flags.FLAGS.correct_sample_idx_csv)
    wrong_sample_idx_csv = Path(flags.FLAGS.wrong_sample_idx_csv)

    correct_sample_idx = pd.read_csv(correct_sample_idx_csv)
    wrong_sample_idx = pd.read_csv(wrong_sample_idx_csv)

    correct_sample_idx = correct_sample_idx[["tissue_name", "tissue_idx", "sample_name", "sample_idx"]]
    correct_sample_idx.rename(
        columns={
            "sample_idx": "sample_idx_correct",
            "tissue_idx": "tissue_idx_correct",
            "tissue_name": "tissue_name_correct",
        },
        inplace=True,
    )
    wrong_sample_idx = wrong_sample_idx[["tissue_name", "tissue_idx", "sample_name", "sample_idx"]]
    wrong_sample_idx.rename(
        columns={
            "sample_idx": "sample_idx_wrong",
            "tissue_idx": "tissue_idx_wrong",
            "tissue_name": "tissue_name_wrong",
        },
        inplace=True,
    )
    correct_sample_idx = correct_sample_idx.sort_values("sample_name").reset_index(drop=True)
    wrong_sample_idx = wrong_sample_idx.sort_values("sample_name").reset_index(drop=True)
    if not correct_sample_idx.equals(wrong_sample_idx):
        logging.warning("correct_sample_idx and wrong_sample_idx are not equal")

    merged_sample_idx = pd.merge(correct_sample_idx, wrong_sample_idx, on="sample_name")
    merged_sample_idx = merged_sample_idx[
        [
            "sample_name",
            "sample_idx_correct",
            "sample_idx_wrong",
            "tissue_name_correct",
            "tissue_name_wrong",
            "tissue_idx_correct",
            "tissue_idx_wrong",
        ]
    ]
    merged_sample_idx_mapping = merged_sample_idx.set_index("sample_idx_wrong")

    # Get list of files to process
    eval_result_files = sorted(input_eval_result_dir.glob("*.parquet"))

    # Set up multiprocessing
    num_processes = flags.FLAGS.num_processes or mp.cpu_count()
    pool = mp.Pool(processes=num_processes)

    # Create partial function with fixed arguments
    process_file_partial = partial(
        process_file,
        output_eval_result_dir=output_eval_result_dir,
        merged_sample_idx_mapping=merged_sample_idx_mapping,
    )

    # Process files in parallel with progress bar
    for _ in tqdm.tqdm(
        pool.imap_unordered(process_file_partial, eval_result_files),
        total=len(eval_result_files),
        desc="Processing files",
    ):
        pass

    pool.close()
    pool.join()


if __name__ == "__main__":
    app.run(main)
