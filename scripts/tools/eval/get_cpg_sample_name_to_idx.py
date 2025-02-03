"""
BASE_DIR=data/parquet/241023-encode_wgbs/metadata/
python misc/get_cpg_sample_name_to_idx.py \
    --input_cpg_chr_pos_df_parquet ${BASE_DIR}/cpg_per_chr_stats/cpg_chr_pos_df.parquet \
    --input_sample_tissue_count_with_idx_csv ${BASE_DIR}/sample_split/sample_tissue_count_with_idx.csv \
    --output_dir ${BASE_DIR}/name_to_idx/
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_cpg_chr_pos_df_parquet", None, "Input CpG chr pos dataframe parquet file.")
flags.mark_flag_as_required("input_cpg_chr_pos_df_parquet")


flags.DEFINE_string("input_sample_tissue_count_with_idx_csv", None, "Input sample tissue count with index csv file.")
flags.mark_flag_as_required("input_sample_tissue_count_with_idx_csv")


flags.DEFINE_string("output_dir", None, "Output directory.")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string(
    "output_cpg_chr_pos_to_idx_file_name", "cpg_chr_pos_to_idx.csv", "Output CpG chr pos dataframe parquet file."
)
flags.DEFINE_string(
    "output_sample_name_to_idx_file_name", "sample_name_to_idx.csv", "Output sample tissue count with index csv file."
)

flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output files.")


def main(_):
    input_cpg_chr_pos_df_parquet = Path(flags.FLAGS.input_cpg_chr_pos_df_parquet)
    input_sample_tissue_count_with_idx_csv = Path(flags.FLAGS.input_sample_tissue_count_with_idx_csv)

    output_dir = Path(flags.FLAGS.output_dir)
    if output_dir.exists():
        if flags.FLAGS.overwrite:
            logging.info(f"Output directory {output_dir} exists and overwrite flag is set.")
        else:
            logging.warning(f"Output directory {output_dir} exists and overwrite flag is not set.")
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cpg_chr_pos_to_idx_file_name = flags.FLAGS.output_cpg_chr_pos_to_idx_file_name
    output_sample_name_to_idx_file_name = flags.FLAGS.output_sample_name_to_idx_file_name
    output_cpg_chr_pos_to_idx_path = output_dir / output_cpg_chr_pos_to_idx_file_name
    output_sample_name_to_idx_path = output_dir / output_sample_name_to_idx_file_name

    cpg_chr_pos_df = pd.read_parquet(input_cpg_chr_pos_df_parquet)
    logging.info(f"Read input CpG chr pos dataframe parquet file: {input_cpg_chr_pos_df_parquet}")

    sample_tissue_count_with_idx = pd.read_csv(input_sample_tissue_count_with_idx_csv)
    logging.info(f"Read input sample tissue count with index csv file: {input_sample_tissue_count_with_idx_csv}")

    cpg_chr_pos_df = cpg_chr_pos_df["chr"] + "_" + cpg_chr_pos_df["pos"].astype(str)
    cpg_chr_pos_df = cpg_chr_pos_df.reset_index()
    cpg_chr_pos_df.columns = ["cpg_idx", "cpg_chr_pos"]
    cpg_chr_pos_df = cpg_chr_pos_df.set_index("cpg_chr_pos")

    sample_tissue_count_with_idx = sample_tissue_count_with_idx[["sample_idx", "sample_name"]]
    sample_tissue_count_with_idx = sample_tissue_count_with_idx.set_index("sample_name")

    cpg_chr_pos_df.to_csv(output_cpg_chr_pos_to_idx_path, index=True)
    logging.info(f"Saved output CpG chr pos dataframe parquet file: {output_cpg_chr_pos_to_idx_path}")

    sample_tissue_count_with_idx.to_csv(output_sample_name_to_idx_path, index=True)
    logging.info(f"Saved output sample tissue count with index csv file: {output_sample_name_to_idx_path}")


if __name__ == "__main__":
    app.run(main)
