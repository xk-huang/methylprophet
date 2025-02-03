"""
python scripts/tools/concat_parquet.py \
    --input_parquet_list data/parquet/241213-tcga_array/gene_expr.parquet,data/parquet/241213-tcga_wgbs/gene_expr.parquet \
    --output_dir data/parquet/241213-tcga_array_and_wgbs/ \
    --output_parquet_name gene_expr.parquet \
    --concat_axis 1 \
    --first_col_index

python scripts/tools/data_preprocessing/create_filtered_gene_expr_parquet.py \
    --input_gene_expr_parquet_file "data/parquet/241213-tcga_array_and_wgbs/gene_expr.parquet" \
    --input_grch38_parquet_file "data/parquet/grch38_hg38/grch38.v41.parquet"  \
    --output_dir "data/parquet/241213-tcga_array_and_wgbs/gene_stats" \
    --output_file_name "gene_expr.filtered.parquet" \
    --nofilter_non_protein_coding_gene \
    --plot_only
"""

from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_list("input_parquet_list", None, "List of input parquet files")
flags.mark_flag_as_required("input_parquet_list")

flags.DEFINE_string("output_dir", None, "Output parquet file")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_parquet_name", None, "Output parquet file")
flags.mark_flag_as_required("output_parquet_name")

flags.DEFINE_integer("concat_axis", None, "Axis to concatenate the dataframes")
flags.mark_flag_as_required("concat_axis")

flags.DEFINE_boolean("first_col_index", False, "Whether the first column is the index")


FLAGS = flags.FLAGS


def main(_):
    input_parquet_list = FLAGS.input_parquet_list

    logging.info(f"Input parquet list: {input_parquet_list}")

    # Read all parquet files
    df_list = [pd.read_parquet(parquet_file) for parquet_file in input_parquet_list]
    if FLAGS.first_col_index:
        for df in df_list:
            df.set_index(df.columns[0], inplace=True)

    # Concatenate all dataframes
    concat_axis = FLAGS.concat_axis
    logging.info(f"Concatenating parquet files along axis {concat_axis}")
    df = pd.concat(df_list, axis=concat_axis)

    # Save the concatenated dataframe
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / FLAGS.output_parquet_name
    logging.info(f"Output parquet: {output_parquet}")
    if FLAGS.first_col_index:
        df.reset_index(inplace=True)
    df.to_parquet(output_parquet)

    logging.info(f"Saved concatenated parquet to {output_parquet}")


def check_index_across_df(df_list):
    for i, df in enumerate(df_list):
        if not df.index.equals(df_list[0].index):
            raise ValueError(f"Index of dataframe {i} does not match the index of the first dataframe")


if __name__ == "__main__":
    app.run(main)
