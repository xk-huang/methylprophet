"""
python scripts/tools/data_preprocessing/merge_gene_expr_parquet.py \
    --input_gene_expr_parquet_list data/parquet/241231-tcga_array/gene_expr.parquet,data/parquet/241231-tcga_epic/gene_expr.parquet,data/parquet/241231-tcga_wgbs/gene_expr.parquet \
    --output_dir data/parquet/241231-tcga/ \
    --overwrite

python scripts/tools/data_preprocessing/check_nan_in_parquet.py \
    --input_parquet_file data/parquet/241231-tcga/gene_expr.parquet \
    --output_dir data/parquet/241231-tcga/metadata/check_nan

python scripts/tools/data_preprocessing/check_unique_in_parquet.py \
    --input_parquet_file data/parquet/241231-tcga/gene_expr.parquet \
    --index_column_name 'Unnamed: 0' \
    --output_dir data/parquet/241231-tcga/metadata/check_unique
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_list("input_gene_expr_parquet_list", None, "List of input gene expression parquet files")
flags.mark_flag_as_required("input_gene_expr_parquet_list")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output file if it already exists")
flags.DEFINE_integer("row_chunk_size", 10000, "Chunk size for reading parquet files")

FLAGS = flags.FLAGS


def prepare_output_dir(output_dir: str, overwrite: bool = False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.warning(f"Output directory {output_dir} already exists, will overwrite")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory {output_dir} already exists, will not overwrite")
            exit()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(_):
    input_gene_expr_parquet_list = FLAGS.input_gene_expr_parquet_list

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_gene_expr_parquet = output_dir / "gene_expr.parquet"
    output_gene_expr_parquet = prepare_output_dir(output_gene_expr_parquet, FLAGS.overwrite)

    gene_expr_df_list = []
    for input_gene_expr_parquet in input_gene_expr_parquet_list:
        gene_expr_df = pd.read_parquet(input_gene_expr_parquet)
        if gene_expr_df.columns[0] != "Unnamed: 0":
            raise ValueError(
                f"First column of gene expression parquet file {input_gene_expr_parquet} is not 'Unnamed: 0'"
            )
        gene_expr_df = gene_expr_df.rename(columns={"Unnamed: 0": "gene_name_id"})
        gene_expr_df = gene_expr_df.set_index("gene_name_id")
        gene_expr_df_list.append(gene_expr_df)

    # compare gene names
    first_gene_expr_df = gene_expr_df_list[0]
    for idx, second_gene_expr_df in enumerate(gene_expr_df_list[1:]):
        if not first_gene_expr_df.index.equals(second_gene_expr_df.index):
            raise ValueError(
                f"Gene names in gene expression parquet files do not, {input_gene_expr_parquet_list[0]} vs {input_gene_expr_parquet_list[idx+1]}"
            )

    # merged columns sample idx to gene expr df idx
    merged_column_sample_idx_to_gene_expr_df_idx = []
    for gene_expr_df_idx in range(len(gene_expr_df_list)):
        merged_column_sample_idx_to_gene_expr_df_idx.extend(
            [gene_expr_df_idx] * gene_expr_df_list[gene_expr_df_idx].shape[1]
        )
    merged_column_sample_idx_to_gene_expr_df_idx = pd.Series(merged_column_sample_idx_to_gene_expr_df_idx)

    merged_gene_expr_df = pd.concat(gene_expr_df_list, axis=1)
    if len(merged_gene_expr_df.columns) != len(merged_column_sample_idx_to_gene_expr_df_idx):
        raise ValueError("Number of columns in merged gene expression dataframe does not match number of samples")

    # columns are samples, their name should be unique
    if len(merged_gene_expr_df.columns) != len(set(merged_gene_expr_df.columns)):
        # find duplicated columns and their index
        duplicated_columns = merged_gene_expr_df.columns[merged_gene_expr_df.columns.duplicated()]
        for duplicated_column in duplicated_columns:
            duplicated_sample_gene_expr_df = merged_gene_expr_df[duplicated_column]

            # check that each column value is the same
            duplicated_column_idx = np.where(merged_gene_expr_df.columns.get_loc(duplicated_column))[0]
            df_idx = merged_column_sample_idx_to_gene_expr_df_idx[duplicated_column_idx].tolist()
            logging.warning(
                f"Sample {duplicated_column} is duplicated at index {duplicated_column_idx}, from {df_idx}"
            )

            first_gene_expr = duplicated_sample_gene_expr_df.iloc[:, 0]
            for idx in range(1, duplicated_sample_gene_expr_df.shape[1]):
                if not np.allclose(first_gene_expr, duplicated_sample_gene_expr_df.iloc[:, idx]):
                    raise ValueError(f"Sample {duplicated_column} has different values in different columns")
        duplicated_columns = list(duplicated_columns)
        duplicated_columns_output_path = output_dir / "duplicated_sample_names_after_merge.txt"
        with open(duplicated_columns_output_path, "w") as f:
            f.write("\n".join(duplicated_columns))
        logging.info(f"Found duplicated columns, saved to {duplicated_columns_output_path}")

    # remove duplicated columns
    merged_gene_expr_df = merged_gene_expr_df.loc[:, ~merged_gene_expr_df.columns.duplicated()]
    # recover `gene_name_id` index to `Unnamed: 0` column
    merged_gene_expr_df = merged_gene_expr_df.reset_index().rename(columns={"gene_name_id": "Unnamed: 0"})
    # write chunk
    num_rows = merged_gene_expr_df.shape[0]
    row_chunk_size = FLAGS.row_chunk_size
    for chunk_idx, idx in enumerate(range(0, num_rows, row_chunk_size)):
        chunk = merged_gene_expr_df.iloc[idx : idx + FLAGS.row_chunk_size]
        output_gene_expr_parquet_file = output_gene_expr_parquet / f"{chunk_idx:05d}.parquet"
        chunk.to_parquet(output_gene_expr_parquet_file)

    logging.info(f"Saved merged gene expression dataframe to {output_gene_expr_parquet}")


if __name__ == "__main__":
    app.run(main)
