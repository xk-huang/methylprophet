"""
Save the data as parquet format.
The mapping is `sample_name` -> all cpg me values whose index is `cpg_id`
The other keys include `keys-sample_name` and `keys-cpg_id`.
See `read_tdb.py` for how to read the data.

NOTE xk: CpG pos_name has duplicated values. We need cpg id to differentiate them.
Specifically, 393309 CpG sites -> 393292 unique CpG sites.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("data_dir", "data/extracted/encode_wgbs-240802", "Directory of the data")
flags.DEFINE_alias("d", "data_dir")
flags.DEFINE_integer("row_chunk_size", 10000, "Row chunk size for parquet conversion")
flags.DEFINE_string("save_dir", "data/parquet/encode_wgbs-240802", "Directory to save the parquet files")
flags.DEFINE_boolean("overwrite", False, "Overwrite the existing parquet files")

FLAGS = flags.FLAGS


def main(_):
    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)

    me_file_name = "me_rownamesloc.csv"  # 393309 cpg sites X 8578 samples, 57GB
    gene_expr_file_name = "ge.csv"  # 58560 genes X 8578 samples, 4.5GB
    # cancer_type_file_name = "project.csv"  # 8578 samples X 2 {sample name, cancer type} pair, 0.2MB
    # cpg_bg_file_name = "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB

    # cpg_bg_dna_seq_path = data_dir / cpg_bg_file_name
    gene_expr_path = data_dir / gene_expr_file_name
    me_path = data_dir / me_file_name

    # cpg_bg_df = pd.read_csv(cpg_bg_dna_seq_path, nrows=10)
    # cpg_bg_df.rename(columns={"CpG_name": "cpg_id", "CpG_location": "cpg_chr_pos"}, inplace=True)

    me_output_path = save_dir / "me.parquet"
    gene_expr_output_path = save_dir / "gene_expr.parquet"
    if me_output_path.exists():
        if not FLAGS.overwrite:
            logging.info(f"{me_output_path} already exists. Skipping...")
            return
        else:
            logging.info(f"Overwriting {me_output_path}")
    if gene_expr_output_path.exists():
        if not FLAGS.overwrite:
            logging.info(f"{gene_expr_output_path} already exists. Skipping...")
            return
        else:
            logging.info(f"Overwriting {gene_expr_output_path}")

    # cpg_bg_output_path = save_dir / "cpg_bg.parquet"
    gene_expr_df = pd.read_csv(gene_expr_path, nrows=10)

    me_df = pd.read_csv(me_path, nrows=10)

    # logging.info(cpg_bg_dna_seq_path, gene_expr_path, me_path)
    # logging.info(cpg_bg_df.head())
    logging.info(gene_expr_df.head())
    logging.info(me_df.head())

    # cpg_bg_df = pd.read_csv(cpg_bg_dna_seq_path, usecols=["CpG_name", "CpG_location"])

    row_chunk_size = FLAGS.row_chunk_size

    def load_and_save_csv_to_parquet(input_path, output_path):
        # df = pd.read_csv(input_path)
        # table = pa.Table.from_pandas(df)
        # pq.write_table(table, output_path)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Converting {input_path} to {output_path}")
        with pd.read_csv(input_path, chunksize=row_chunk_size) as reader:
            for i, chunk in enumerate(tqdm.tqdm(reader)):
                table = pa.Table.from_pandas(chunk)
                pq.write_table(table, output_path / f"{i:05d}.parquet")

    # load_and_save_csv_to_parquet(cpg_bg_dna_seq_path, cpg_bg_output_path)
    load_and_save_csv_to_parquet(gene_expr_path, gene_expr_output_path)
    load_and_save_csv_to_parquet(me_path, me_output_path)


if __name__ == "__main__":
    app.run(main)
