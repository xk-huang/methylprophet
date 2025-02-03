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

flags.DEFINE_string("data_dir", None, "Directory of the data")
flags.mark_flag_as_required("data_dir")
flags.DEFINE_string("file_name", None, "File name")
flags.mark_flag_as_required("file_name")
# either "me_rownamesloc.csv"
# ENCODE WGBS: 28,301,739 cpg sites X 95 samples, 46GB
# TCGA ARRAY: 393,309 cpg sites X 8578 samples, 57GB
# or "ge.csv"
# ENCODE WGBS: 55503 genes X 96 samples, 48MB
# TCGA ARRAY: 58560 genes X 8578 samples, 4.5GB

flags.DEFINE_integer("row_chunk_size", 10000, "Row chunk size for parquet conversion")
flags.DEFINE_string("sep", ",", "Separator for csv files")

flags.DEFINE_string("output_dir", None, "Directory to save the parquet files")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output file name")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_boolean("overwrite", False, "Overwrite the existing parquet files")

FLAGS = flags.FLAGS


def main(_):
    data_dir = Path(FLAGS.data_dir)
    output_dir = Path(FLAGS.output_dir)

    file_name = FLAGS.file_name
    file_path = data_dir / file_name

    output_file_name = FLAGS.output_file_name
    output_file_path = output_dir / output_file_name

    if output_file_path.exists():
        if not FLAGS.overwrite:
            logging.warning(f"{output_file_path} already exists. Skipping...")
            return
        else:
            logging.warning(f"Overwriting {output_file_path}")

    csv_sep = FLAGS.sep
    short_df = pd.read_csv(file_path, nrows=10, sep=csv_sep)
    logging.info(f"View the head: {short_df.head()}")

    def count_rows(file_path):
        count = 0
        with open(file_path, "rb") as f:
            # Count newlines, skip header
            count = sum(1 for _ in f) - 1
        return count

    logging.info(f"Counting rows in {file_path}")
    num_rows = count_rows(file_path)
    logging.info(f"Counting rows in {file_path}: {num_rows}")

    def load_and_save_csv_to_parquet(input_path, output_path, row_chunk_size, num_rows):
        # df = pd.read_csv(input_path)
        # table = pa.Table.from_pandas(df)
        # pq.write_table(table, output_path)

        Path(output_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Converting {input_path} to {output_path}")
        with pd.read_csv(input_path, chunksize=row_chunk_size, sep=csv_sep) as reader:
            pbar = tqdm.tqdm(total=num_rows)
            for i, chunk in enumerate(reader):
                table = pa.Table.from_pandas(chunk)
                pq.write_table(table, output_path / f"{i:05d}.parquet")
                pbar.update(row_chunk_size)
        logging.info(f"Saved {output_path}")

    row_chunk_size = FLAGS.row_chunk_size
    load_and_save_csv_to_parquet(file_path, output_file_path, row_chunk_size, num_rows)


if __name__ == "__main__":
    app.run(main)
