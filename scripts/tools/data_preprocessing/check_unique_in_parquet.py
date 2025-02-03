from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")
flags.DEFINE_alias("i", "input_parquet_file")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_string("index_column_name", None, "Column name to check for unique values")
flags.mark_flag_as_required("index_column_name")

flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")
flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")

FLAGS = flags.FLAGS


def main(argv):
    input_parquet_file = Path(FLAGS.input_parquet_file)
    index_column_name = FLAGS.index_column_name
    logging.info(f"Check unique values in the index column: {index_column_name}")

    input_file_name = input_parquet_file.stem

    output_dir = Path(FLAGS.output_dir) / input_file_name
    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.warning(f"Output directory already exists: {output_dir}")
            return
        else:
            logging.warning(f"Overwriting the output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading parquet file: {input_parquet_file}")
    df = pd.read_parquet(input_parquet_file)
    logging.info(df)

    df_index = df[index_column_name]
    is_duplicated_index = df_index.duplicated()
    if is_duplicated_index.sum() == 0:
        logging.info("No duplicated index")
        with open(output_dir / "no_duplicated_index.txt", "w") as f:
            f.write("No duplicated index")
        return

    logging.info(f"Duplicated index: {df_index[is_duplicated_index]}")
    df_duplicated_index = df_index[df_index.duplicated(False)]
    df_duplicated_index = df_duplicated_index.sort_values()
    df_duplicated_index_path = output_dir / "duplicated_index.csv"
    logging.info(f"Save duplicated index to: {df_duplicated_index_path}")
    df_duplicated_index.to_csv(df_duplicated_index_path)

    metainfo_output_file = output_dir / "metainfo.txt"
    with open(metainfo_output_file, "w") as f:
        shape_str = f"Shape: {df.shape}\n"
        num_duplicated_index_str = f"Number of duplicated index: {is_duplicated_index.sum()}\n"
        num_duplicated_index_percent_str = f"Percentage of duplicated index: {is_duplicated_index.mean() * 100:.2f}%\n"

        f.write(shape_str)
        f.write(num_duplicated_index_str)
        f.write(num_duplicated_index_percent_str)
    logging.info(f"Saved metainfo to: {metainfo_output_file}")


if __name__ == "__main__":
    app.run(main)
