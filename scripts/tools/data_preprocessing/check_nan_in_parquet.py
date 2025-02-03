from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_parquet_file")
flags.DEFINE_alias("i", "input_parquet_file")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")
flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")

FLAGS = flags.FLAGS


def main(argv):
    input_parquet_file = Path(FLAGS.input_parquet_file)
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

    logging.info("Checking for NaN values")
    is_nan_df = df.isna()

    if is_nan_df.sum().sum() == 0:
        logging.info("No NaN values found")
        with open(output_dir / "no_nan.txt", "w") as f:
            f.write("No NaN values found")
        return

    nan_counts_per_column = is_nan_df.sum()
    logging.info(f"NaN counts in each column:\n{pformat(nan_counts_per_column)}")

    # Saving NaN counts per column to a CSV file
    logging.info(f"Saving NaN counts per column to: {output_dir}")
    output_file = output_dir / "nan_count_per_col.csv"
    nan_counts_per_column.to_csv(output_file)
    logging.info(f"Saved NaN per column to: {output_file}")

    # logging NaN raw and column to a CSV file
    df_nan_row = df[is_nan_df.sum(axis=1) > 0]
    logging.info(f"df with NaN rows:\n{df_nan_row}")
    logging.info(f"Saving df with NaN rows to: {output_dir}")
    df_nan_row.to_parquet(output_dir / "df_nan_row.parquet")
    logging.info(f"Saved df with NaN rows to: {output_dir}")

    # Plotting NaN values heatmap
    logging.info("Plotting sorted NaN binary heatmap")

    def sort_binary_dataframe(df):
        # Calculate row and column sums
        row_sums = df.sum(axis=1)
        col_sums = df.sum(axis=0)

        # Sort both rows and columns by their sums in descending order
        sorted_df = df.loc[row_sums.sort_values(ascending=False).index, col_sums.sort_values(ascending=False).index]

        return sorted_df

    is_nan_df_only_nan = is_nan_df[is_nan_df.sum(axis=1) > 0]

    plt.figure(figsize=(15, 10), dpi=300)
    # NOTE: colormap https://matplotlib.org/stable/users/explain/colors/colormaps.html
    plt.imshow(sort_binary_dataframe(is_nan_df_only_nan), cmap="Reds", aspect="auto")
    plt.title("Sroted NaN Binary Heatmap")
    plt.colorbar()
    heatmap_output_file = output_dir / "nan_binary_heatmap-sorted.png"
    plt.savefig(heatmap_output_file)
    logging.info(f"Saved sorted NaN binary heatmap to: {heatmap_output_file}")

    metainfo_output_file = output_dir / "metadata.txt"
    with open(metainfo_output_file, "w") as f:
        shape_str = f"Shape: {df.shape}\n"
        num_nan_str = f"Number of NaN: {nan_counts_per_column.sum()}\n"
        num_nan_percent_str = f"Percentage of NaN: {nan_counts_per_column.sum() / df.size * 100:.2f}%\n"

        num_rows_with_nan_str = f"Number of rows with NaN: {len(is_nan_df_only_nan)}\n"
        num_columns_with_nan_str = f"Number of columns with NaN: {(nan_counts_per_column > 0).sum()}\n"

        f.write(shape_str)
        f.write(num_nan_str)
        f.write(num_nan_percent_str)

        f.write(num_rows_with_nan_str)
        f.write(num_columns_with_nan_str)

    logging.info(f"Saved metadata to: {metainfo_output_file}")


if __name__ == "__main__":
    app.run(main)
