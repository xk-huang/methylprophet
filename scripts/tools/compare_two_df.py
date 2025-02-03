from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_df_a", None, "Path to the parquet file")
flags.mark_flag_as_required("input_df_a")
flags.DEFINE_alias("a", "input_df_a")
flags.DEFINE_string("input_df_b", None, "Path to the parquet file")
flags.mark_flag_as_required("input_df_b")
flags.DEFINE_alias("b", "input_df_b")

flags.DEFINE_string("threshold_column_name", None, "Column name to compare")
flags.DEFINE_float("threshold_lower_bound", 1e-7, "Column name to compare")

flags.DEFINE_bool("ipython", False, "Enable IPython at the end of the script")

FLAGS = flags.FLAGS


def main(_):
    input_df_a = Path(FLAGS.input_df_a)
    input_df_b = Path(FLAGS.input_df_b)
    if not input_df_a.exists():
        logging.error(f"File not found: {input_df_a}")
        return
    if not input_df_b.exists():
        logging.error(f"File not found: {input_df_b}")
        return

    logging.info(f"Reading parquet file: {input_df_a}")
    df_a = read_df(input_df_a)
    logging.info(f"Reading parquet file: {input_df_b}")
    df_b = read_df(input_df_b)

    logging.info(f"shape: df_a {df_a.shape} vs df_b {df_b.shape}")
    if df_a.equals(df_b):
        logging.info("DataFrames are equal")
    else:
        logging.warning("DataFrames are not equal")
        diff_df_a_b = (df_a - df_b).abs()
        max_abs_df_a_b = diff_df_a_b.max()
        logging.warning(f"Max absolute difference between df_a and df_b:\n{max_abs_df_a_b}")

        threshold_column_name = FLAGS.threshold_column_name
        threshold_lower_bound = FLAGS.threshold_lower_bound
        if threshold_column_name:
            diff_df_a_b_sorted = diff_df_a_b.sort_values(by=list(diff_df_a_b.columns), ascending=False)
            logging.warning(
                f"diff:\n{diff_df_a_b_sorted[diff_df_a_b_sorted[threshold_column_name] > threshold_lower_bound]}"
            )

    if FLAGS.ipython:
        from IPython import embed

        embed()


def read_df(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


if __name__ == "__main__":
    app.run(main)
