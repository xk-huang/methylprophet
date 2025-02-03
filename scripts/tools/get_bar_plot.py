from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from absl import app, flags, logging


flags.DEFINE_string("input_df_path", None, "Input file path")
flags.mark_flag_as_required("input_df_path")
flags.DEFINE_alias("i", "input_df_path")

flags.DEFINE_string("select_column_name", None, "Output directory")
flags.DEFINE_string("barplot_x_label", None, "X-axis label for the bar plot")
flags.DEFINE_string("barplot_y_label", None, "Y-axis label for the bar plot")


FLAGS = flags.FLAGS


def main(_):
    input_df_path = Path(FLAGS.input_df_path)

    df = read_df(input_df_path)
    logging.info(f"df:\n{df}")

    select_column_name = FLAGS.select_column_name
    if select_column_name is None:
        select_column_name = df.columns[0]
        logging.warning(f"select_column_name is not provided. Using the first column: {select_column_name}")

    select_column_df = df[select_column_name]
    select_column_df = select_column_df.sort_values(ascending=False)
    # No x-axis label
    sns.barplot(x=select_column_df.index, y=select_column_df)
    plt.xticks([])
    barplot_x_label = FLAGS.barplot_x_label
    if barplot_x_label:
        plt.xlabel(barplot_x_label)
    barplot_y_label = FLAGS.barplot_y_label
    if barplot_y_label:
        plt.ylabel(barplot_y_label)

    logging.info(f"Plotting bar plot for column: {select_column_name}")
    output_plot_path = input_df_path.parent / f"barplot-{input_df_path.stem}.png"
    plt.savefig(output_plot_path)
    logging.info(f"Saved bar plot to {output_plot_path}")


def read_df(input_df_path):
    if input_df_path.suffix == ".csv":
        df = pd.read_csv(input_df_path, index_col=0)
    elif input_df_path.suffix == ".xlsx":
        df = pd.read_excel(input_df_path, index_col=0)
    elif input_df_path.suffix == ".parquet":
        df = pd.read_parquet(input_df_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {input_df_path.suffix}")
    return df


if __name__ == "__main__":
    app.run(main)
