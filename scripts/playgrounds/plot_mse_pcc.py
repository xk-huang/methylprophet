import os.path as osp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app, flags
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_theme()

flags.DEFINE_string("input_cpg_me_csv", "", "Directory containing the data")
flags.DEFINE_string("output_dir", "", "Directory to save the plots")
flags.DEFINE_alias("i", "input_cpg_me_csv")
flags.DEFINE_alias("o", "output_dir")

FLAGS = flags.FLAGS


def main(_):
    input_cpg_md_csv = FLAGS.input_cpg_me_csv
    input_cpg_md_csv = Path(input_cpg_md_csv)
    data_split_name = input_cpg_md_csv.stem.removeprefix("eval_results-val-")

    output_dir = FLAGS.output_dir
    if output_dir == "":
        output_dir = input_cpg_md_csv.parent / input_cpg_md_csv.stem
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpg_md_df = pd.read_csv(input_cpg_md_csv)
    result_df = cpg_md_df
    result_df = result_df.dropna()

    def _mse_for_group(df):
        mse = mean_squared_error(df["pred_methyl"], df["gt_methyl"])
        return pd.Series({"mse": mse})

    def _mae_for_group(df):
        mae = mean_absolute_error(df["pred_methyl"], df["gt_methyl"])
        return pd.Series({"mae": mae})

    mean_mse_by_sample_id_sklearn = result_df.groupby(["sample_id"]).apply(_mse_for_group)["mse"]
    mean_mse_by_cpg_id_sklearn = result_df.groupby(["cpg_id"]).apply(_mse_for_group)["mse"]
    mean_mae_by_sample_id_sklearn = result_df.groupby(["sample_id"]).apply(_mae_for_group)["mae"]
    mean_mae_by_cpg_id_sklearn = result_df.groupby(["cpg_id"]).apply(_mae_for_group)["mae"]

    def _get_metric_func(metric):
        if metric == "mse":

            def metric_func(df):
                mse = np.square(df["pred_methyl"] - df["gt_methyl"])
                return pd.Series({"mse_loss": mse})

        elif metric == "mae":

            def metric_func(df):
                mae = np.abs(df["pred_methyl"] - df["gt_methyl"])
                return pd.Series({"mae_loss": mae})

        else:
            raise ValueError(f"Invalid metric: {metric}")
        return metric_func

    result_df["mse_loss"] = result_df.apply(_get_metric_func("mse"), axis=1)["mse_loss"]
    result_df["mae_loss"] = result_df.apply(_get_metric_func("mae"), axis=1)["mae_loss"]

    mean_mse_by_sample_id = result_df.groupby(["sample_id"])["mse_loss"].mean()
    mean_mse_by_cpg_id = result_df.groupby(["cpg_id"])["mse_loss"].mean()
    mean_mae_by_sample_id = result_df.groupby(["sample_id"])["mae_loss"].mean()
    mean_mae_by_cpg_id = result_df.groupby(["cpg_id"])["mae_loss"].mean()

    # Compute diff
    mean_mse_by_sample_id_diff = mean_mse_by_sample_id - mean_mse_by_sample_id_sklearn
    mean_mse_by_cpg_id_diff = mean_mse_by_cpg_id - mean_mse_by_cpg_id_sklearn
    mean_mae_by_sample_id_diff = mean_mae_by_sample_id - mean_mae_by_sample_id_sklearn
    mean_mae_by_cpg_id_diff = mean_mae_by_cpg_id - mean_mae_by_cpg_id_sklearn
    print(f"Sum MSE by Sample ID Diff: {mean_mse_by_sample_id_diff.sum()}")
    print(f"Sum MSE by CpG ID Diff: {mean_mse_by_cpg_id_diff.sum()}")
    print(f"Sum MAE by Sample ID Diff: {mean_mae_by_sample_id_diff.sum()}")
    print(f"Sum MAE by CpG ID Diff: {mean_mae_by_cpg_id_diff.sum()}")

    # Plot MSE and MAE
    def box_plot(df, x, y, title, save_dir, title_prefix=""):
        fig, ax = plt.subplots()
        sns.pointplot(df, x=x, y=y, linestyle="")
        ax.set_title(f"{title_prefix}{title}")
        ax.axhline(0, ls="--")

        file_name = title.lower().replace(" ", "_")
        fig.savefig(osp.join(save_dir, f"{file_name}.png"))
        print(f"Save {file_name}.png")

        fig, ax = plt.subplots()
        sorted_df = df.groupby([x])[y].mean().sort_values(ascending=False)
        sns.pointplot(df, x=x, y=y, order=sorted_df.index, linestyle="")
        ax.set_title(f"{title_prefix}{title}")
        ax.axhline(0, ls="--")

        file_name = title.lower().replace(" ", "_")
        fig.savefig(osp.join(save_dir, f"{file_name}-sorted.png"))
        print(f"save {file_name}-sorted.png")

    box_plot(
        result_df,
        "sample_id",
        "mse_loss",
        "MSE by Sample ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )
    box_plot(
        result_df,
        "cpg_id",
        "mse_loss",
        "MSE by CpG ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )
    box_plot(
        result_df,
        "sample_id",
        "mae_loss",
        "MAE by Sample ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )
    box_plot(
        result_df,
        "cpg_id",
        "mae_loss",
        "MAE by CpG ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )
    # Plot PCC
    pcc_by_cpg_id = result_df.groupby(["cpg_id"])[["pred_methyl", "gt_methyl"]].corr().unstack().iloc[:, 1]
    pcc_by_sample_id = result_df.groupby(["sample_id"])[["pred_methyl", "gt_methyl"]].corr().unstack().iloc[:, 1]

    def scatter_plot(df, title, save_dir, title_prefix=""):
        fig, ax = plt.subplots()
        sns.pointplot(df, linestyle="")
        ax.set_title(f"{title_prefix}{title}")
        ax.axhline(0, ls="--")
        file_name = title.lower().replace(" ", "_")
        fig.savefig(osp.join(save_dir, f"{file_name}.png"))
        print(f"Save {file_name}.png")

        sorted_df = df.sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.pointplot(sorted_df, order=sorted_df.index, linestyle="")
        ax.set_title(f"{title_prefix}{title}")
        ax.axhline(0, ls="--")
        reset_index_sorted_df = sorted_df.reset_index(drop=True)

        for percentile in [0, 0.25, 0.5, 0.75]:
            try:
                v_line_value = reset_index_sorted_df[reset_index_sorted_df.ge(percentile)].idxmin()
                ax.axvline(v_line_value, ls="--")
                ax.text(v_line_value, 0, f"{(v_line_value + 1) / len(df) * 100:.0f}%ile", rotation=-90)
            except ValueError:
                pass

        file_name = title.lower().replace(" ", "_")
        save_path = osp.join(save_dir, f"{file_name}-sorted.png")
        fig.savefig(save_path)
        print(f"Save {save_path}")

    scatter_plot(
        pcc_by_cpg_id,
        "PCC by CpG ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )
    scatter_plot(
        pcc_by_sample_id,
        "PCC by Sample ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )

    # NOTE: pcc vs. std
    pred_methyl_std_by_cpg_id = result_df.groupby(["cpg_id"])["pred_methyl"].std()
    gt_methyl_std_by_cpg_id = result_df.groupby(["cpg_id"])["gt_methyl"].std()

    def x_y_scatter_plot(x, y, title, save_dir, title_prefix=""):
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y)
        ax.set_title(f"{title_prefix}{title}")
        # plot x-y line
        ax.plot([-1, 1], [-1, 1], ls="--")
        # Set range
        ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
        ax.set_ylim(y.min() - 0.05, y.max() + 0.05)

        file_name = title.lower().replace(" ", "_")
        save_path = osp.join(save_dir, f"{file_name}.png")
        fig.savefig(save_path)
        print(f"Save {save_path}")

    x_y_scatter_plot(
        pcc_by_cpg_id,
        pred_methyl_std_by_cpg_id,
        "PCC vs Pred Me Std - by CpG ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )
    x_y_scatter_plot(
        pcc_by_cpg_id,
        gt_methyl_std_by_cpg_id,
        "PCC vs GT Me Std - by CpG ID",
        output_dir,
        title_prefix=data_split_name + ": ",
    )


if __name__ == "__main__":
    app.run(main)
