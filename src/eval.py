import datetime
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from absl import app, flags, logging


GROUP_KEYS = ["sample_idx", "cpg_idx"]
BACKENDS = ["pandas", "torchmetrics_cpu", "torchmetrics_cuda"]


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(
            f"Elapsed time for function ({func.__name__}): {datetime.timedelta(seconds=end_time - start_time)}"
        )
        return result

    return wrapper


@timer
def compute_pcc_by_group(df, group_key, backend="pandas", num_group_per_batch=100):
    if group_key not in GROUP_KEYS:
        raise ValueError(f"group_key ({group_key}) must be in {GROUP_KEYS}")

    if backend not in BACKENDS:
        raise ValueError(f"backend ({backend}) must be in {BACKENDS}")

    if backend == "pandas":
        pcc_by_group = df.groupby(group_key)[["pred_methyl", "gt_methyl"]].corr().unstack().iloc[:, 1]
    elif backend == "torchmetrics_cpu" or backend == "torchmetrics_cuda":
        df = df[[group_key, "pred_methyl", "gt_methyl"]].copy()
        df.sort_values(by=group_key, inplace=True)
        # NOTE: keep the order of the group_key
        group_counter = df[group_key].value_counts().sort_index()
        group_series_index = group_counter.index
        num_pairs_in_group = group_counter.values[0]
        num_groups = len(group_series_index)

        pred_methyl = df["pred_methyl"].values
        gt_methtyl = df["gt_methyl"].values

        logging.info(
            f"num_groups: {num_groups}, num_group_per_batch: {num_group_per_batch}, num_pairs_in_group: {num_pairs_in_group}"
        )
        pcc_by_group_list = []
        for group_start_idx in range(0, num_groups, num_group_per_batch):
            group_end_idx = min(group_start_idx + num_group_per_batch, num_groups)
            pair_start_idx = group_start_idx * num_pairs_in_group
            pair_end_idx = (group_start_idx + num_group_per_batch) * num_pairs_in_group
            pair_end_idx = min(pair_end_idx, len(df))
            pcc_by_group = compute_batch_pcc_by_group(
                backend,
                group_series_index[group_start_idx:group_end_idx],
                num_pairs_in_group,
                pred_methyl[pair_start_idx:pair_end_idx],
                gt_methtyl[pair_start_idx:pair_end_idx],
            )
            pcc_by_group_list.append(pcc_by_group)
        pcc_by_group = pd.concat(pcc_by_group_list)
        del pcc_by_group_list

    pcc_by_group.name = "Me PCC"
    pcc_by_group.index.name = f"By {group_key}"

    return pcc_by_group


def compute_batch_pcc_by_group(backend, group_series_index, num_pairs_in_group, pred_methyl, gt_methtyl):
    # NOTE: torchmetrics.functional.pearson_corrcoef() requires the input shape to be (num_samples, num_batch)
    pred_methyl = torch.from_numpy(pred_methyl).view(-1, num_pairs_in_group).transpose(0, 1)
    gt_methtyl = torch.from_numpy(gt_methtyl).view(-1, num_pairs_in_group).transpose(0, 1)

    if backend == "torchmetrics_cpu":
        pred_methyl = pred_methyl.cpu()
        gt_methtyl = gt_methtyl.cpu()
    elif backend == "torchmetrics_cuda":
        pred_methyl = pred_methyl.cuda()
        gt_methtyl = gt_methtyl.cuda()
    else:
        raise ValueError(f"Invalid backend: {backend} for torchmetrics")

    pcc_by_group = torchmetrics.functional.pearson_corrcoef(pred_methyl, gt_methtyl)
    pcc_by_group = pcc_by_group.cpu().numpy()
    pcc_by_group = pd.Series(pcc_by_group, index=group_series_index)
    return pcc_by_group


@timer
def compute_std_by_group(df, group_key, output_name_prefix=None):
    std_by_group = df.groupby(group_key).std()
    std_by_group.columns = [f"{output_name_prefix} Me Std" if output_name_prefix is not None else "Me Std"]
    std_by_group.index.name = f"By {group_key}"
    return std_by_group


@timer
def plot_scatter_x_y_with_line(
    data,
    x,
    y,
    output_dir,
    output_plot_name,
    xlim_bottom=None,
    xlim_top=None,
    ylim_bottom=None,
    ylim_top=None,
):
    fig, ax = plt.subplots()
    logging.debug(f"Plot scatter: {output_plot_name}")
    sns.scatterplot(data=data, x=x, y=y, s=1, alpha=0.5)
    plot_post_process_and_save(
        fig=fig,
        ax=ax,
        output_dir=output_dir,
        output_plot_name=output_plot_name,
        xlim_bottom=xlim_bottom,
        xlim_top=xlim_top,
        ylim_bottom=ylim_bottom,
        ylim_top=ylim_top,
        plot_type="scatter",
    )
    return fig, ax


@timer
def plot_boxenplot(
    output_dir,
    pcc_by_sample_id,
    output_plot_name,
    ylim_bottom=None,
    ylim_top=None,
):
    fig, ax = plt.subplots()
    logging.debug(f"Plot boxenplot: {output_plot_name}")
    sns.boxenplot(pcc_by_sample_id, fill=False)
    plot_post_process_and_save(
        fig=fig,
        ax=ax,
        output_dir=output_dir,
        output_plot_name=output_plot_name,
        ylim_bottom=ylim_bottom,
        ylim_top=ylim_top,
        plot_type="boxenplot",
    )
    return fig, ax


@timer
def plot_boxplot(
    output_dir,
    pcc_by_sample_id,
    output_plot_name,
    ylim_bottom=None,
    ylim_top=None,
):
    fig, ax = plt.subplots()
    logging.debug(f"Plot boxplot: {output_plot_name}")
    try:
        sns.boxplot(pcc_by_sample_id, fill=False)
    except Exception as e:
        logging.warning(f"Failed to plot boxplot: {output_plot_name}. {e}\n{pcc_by_sample_id}")

    plot_post_process_and_save(
        fig=fig,
        ax=ax,
        output_dir=output_dir,
        output_plot_name=output_plot_name,
        ylim_bottom=ylim_bottom,
        ylim_top=ylim_top,
        plot_type="boxplot",
    )

    return fig, ax


@timer
def plot_violinplot(
    output_dir,
    pcc_by_sample_id,
    output_plot_name,
    ylim_bottom=None,
    ylim_top=None,
):
    fig, ax = plt.subplots()
    logging.debug(f"Plot violinplot: {output_plot_name}")
    sns.violinplot(pcc_by_sample_id, inner="quart", fill=False)
    plot_post_process_and_save(
        fig=fig,
        ax=ax,
        output_dir=output_dir,
        output_plot_name=output_plot_name,
        ylim_bottom=ylim_bottom,
        ylim_top=ylim_top,
        plot_type="violinplot",
    )
    return fig, ax


def plot_post_process_and_save(
    *,
    fig,
    ax,
    output_dir,
    output_plot_name,
    xlim_bottom=None,
    xlim_top=None,
    ylim_bottom=None,
    ylim_top=None,
    plot_type=None,
):
    # title
    title_name = output_plot_name.replace("_", " ").title()
    ax.set_title(title_name)

    # set x-axis range
    if xlim_bottom is not None and xlim_top is not None:
        logging.debug(f"Set x-axis range: {xlim_bottom} to {xlim_top}")
        ax.set_xlim(xlim_bottom, xlim_top)

    # set y-axis range
    if ylim_bottom is not None and ylim_top is not None:
        logging.debug(f"Set y-axis range: {ylim_bottom} to {ylim_top}")
        ax.set_ylim(ylim_bottom, ylim_top)

    output_plot_name = f"{output_plot_name}-{plot_type}" if plot_type else output_plot_name
    output_plot_path = output_dir / f"{output_plot_name}.png"
    fig.savefig(output_plot_path)
    logging.info(f"Output plots to {output_plot_path}")


def get_numpy_image_from_fig(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8").reshape(height, width, 4)
    return image


def check_all_value_same(series):
    return len(series.unique()) == 1


def create_group_counts(df):
    pair_counts_by_sample_id = df["sample_idx"].value_counts()
    pair_counts_by_cpg_id = df["cpg_idx"].value_counts()
    return pair_counts_by_sample_id, pair_counts_by_cpg_id


def check_group_counts(pair_counts_by_sample_id, pair_counts_by_cpg_id):
    check_flag = True
    if not check_all_value_same(pair_counts_by_sample_id):
        logging.warning(f"The number of pairs is not the same by sample_ids. {pair_counts_by_sample_id}")
        check_flag = False
    if not check_all_value_same(pair_counts_by_cpg_id):
        logging.warning(f"The number of pairs is not the same by cpg_ids. {pair_counts_by_cpg_id}")
        check_flag = False
    return check_flag


class MethylEval:
    def __init__(self, df, output_dir, num_batches_per_group=10, backend=None):
        logging.info(
            f"Init MethylEval: df shape: {df.shape}, output_dir: {output_dir}, num_batches_per_group: {num_batches_per_group}"
        )
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_batches_per_group = num_batches_per_group

        self.pcc_by_sample_id = None
        self.pcc_by_cpg_id = None
        self.pred_methyl_std_by_cpg_id = None
        self.gt_methtyl_std_by_cpg_id = None
        self.backend = backend

        self.compute()

    def compute(self):
        logging.info("Start computing PCC and Std")
        df = self.df
        num_batches_per_group = self.num_batches_per_group
        pair_counts_by_sample_id, pair_counts_by_cpg_id = create_group_counts(df)

        backend = self.backend
        if backend is None:
            if check_group_counts(pair_counts_by_sample_id, pair_counts_by_cpg_id) is True:
                logging.debug("The number of pairs per group is the same. Using torchmetrics backend.")
                if torch.cuda.is_available():
                    backend = "torchmetrics_cuda"
                else:
                    backend = "torchmetrics_cpu"
            else:
                logging.warning(
                    "The number of pairs per group is not the same. Using pandas backend which may be slow for large scale data."
                )
                backend = "pandas"
        logging.info(f"Using eval backend: {backend}")

        # Compute PCC
        self.pcc_by_sample_id = compute_pcc_by_group(
            df,
            "sample_idx",
            backend,
            num_group_per_batch=max(1, len(pair_counts_by_sample_id) // num_batches_per_group),
        )
        self.pcc_by_cpg_id = compute_pcc_by_group(
            df,
            "cpg_idx",
            backend,
            num_group_per_batch=max(1, len(pair_counts_by_cpg_id) // num_batches_per_group),
        )

        # Compute Std
        self.pred_methyl_std_by_cpg_id = compute_std_by_group(df[["pred_methyl", "cpg_idx"]], "cpg_idx", "Prediction")
        self.gt_methtyl_std_by_cpg_id = compute_std_by_group(df[["gt_methyl", "cpg_idx"]], "cpg_idx", "Ground Truth")

        # Compute loss
        self.mae_loss_mean = (df["pred_methyl"] - df["gt_methyl"]).abs().mean()
        self.mse_loss_mean = ((df["pred_methyl"] - df["gt_methyl"]) ** 2).mean()

        # Delete df to save memory
        del df
        self.df = None

    def eval_results_to_log_dict(self):
        if self.pcc_by_sample_id is None or self.pcc_by_cpg_id is None:
            raise ValueError("PCC is not computed yet. Run compute() first.")

        log_dict = {
            "mse_loss_per_point": self.mse_loss_mean,
            "mae_loss_per_point": self.mae_loss_mean,
            "pcc_by_cpg_id-mean": self.pcc_by_cpg_id.mean(),
            "pcc_by_sample_id-mean": self.pcc_by_sample_id.mean(),
            "pcc_by_cpg_id-median": self.pcc_by_cpg_id.median(),
            "pcc_by_sample_id-median": self.pcc_by_sample_id.median(),
        }
        return log_dict

    def eval_results_to_plot(self, prefix_name=""):
        output_dir = self.output_dir
        pcc_by_sample_id = self.pcc_by_sample_id
        pcc_by_cpg_id = self.pcc_by_cpg_id
        pred_methyl_std_by_cpg_id = self.pred_methyl_std_by_cpg_id
        gt_methtyl_std_by_cpg_id = self.gt_methtyl_std_by_cpg_id

        if pcc_by_sample_id is None or pcc_by_cpg_id is None:
            raise ValueError("PCC is not computed yet. Run compute() first.")
        if pred_methyl_std_by_cpg_id is None or gt_methtyl_std_by_cpg_id is None:
            raise ValueError("Std is not computed yet. Run compute() first.")

        numpy_image_dict = {}
        # Plot PCC by Sample ID
        output_plot_name = prefix_name + "-" + "me_pcc_by_sample_id"
        fig, ax = plot_boxenplot(output_dir, pcc_by_sample_id, output_plot_name, ylim_bottom=0, ylim_top=1)
        numpy_image_dict[f"{output_plot_name}-boxenplot"] = get_numpy_image_from_fig(fig)

        fig, ax = plot_boxplot(output_dir, pcc_by_sample_id, output_plot_name, ylim_bottom=0, ylim_top=1)
        numpy_image_dict[f"{output_plot_name}-boxplot"] = get_numpy_image_from_fig(fig)

        fig, ax = plot_violinplot(output_dir, pcc_by_sample_id, output_plot_name, ylim_bottom=0, ylim_top=1)
        numpy_image_dict[f"{output_plot_name}-violinplot"] = get_numpy_image_from_fig(fig)

        # Plot PCC by CpG ID
        output_plot_name = prefix_name + "-" + "me_pcc_by_cpg_id"
        fig, ax = plot_boxenplot(output_dir, pcc_by_cpg_id, output_plot_name, ylim_bottom=-1, ylim_top=1)
        numpy_image_dict[f"{output_plot_name}-boxenplot"] = get_numpy_image_from_fig(fig)

        fig, ax = plot_boxplot(output_dir, pcc_by_cpg_id, output_plot_name, ylim_bottom=-1, ylim_top=1)
        numpy_image_dict[f"{output_plot_name}-boxplot"] = get_numpy_image_from_fig(fig)

        fig, ax = plot_violinplot(output_dir, pcc_by_cpg_id, output_plot_name, ylim_bottom=-1, ylim_top=1)
        numpy_image_dict[f"{output_plot_name}-violinplot"] = get_numpy_image_from_fig(fig)

        # Plot PCC vs {Prediction, Ground Truth} Std by CpG ID
        pcc_pred_std_by_cpg_id = pd.concat([pcc_by_cpg_id, pred_methyl_std_by_cpg_id], axis=1)
        pcc_gt_std_by_cpg_id = pd.concat([pcc_by_cpg_id, gt_methtyl_std_by_cpg_id], axis=1)

        x_name, y_name = pcc_pred_std_by_cpg_id.columns
        fig, ax = plot_scatter_x_y_with_line(
            data=pcc_pred_std_by_cpg_id,
            x=x_name,
            y=y_name,
            output_dir=self.output_dir,
            output_plot_name=prefix_name + "-" + "pcc_pred_std_by_cpg_id",
            xlim_bottom=-1,
            xlim_top=1,
            ylim_bottom=-0.05,
            ylim_top=0.8,
        )
        numpy_image_dict["pcc_pred_std_by_cpg_id-scatter"] = get_numpy_image_from_fig(fig)

        x_name, y_name = pcc_gt_std_by_cpg_id.columns
        fig, ax = plot_scatter_x_y_with_line(
            data=pcc_gt_std_by_cpg_id,
            x=x_name,
            y=y_name,
            output_dir=output_dir,
            output_plot_name=prefix_name + "-" + "pcc_gt_std_by_cpg_id",
            xlim_bottom=-1,
            xlim_top=1,
            ylim_bottom=-0.05,
            ylim_top=0.8,
        )
        numpy_image_dict["pcc_gt_std_by_cpg_id-scatter"] = get_numpy_image_from_fig(fig)
        return numpy_image_dict
