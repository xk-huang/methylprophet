"""
python scripts/tools/eval/log_metrics_to_wandb.py \
    --input_eval_dir outputs/eval/eval-encode_wgbs-bs_512-64xl40s-aws/ \
    --exp_name "250125-methylfoundation-encode" \
    --job_name "eval-encode_wgbs-bs_512-64xl40s-aws"
"""

import json
import os
import pprint
from pathlib import Path

import dotenv
import pandas as pd
from absl import app, flags, logging

import wandb


dotenv.load_dotenv(override=True)
logging.info(os.getenv("WANDB_MODE"))

flags.DEFINE_string(
    "input_eval_dir",
    "outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/",
    "Path to the sample tissue count csv",
)
flags.DEFINE_string("exp_name", "test_wandb_upload", "Experiment name")
flags.DEFINE_string("job_name", "debug_upload_wandb-eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws", "Job name")

FLAGS = flags.FLAGS


def main(_):
    input_eval_dir = Path(FLAGS.input_eval_dir)
    me_df = pd.read_parquet(
        input_eval_dir / "eval_results-test.parquet",
        columns=["group_idx", "pred_methyl", "gt_methyl"],
    )
    me_df["mae_loss"] = (me_df["pred_methyl"] - me_df["gt_methyl"]).abs()
    me_df["mse_loss"] = (me_df["pred_methyl"] - me_df["gt_methyl"]) ** 2

    group_idx_name_mapping = None
    group_mapping_path = input_eval_dir / "group_idx_name_mapping-test.json"
    with open(group_mapping_path) as f:
        group_idx_name_mapping = json.load(f)
    group_idx_name_mapping = {int(k): Path(v).name for k, v in group_idx_name_mapping.items()}

    log_dict = {"trainer/global_step": 0}
    stage_name = "eval-test"

    for group_idx, group_name in group_idx_name_mapping.items():
        loss_df = me_df[["group_idx", "mae_loss", "mse_loss"]][me_df["group_idx"] == group_idx]
        mae_loss_median = loss_df["mae_loss"].mean()
        mse_loss_median = loss_df["mse_loss"].mean()

        data_split_name = group_name
        eval_metric_name = "mse_loss_per_point"
        log_name = f"{stage_name}/{data_split_name}/{eval_metric_name}"
        log_dict[log_name] = mse_loss_median

        eval_metric_name = "mae_loss_per_point"
        log_name = f"{stage_name}/{data_split_name}/{eval_metric_name}"
        log_dict[log_name] = mae_loss_median

    pcc_me_std_prefix_list = ["pcc_by_cpg_id_me_std", "pcc_by_sample_id"]
    pcc_metric_name_list = ["pcc_by_cpg_id-median", "pcc_by_sample_id-median"]
    for pcc_me_std_prefix, pcc_metric_name in zip(pcc_me_std_prefix_list, pcc_metric_name_list):
        for group_idx, group_name in group_idx_name_mapping.items():
            pcc_file_path = input_eval_dir / "pcc_and_me_std" / f"{pcc_me_std_prefix}-{group_name}.csv"
            pcc_df = pd.read_csv(pcc_file_path)

            pcc_median = pcc_df["pcc"].median()
            data_split_name = group_name
            eval_metric_name = pcc_metric_name
            log_name = f"{stage_name}/{data_split_name}/{eval_metric_name}"
            log_dict[log_name] = pcc_median

    run = wandb.init(
        name=FLAGS.job_name,
        group=FLAGS.exp_name,
        config={
            "main": {
                "exp_name": FLAGS.exp_name,
                "job_name": FLAGS.job_name,
            }
        },
    )
    wandb.log(log_dict)
    logging.info(f"Logged metrics to wandb:\n{pprint.pformat(log_dict)}")


if __name__ == "__main__":
    app.run(main)
