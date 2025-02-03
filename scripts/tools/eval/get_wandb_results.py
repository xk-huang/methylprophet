from pathlib import Path

import pandas
from absl import app, flags, logging

import wandb


flags.DEFINE_string("entity", "xk-huang", "Wandb entity")
flags.DEFINE_string("project", "methylformer", "Wandb project")
flags.DEFINE_list(
    "run_ids",
    ["k5steu2p", "a5jxnyl4", "a48duxzh", "7hqpi5bs", "erjb8lqv", "r8lxojd6", "fhkr5exp", "7rpmtw6t"],
    "Wandb run IDs",
)
flags.DEFINE_string("output_dir", "outputs/eval/", "Output directory")
flags.DEFINE_string("output_filename", "wandb_results", "Output filename")
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
FLAGS = flags.FLAGS


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{FLAGS.output_filename}.csv"
    if output_file_path.exists():
        if FLAGS.overwrite:
            logging.info(f"Overwrite existing file: {output_file_path}")
            output_file_path.unlink()
        else:
            logging.warning(f"File already exists: {output_file_path}")
            return

    entity = FLAGS.entity
    project = FLAGS.project
    run_ids = FLAGS.run_ids

    api = wandb.Api()

    stage_eval_metric_names = [
        ("eval-test", "pcc_by_cpg_id-median"),
        ("eval-test", "pcc_by_sample_id-median"),
        ("eval-test", "mse_loss_per_point"),
        ("eval-test", "mae_loss_per_point"),
    ]
    data_split_names = [
        "train_cpg-val_sample.parquet",
        "val_cpg-train_sample.parquet",
        "val_cpg-val_sample.parquet",
    ]

    result_df_list = []
    for stage_name, eval_metric_name in stage_eval_metric_names:
        result_dict_list = []
        exp_job_list = []

        for run_id in run_ids:
            run = api.run(f"{entity}/{project}/{run_id}")
            # print(f"Run ID: {run_id}")
            job_name = run.config["main"]["job_name"]
            exp_name = run.config["main"]["exp_name"]
            # print(f"Experiment Name: {exp_name}")
            # print(f"Job Name: {job_name}")
            exp_job_list.append(f"{exp_name}/{job_name}")

            result_dict = {}
            for data_split_name in data_split_names:
                metric_name = f"{stage_name}/{data_split_name}/{eval_metric_name}"
                metric = run.summary.get(metric_name)

                result_dict_key_name = (f"{stage_name}/{eval_metric_name}", data_split_name)
                result_dict[result_dict_key_name] = metric

            # pprint.pprint(result_dict)
            result_dict_list.append(result_dict)

        result_df = pandas.DataFrame(result_dict_list, index=exp_job_list)
        result_df.columns = pandas.MultiIndex.from_tuples(result_df.columns)
        result_df_list.append(result_df)

    result_df = pandas.concat(result_df_list, axis=1)

    result_df.to_csv(output_file_path)
    print(f"Write outputs to: {output_file_path}")


if __name__ == "__main__":
    app.run(main)
