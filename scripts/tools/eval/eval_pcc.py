"""
pip install absl-py

python scripts/tools/eval/eval_pcc.py \
    --input_result_df outputs/241229-methylformer_bert-ins/eval-241229-train-encode-base-12xl40s/eval/version_0/eval_results-test.csv \
    --input_group_idx_name_mapping_json outputs/241229-methylformer_bert-ins/eval-241229-train-encode-base-12xl40s/eval/version_0/group_idx_name_mapping-test.json \
    --output_dir outputs/241229-methylformer_bert-ins/eval-241229-train-encode-base-12xl40s/eval/version_0/eval \
    --pcc_backend pandas

# BACKENDS = ["pandas", "torchmetrics_cpu", "torchmetrics_cuda"]
"""

import json
import os.path as osp
import shutil
from pathlib import Path

import pandas as pd
from absl import app, flags, logging

from src.eval import MethylEval


flags.DEFINE_string("input_result_df", None, "Input result csv file")
flags.mark_flag_as_required("input_result_df")

flags.DEFINE_string("input_group_idx_name_mapping_json", None, "Input group index name mapping json file")
flags.mark_flag_as_required("input_group_idx_name_mapping_json")


flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")
flags.DEFINE_bool("only_plot", False, "Only plot the figures")
flags.DEFINE_string("pcc_backend", "torchmetrics_cuda", "PCC backend")
flags.DEFINE_integer("num_batches_per_group", 10, "Number of batches per group")

FLAGS = flags.FLAGS


def enlarge_data_scale_10x(df):
    # XXX (xk): repeat df for 10x to simulate 10% cpg data scale.
    df = pd.concat([df] * 23, ignore_index=True)
    return df


# def test_equal_pandas(df, pcc_by_sample_id, pcc_by_cpg_id):
#     pcc_by_sample_id_pd = compute_pcc_by_group(df, "sample_id", "pandas")
#     pcc_by_cpg_id_pd = compute_pcc_by_group(df, "cpg_id", "pandas")
#     pcc_by_sample_id_max_diff = (pcc_by_sample_id - pcc_by_sample_id_pd).abs().max()
#     pcc_by_cpg_id_max_diff = (pcc_by_cpg_id - pcc_by_cpg_id_pd).abs().max()
#     logging.info(f"pcc_by_sample_id_max_diff: {pcc_by_sample_id_max_diff}")
#     logging.info(f"pcc_by_cpg_id_max_diff: {pcc_by_cpg_id_max_diff}")
#     assert pcc_by_sample_id_max_diff < 1e-10
#     assert pcc_by_cpg_id_max_diff < 1e-10


def read_df(file_path: str) -> pd.DataFrame:
    file_path = Path(file_path)
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def main(_):
    output_dir = Path(FLAGS.output_dir)
    if output_dir.exists():
        if FLAGS.overwrite:
            logging.info(f"Remove existing output_dir: {output_dir}, overwrite.")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"output_dir ({output_dir}) already exists.")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"loading result df: {FLAGS.input_result_df}")
    df = read_df(FLAGS.input_result_df)
    logging.info(f"df.shape: {df.shape}")

    num_batches_per_group = FLAGS.num_batches_per_group
    input_group_idx_name_mapping = None
    with open(FLAGS.input_group_idx_name_mapping_json, "r") as f:
        input_group_idx_name_mapping = json.load(f)

    backend = FLAGS.pcc_backend
    logging.info(f"backend: {backend}")
    # Start evaluation
    for group_idx, group_name in input_group_idx_name_mapping.items():
        logging.info(f"Start evaluating group: {group_idx}, {group_name}")
        group_idx = int(group_idx)
        group_name = osp.basename(group_name)
        result_df = df[df["group_idx"] == group_idx]
        result_df = result_df.dropna()

        methyl_eval = MethylEval(result_df, output_dir, num_batches_per_group=num_batches_per_group, backend=backend)

        log_dict = methyl_eval.eval_results_to_log_dict()
        logging.info(f"log_dict: {log_dict}")
        output_log_dict_path = output_dir / f"log_dict-{group_name}.json"
        logging.info(f"Save log_dict to: {output_log_dict_path}")

        # NOTE xk: convert the tensor to float for json serialization.
        log_dict = {k: float(v) for k, v in log_dict.items()}
        with open(output_log_dict_path, "w") as f:
            json.dump(log_dict, f, indent=4)
        methyl_eval.eval_results_to_plot(prefix_name=group_name)


if __name__ == "__main__":
    app.run(main)
