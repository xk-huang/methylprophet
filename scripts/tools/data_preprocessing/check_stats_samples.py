"""
python scripts/tools/data_preprocessing/check_stats_samples.py \
    --input_me_paruqet data/parquet/241213-encode_wgbs/me.parquet/00000.parquet \
    --input_gene_expr_parquet data/parquet/241213-encode_wgbs/gene_expr.parquet/00000.parquet \
    --output_dir data/parquet/241213-encode_wgbs/metadata/check_stats_samples \
"""

import json
import shutil
from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_me_paruqet", None, "Path to the input parquet file")
flags.DEFINE_string("input_gene_expr_parquet", None, "Path to the input parquet file")
flags.mark_flag_as_required("input_me_paruqet")
flags.mark_flag_as_required("input_gene_expr_parquet")

flags.DEFINE_string("output_dir", None, "Path to the output directory")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output directory")

FLAGS = flags.FLAGS


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, overwrite=True)
    logging.info(f"Output directory: {output_dir}")

    me_df = pd.read_parquet(FLAGS.input_me_paruqet)
    gene_expr_df = pd.read_parquet(FLAGS.input_gene_expr_parquet)

    if me_df.columns[0] != "Unnamed: 0":
        raise ValueError(f"First column of me_df is not 'Unnamed: 0'. Got: {me_df.columns[0]}")
    if gene_expr_df.columns[0] != "Unnamed: 0":
        raise ValueError(f"First column of gene_expr_df is not 'Unnamed: 0'. Got: {gene_expr_df.columns[0]}")

    me_samples = me_df.columns[1:]
    gene_expr_samples = gene_expr_df.columns[1:]
    if (me_samples != gene_expr_samples).any():
        with open(output_dir / "mismatched_samples.txt", "w") as f:
            me_sample_set = set(me_samples)
            gene_expr_sample_set = set(gene_expr_samples)
            f.write("me_samples - gene_expr_samples\n")
            f.write("\n".join(sorted(me_sample_set - gene_expr_sample_set)))
            f.write("\n\n")
            f.write("gene_expr_samples - me_samples\n")
            f.write("\n".join(sorted(gene_expr_sample_set - me_sample_set)))
        raise ValueError("ME and gene expression samples are not the same")
    else:
        logging.info("ME and gene expression samples are the same")
        with open(output_dir / "all_samples_matched.txt", "w") as f:
            f.write("all_samples_matched")

    stats = {
        "num_samples": len(me_samples),
        "sample_names": list(me_samples),
    }
    output_stats_path = output_dir / "sample_stats.json"
    with open(output_stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logging.info(f"Sample stats saved to: {output_stats_path}")


def prepare_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Output directory {output_dir} already exists. Deleting.")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory {output_dir} already exists. Exiting.")
            exit()
    output_dir.mkdir(parents=True)
    return output_dir


if __name__ == "__main__":
    app.run(main)
