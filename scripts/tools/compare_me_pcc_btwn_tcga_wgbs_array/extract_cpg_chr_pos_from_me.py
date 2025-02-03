import datetime
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string(
    "tcga_array_me_csv", "data/extracted/241213-tcga_array/me_rownamesloc.csv", "Path to the TCGA array ME CSV file"
)

flags.DEFINE_string(
    "tcga_wgbs_me_csv", "data/extracted/241213-tcga_wgbs/me_rownamesloc.csv", "Path to the TCGA WGBS ME CSV file"
)
flags.DEFINE_string("output_dir", "misc/241214-compare_me_pcc_btwn_tcga_wgbs_array/", "Path to the output directory")
flags.DEFINE_bool("overwrite", False, "Whether to overwrite the output directory")

FLAGS = flags.FLAGS


def main(_):
    output_dir = prepare_output_dir(FLAGS.output_dir, overwrite=FLAGS.overwrite)

    logging.info(f"Reading TCGA array ME chr pos from {FLAGS.tcga_array_me_csv}")
    tcga_array_me_chr_pos = read_cpg_chr_pos_from_me(FLAGS.tcga_array_me_csv)
    logging.info(f"Reading TCGA WGBS ME chr pos from {FLAGS.tcga_wgbs_me_csv}")
    tcga_wgbs_cpg_chr_pos = read_cpg_chr_pos_from_me(FLAGS.tcga_wgbs_me_csv)

    tcga_array_me_chr_pos_path = output_dir / "tcga_array_me_chr_pos.csv"
    tcga_array_me_chr_pos.to_csv(tcga_array_me_chr_pos_path, index=False)
    logging.info(f"Saved TCGA array ME chr pos to {tcga_array_me_chr_pos_path}")

    tcga_wgbs_cpg_chr_pos_path = output_dir / "tcga_wgbs_cpg_chr_pos.csv"
    tcga_wgbs_cpg_chr_pos.to_csv(tcga_wgbs_cpg_chr_pos_path, index=False)
    logging.info(f"Saved TCGA WGBS ME chr pos to {tcga_wgbs_cpg_chr_pos_path}")


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {datetime.timedelta(seconds=end - start)}")
        return result

    return wrapper


@timer
def read_cpg_chr_pos_from_me(me_csv):
    return pd.read_csv(me_csv, usecols=[0])


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    app.run(main)
