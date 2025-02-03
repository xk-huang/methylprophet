from pathlib import Path

import numpy as np
import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string(
    "processed_data_dir",
    "data/processed/encode_wgbs-240802-1e4x1e2-val_ind_tissue",
    "Directory containing processed data",
)
flags.DEFINE_string(
    "metadata_dir",
    "data/metadata/encode_wgbs-240802-1e4x1e2-val_ind_tissue",
    "Directory containing metadata",
)
flags.DEFINE_bool("ipython", False, "Enable IPython")

FLAGS = flags.FLAGS


def main(_):
    processed_data_dir = Path(FLAGS.processed_data_dir)
    metadata_dir = Path(FLAGS.metadata_dir)

    sample_name_to_idx_file = processed_data_dir / "sample_name_to_idx.csv"

    sample_tissue_count_with_idx_file = metadata_dir / "sample_tissue_count_with_idx.csv"
    sample_with_idx_file = metadata_dir / "sample_with_idx.csv"

    sample_name_to_idx = pd.read_csv(sample_name_to_idx_file, names=["sample_name", "sample_idx"], header=0)
    sample_tissue_count_with_idx = pd.read_csv(sample_tissue_count_with_idx_file)
    sample_with_idx = pd.read_csv(sample_with_idx_file)

    selected_columns = ["sample_name", "sample_idx"]
    if not sample_name_to_idx[selected_columns].equals(sample_with_idx[selected_columns]):
        logging.error(
            f"sample_name_to_idx ({sample_name_to_idx_file}) and sample_with_idx ({sample_with_idx_file}) are not equal in {metadata_dir.name}."
        )
    else:
        logging.info(
            f"sample_name_to_idx ({sample_name_to_idx_file}) and sample_with_idx ({sample_with_idx_file}) are equal."
        )
    if not sample_name_to_idx[selected_columns].equals(sample_tissue_count_with_idx[selected_columns]):
        logging.error(
            f"sample_name_to_idx ({sample_name_to_idx_file}) and sample_tissue_count_with_idx ({sample_tissue_count_with_idx_file}) are not equal in {metadata_dir.name}."
        )
    else:
        logging.info(
            f"sample_name_to_idx ({sample_name_to_idx_file}) and sample_tissue_count_with_idx ({sample_tissue_count_with_idx_file}) are equal."
        )

    if FLAGS.ipython:
        from IPython import embed

        embed()


if __name__ == "__main__":
    app.run(main)
