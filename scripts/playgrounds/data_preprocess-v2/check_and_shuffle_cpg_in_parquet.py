"""
Save the data as parquet format.
The mapping is `sample_name` -> all cpg me values whose index is `cpg_id`
The other keys include `keys-sample_name` and `keys-cpg_id`.
See `read_tdb.py` for how to read the data.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("data_dir", "data/parquet/encode_wgbs-240802", "Directory of the data")
flags.DEFINE_alias("d", "data_dir")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_integer("row_chunk_size", 10000, "Row chunk size for parquet conversion")
flags.DEFINE_boolean("overwrite", False, "Overwrite the existing files")


FLAGS = flags.FLAGS


def shuffle_shard(df, shuffled_df_index):
    return df.iloc[shuffled_df_index]


def shuffle_and_save_df(df, shuffled_cpg_id, output_path, row_chunk_size):
    logging.info(f"Save {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(tqdm.trange(0, len(df), row_chunk_size)):
        end = start + row_chunk_size

        shuffled_df_index = shuffled_cpg_id.index[start:end]
        shuffled_df = df.iloc[shuffled_df_index]

        # NOTE xk: reset global index
        shuffled_df.reset_index(drop=True, inplace=True)
        shuffled_df.index += start

        table = pa.Table.from_pandas(shuffled_df)
        pq.write_table(table, output_path / f"{i:05d}.parquet")

        if i == 0:
            logging.info(f"Shuffled {len(shuffled_df)} rows")
            logging.info(shuffled_df.head())
            logging.info(f"Saved {output_path / f'{i:05d}.parquet'}")
            logging.info(f"Saved {output_path}")


def main(_):
    data_dir = FLAGS.data_dir
    data_dir = Path(data_dir)

    me_input_path = data_dir / "me.parquet"
    gene_expr_input_path = data_dir / "gene_expr.parquet"
    cpg_bg_input_path = data_dir / "cpg_bg.parquet"

    logging.info(f"Loading {me_input_path}, {cpg_bg_input_path}")
    me_df = pd.read_parquet(me_input_path)
    logging.info(f"Loaded {me_input_path}")
    # gene_expr_df = pd.read_parquet(gene_expr_input_path)
    cpg_bg_df = pd.read_parquet(cpg_bg_input_path)
    logging.info(f"Loaded {cpg_bg_input_path}")

    def print_load_info(df, name):
        logging.info(f"Loaded {name}")
        logging.info(df.head())
        logging.info(f"Shape: {df.shape}")

    print_load_info(me_df, "me_df")
    print_load_info(cpg_bg_df, "cpg_bg_df")

    # NOTE xk: Check the intersection between me and cpg_bg
    # And make sure the intersection has no repeated cpg_id
    cpg_bg_df_cpg_chr_pos = cpg_bg_df["CpG_location"]
    me_df_joint_cpg_chr_pos = me_df["Unnamed: 0"]
    bg_in_me = cpg_bg_df_cpg_chr_pos.isin(me_df_joint_cpg_chr_pos)
    num_unused_bg = (~bg_in_me).sum()
    logging.info(f"According to cpg_bg, Missing {num_unused_bg} CpG sites from me. Remaining {bg_in_me.sum()}")
    logging.info(f"\t: {cpg_bg_df_cpg_chr_pos[~bg_in_me]}")

    if num_unused_bg > 0:
        raise ValueError("There are missing CpG sites in me")

    me_in_bg = me_df_joint_cpg_chr_pos.isin(cpg_bg_df_cpg_chr_pos)
    num_missing_bg = (~me_in_bg).sum()
    logging.info(f"According to cpg_bg, unknown {num_missing_bg} CpG sites from me. Remaining {me_in_bg.sum()}")
    logging.info(f"\t: {me_df_joint_cpg_chr_pos[~me_in_bg]}")

    if num_missing_bg > 0:
        raise ValueError("There are missing CpG sites in cpg_bg")

    joint_cpg_bg_df_cpg_chr_pos = cpg_bg_df_cpg_chr_pos[bg_in_me]
    joint_me_df_cpg_chr_pos = me_df_joint_cpg_chr_pos[me_in_bg]

    logging.info("Shuffle cpg and me")
    joint_cpg_bg_df_cpg_chr_pos = joint_cpg_bg_df_cpg_chr_pos.sample(
        frac=1, random_state=np.random.default_rng(FLAGS.seed)
    )
    joint_me_df_cpg_chr_pos = joint_me_df_cpg_chr_pos.sample(frac=1, random_state=np.random.default_rng(FLAGS.seed))

    logging.info("Check uniqueness of cpg_chr_pos")
    if not (joint_cpg_bg_df_cpg_chr_pos.values == joint_me_df_cpg_chr_pos.values).all():
        raise ValueError("cpg_chr_pos is not the same between me and cpg_bg after shuffling.")

    # FIXME xk: we just remove the duplicated cpg_chr_pos, but we should make sure the cpg_chr_pos is unique in me.csv
    logging.warning(
        "The joint cpg_chr_pos is not unique! Remove duplicated cpg_chr_pos. In the future, we need to make sure the cpg_chr_pos is unique in me.csv."
    )
    joint_me_df_cpg_chr_pos = joint_me_df_cpg_chr_pos[~joint_me_df_cpg_chr_pos.duplicated(keep="last")]
    joint_cpg_bg_df_cpg_chr_pos = joint_cpg_bg_df_cpg_chr_pos[~joint_cpg_bg_df_cpg_chr_pos.duplicated(keep="last")]
    if not (joint_cpg_bg_df_cpg_chr_pos.values == joint_me_df_cpg_chr_pos.values).all():
        raise ValueError("cpg_chr_pos is not the same between me and cpg_bg after removing duplicated cpg_chr_pos.")

    joint_cpg_bg_df_cpg_chr_pos_unique = joint_cpg_bg_df_cpg_chr_pos.unique()
    if len(joint_cpg_bg_df_cpg_chr_pos) != len(joint_cpg_bg_df_cpg_chr_pos_unique):
        cpg_bg_cpg_chr_pos_duplicated = joint_me_df_cpg_chr_pos.duplicated(keep=False)
        cpg_bg_seq_duplicated = cpg_bg_df["sequence"].duplicated(keep=False)
        if not (cpg_bg_df[cpg_bg_cpg_chr_pos_duplicated].index == cpg_bg_df[cpg_bg_seq_duplicated].index).all():
            raise ValueError("The duplicated cpg_chr_pos is not the same as duplicated sequence.")
        me_df_duplicated_cpg_chr_pos = me_df[joint_me_df_cpg_chr_pos.duplicated(keep=False)].sort_values("Unnamed: 0")
        save_me_df_duplicated_cpg_chr_pos = data_dir / "me.duplicated_cpg_chr_pos.csv"
        me_df_duplicated_cpg_chr_pos = me_df_duplicated_cpg_chr_pos.iloc[:, :10]
        me_df_duplicated_cpg_chr_pos.to_csv(save_me_df_duplicated_cpg_chr_pos)
        logging.info(f"Saved 10 columns of duplicated cpg_chr_pos in me to {save_me_df_duplicated_cpg_chr_pos}")
        raise ValueError("The joint cpg_chr_pos is not unique!")

    logging.info(f"Unique CpG sites: {len(joint_cpg_bg_df_cpg_chr_pos_unique)} vs. Full CpG sites: {len(joint_cpg_bg_df_cpg_chr_pos)}")

    shuffled_me_output_path = data_dir / "me.shuffled.parquet"
    shuffled_cpg_bg_output_path = data_dir / "cpg_bg.shuffled.parquet"
    row_chunk_size = FLAGS.row_chunk_size
    if shuffled_me_output_path.exists():
        if FLAGS.overwrite:
            shutil.rmtree(shuffled_me_output_path)
        else:
            logging.info(f"{shuffled_me_output_path} exists. Skip")
            return
    if shuffled_cpg_bg_output_path.exists():
        if FLAGS.overwrite:
            shutil.rmtree(shuffled_cpg_bg_output_path)
        else:
            logging.info(f"{shuffled_cpg_bg_output_path} exists. Skip")
            return

    shuffle_and_save_df(me_df, joint_me_df_cpg_chr_pos, shuffled_me_output_path, row_chunk_size)
    shuffle_and_save_df(cpg_bg_df, joint_cpg_bg_df_cpg_chr_pos, shuffled_cpg_bg_output_path, row_chunk_size)


if __name__ == "__main__":
    app.run(main)
