import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("data_dir", "data/parquet/encode_wgbs-240802", "Directory containing the parquet files")
flags.DEFINE_alias("d", "data_dir")
flags.DEFINE_string("save_dir", "data/processed/encode_wgbs-240802", "Directory to save the processed data")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite the existing files")

flags.DEFINE_string("cpg_bg_file_name", "cpg_bg.shuffled.parquet", "Name of the CpG background file")
flags.DEFINE_string("me_file_name", "me.shuffled.parquet", "Name of the methylation file")
flags.DEFINE_string("gene_expr_file_name", "gene_expr.parquet", "Name of the gene expression file")
flags.DEFINE_integer("sample_chunk_size", 100, "Number of samples to process in each chunk")
flags.DEFINE_integer("cpg_chunk_size", 10000, "Number of samples to process in each chunk")

flags.DEFINE_string("split_name", None, "Name of the dataset split to process", required=True)
flags.DEFINE_string("selected_cpg_shard_list", None, "The range of chosen cpg shards, [left, right)", required=True)
flags.DEFINE_string("selected_sample_list", None, "The range of chosen samples [left, right)")
flags.DEFINE_string("selected_sample_from_csv_file", None, "Path to the sample shard list file")
flags.DEFINE_integer("limit_num_cpg_per_shard", None, "Number of CpGs to limit per shard")

flags.DEFINE_boolean("shuffle", False, "Whether to shuffle the cpg-sample pairs")
flags.DEFINE_integer("seed", 42, "Random seed for shuffling")

FLAGS = flags.FLAGS


def get_gene_expr_for_each_row(gene_expr_df):
    def _get_gene_expr_for_each_row(row):
        sample_name = row["sample_name"]
        gene_expr = gene_expr_df[sample_name]
        return gene_expr.to_numpy()

    return _get_gene_expr_for_each_row


def main(_):
    data_dir = Path(FLAGS.data_dir)
    save_dir = Path(FLAGS.save_dir)

    cpg_bg_file_name = FLAGS.cpg_bg_file_name
    me_file_name = FLAGS.me_file_name
    gene_expr_file_name = FLAGS.gene_expr_file_name

    print(f"Loading gene expr and cpg bg from {data_dir}")
    gene_expr_df = pd.read_parquet(str(data_dir / gene_expr_file_name))
    tic = time.time()
    cpg_bg_df = pd.read_parquet(str(data_dir / cpg_bg_file_name), columns=["CpG_location"])
    toc = time.time()
    logging.info(f"Time taken to load CpG background dataframe: {toc - tic:.2f} seconds")
    print(f"Gene expression dataframe shape: {gene_expr_df.shape}")
    print(f"CPG background dataframe shape: {cpg_bg_df.shape}")

    # Get keys to index mappings
    cpg_chr_pos_keys = cpg_bg_df["CpG_location"]
    cpg_chr_pos_keys_to_idx = pd.Series(np.arange(len(cpg_chr_pos_keys)), index=cpg_chr_pos_keys)

    cpg_chr_pos_key_to_idx_csv_file = save_dir / "cpg_chr_pos_to_idx.csv"
    if not cpg_chr_pos_key_to_idx_csv_file.exists():
        save_dir.mkdir(exist_ok=True, parents=True)
        cpg_chr_pos_keys_to_idx.to_csv(cpg_chr_pos_key_to_idx_csv_file)

    # Get gene expression keys to index mappings
    gene_expr_df.rename(columns={"Unnamed: 0": "gene_name"}, inplace=True)
    # gene_name_keys = gene_expr_df["gene_name"]
    # gene_name_keys_to_idx = pd.Series(np.arange(len(gene_name_keys)), index=gene_name_keys)
    gene_expr_df.set_index("gene_name", inplace=True)

    split_name = FLAGS.split_name
    if split_name is None:
        raise ValueError("split_name is required")

    dataset_output_path = save_dir / "me_cpg_dataset" / (FLAGS.split_name + ".parquet")
    if dataset_output_path.exists():
        if FLAGS.overwrite:
            print(f"Removing existing save directory {dataset_output_path}")
            shutil.rmtree(dataset_output_path)
        else:
            logging.info(f"Save directory {dataset_output_path} already exists. Exiting...")
            return
    dataset_output_path.mkdir(exist_ok=True, parents=True)
    me_df_path_ls = sorted(data_dir.glob(f"{me_file_name}/*.parquet"))

    selected_cpg_shard_list = FLAGS.selected_cpg_shard_list
    selected_cpg_shard_left, selected_cpg_shard_right = map(int, selected_cpg_shard_list.split(","))
    if selected_cpg_shard_left < 0:
        raise ValueError(f"selected_cpg_shard_left {selected_cpg_shard_left} < 0")
    if selected_cpg_shard_left >= selected_cpg_shard_right:
        raise ValueError(
            f"selected_cpg_shard_left {selected_cpg_shard_left} >= selected_cpg_shard_right {selected_cpg_shard_right}"
        )
    if selected_cpg_shard_right > len(me_df_path_ls):
        raise ValueError(
            f"selected_cpg_shard_right {selected_cpg_shard_right} > len(me_df_path_ls) {len(me_df_path_ls)}"
        )
    me_df_path_ls = me_df_path_ls[selected_cpg_shard_left:selected_cpg_shard_right]
    logging.info(f"Selected CpG shard list: {me_df_path_ls[:3]}, ..., {me_df_path_ls[-3:]}")

    selected_sample_list = FLAGS.selected_sample_list
    if selected_sample_list is None:
        selected_sample_left, selected_sample_right = None, None
    else:
        selected_sample_left, selected_sample_right = map(int, selected_sample_list.split(","))
        if selected_sample_left < 0:
            raise ValueError(f"selected_sample_left {selected_sample_left} < 0")
        if selected_sample_left >= selected_sample_right:
            raise ValueError(
                f"selected_sample_left {selected_sample_left} >= selected_sample_right {selected_sample_right}"
            )
        logging.info(f"Selected sample list: {selected_sample_left}, ..., {selected_sample_right}")

    sample_chunk_size = FLAGS.sample_chunk_size
    processed_num_cpg_sample_pairs = 0
    processed_num_cpgs = 0
    limit_num_cpg_per_shard = FLAGS.limit_num_cpg_per_shard

    old_sample_name_keys_to_idx = None
    pbar = tqdm.tqdm(me_df_path_ls)
    for me_df_path in pbar:
        me_df_path = Path(me_df_path)
        cpg_bg_df_path = data_dir / cpg_bg_file_name / me_df_path.name

        # print(f"Loading ME dataframe from {me_df_path}")
        me_df = pd.read_parquet(me_df_path)
        cpg_bg_df = pd.read_parquet(cpg_bg_df_path)

        processed_num_cpgs += len(me_df)

        me_df.rename(columns={"Unnamed: 0": "cpg_chr_pos"}, inplace=True)
        me_df.set_index("cpg_chr_pos", inplace=True)
        sample_name_keys_from_me = me_df.columns
        # NOTE: Sort the sample names, align with `stats_sample_name-encode_wgbs.py`
        sample_name_keys_from_me = sorted(sample_name_keys_from_me)
        sample_name_keys_to_idx = pd.Series(np.arange(len(sample_name_keys_from_me)), index=sample_name_keys_from_me)

        sample_name_keys_to_idx_csv_file = save_dir / "sample_name_to_idx.csv"
        if old_sample_name_keys_to_idx is None:
            sample_name_keys_to_idx.to_csv(sample_name_keys_to_idx_csv_file)
            old_sample_name_keys_to_idx = sample_name_keys_to_idx
        else:
            if not old_sample_name_keys_to_idx.equals(sample_name_keys_to_idx):
                raise ValueError(
                    "Sample names in ME dataframes do not match with sample names in old sample_name_keys_to_idx"
                )

        cpg_bg_df.rename(columns={"CpG_location": "cpg_chr_pos"}, inplace=True)
        cpg_bg_df.set_index("cpg_chr_pos", inplace=True)

        if not (cpg_bg_df.index == me_df.index).all():
            raise ValueError(
                f"CpG locations in CpG background and ME dataframes do not match in {me_df_path} and {cpg_bg_df_path}"
            )
        if not (gene_expr_df.columns == me_df.columns).all():
            raise ValueError(
                f"Sample names in gene expression and ME dataframes do not match in {me_df_path} and {gene_expr_file_name}"
            )

        selected_sample_from_csv_file = FLAGS.selected_sample_from_csv_file
        if selected_sample_from_csv_file is not None:
            selected_sample_csv = pd.read_csv(selected_sample_from_csv_file)

            selected_sample_name_keys_to_idx = pd.Series(
                selected_sample_csv["sample_idx"].to_numpy(), selected_sample_csv["sample_name"]
            )
            if not (selected_sample_name_keys_to_idx.index.isin(sample_name_keys_to_idx.index)).all():
                raise ValueError("Sample names in selected_sample_csv do not match with sample names in me_df")
            if not (selected_sample_name_keys_to_idx.isin(sample_name_keys_to_idx)).all():
                raise ValueError("Sample names in selected_sample_csv do not match with sample names in me_df")

            if not (
                sample_name_keys_to_idx[selected_sample_name_keys_to_idx.index] == selected_sample_name_keys_to_idx
            ).all():
                raise ValueError("Sample names in me_df do not match with sample names in selected_sample_csv")

            logging.info(f"Selected sample names from {selected_sample_from_csv_file}")
            logging.info(f"Previous sample names:\n{sample_name_keys_to_idx}")
            logging.info(f"Current selected sample names:\n{selected_sample_name_keys_to_idx}")
            me_df = me_df[selected_sample_name_keys_to_idx.index]

        num_samples_in_shard = len(me_df.columns)
        if selected_sample_left is None:
            selected_sample_left = 0
        if selected_sample_right is None:
            selected_sample_right = num_samples_in_shard

        if selected_sample_left >= num_samples_in_shard:
            raise ValueError(
                f"selected_sample_left {selected_sample_left} >= num_samples_in_shard {num_samples_in_shard}"
            )
        if selected_sample_right > num_samples_in_shard:
            raise ValueError(
                f"selected_sample_right {selected_sample_right} > num_samples_in_shard {num_samples_in_shard}"
            )

        shuffle = FLAGS.shuffle
        rng = np.random.default_rng(FLAGS.seed)
        logging.info(f"Shuffling: {shuffle}, Random seed {FLAGS.seed}")

        sample_chunk_size_in_shard = min(sample_chunk_size, selected_sample_right - selected_sample_left)
        cpg_chunk_size_in_shard = min(FLAGS.cpg_chunk_size, len(me_df))
        for cpg_chunk_idx in range(0, len(me_df), cpg_chunk_size_in_shard):
            for sample_chunk_idx in range(selected_sample_left, selected_sample_right, sample_chunk_size_in_shard):
                me_chunk = me_df.iloc[
                    cpg_chunk_idx : cpg_chunk_idx + cpg_chunk_size_in_shard,
                    sample_chunk_idx : sample_chunk_idx + sample_chunk_size_in_shard,
                ].copy()

                id_vars_for_melt = []
                me_chunk["cpg_idx"] = me_chunk.index.map(cpg_chr_pos_keys_to_idx)
                id_vars_for_melt.append("cpg_idx")
                me_chunk["sequence"] = cpg_bg_df.loc[me_chunk.index, "sequence"]
                id_vars_for_melt.append("sequence")

                if limit_num_cpg_per_shard is not None:
                    if len(me_chunk) > limit_num_cpg_per_shard:
                        logging.info(
                            f"Limiting number of CpGs per shard to {limit_num_cpg_per_shard} in {me_df_path}, "
                            f"dropping {len(me_chunk) - limit_num_cpg_per_shard} CpGs"
                        )
                        me_chunk = me_chunk.head(limit_num_cpg_per_shard)

                me_flat_chunk = me_chunk.melt(
                    ignore_index=False, id_vars=id_vars_for_melt, var_name="sample_name", value_name="methylation"
                ).reset_index()
                me_flat_chunk["sample_idx"] = me_flat_chunk["sample_name"].map(sample_name_keys_to_idx)

                # NOTE xk: take too long to save those gene expression data
                # tic = time.time()
                # me_flat_chunk["gene_expr"] = me_flat_chunk.apply(get_gene_expr_for_each_row(gene_expr_df), axis=1)
                # toc = time.time()
                # print(f"Getting gene expression for each row. Time taken: {toc - tic:.1f} seconds")

                me_flat_chunk.reset_index(drop=True, inplace=True)
                if shuffle:
                    me_flat_chunk = me_flat_chunk.sample(frac=1, random_state=rng)

                me_flat_chunk.index += processed_num_cpg_sample_pairs
                processed_num_cpg_sample_pairs += len(me_flat_chunk)
                # print(me_flat_chunk)

                # Filter out rows with NaN values, esp. for methylation values
                num_nan_me = me_flat_chunk["methylation"].isna().sum()
                if num_nan_me > 0:
                    logging.warning(
                        f"Found {num_nan_me} NaN values in methylation column in {me_df_path}, dropping them"
                    )
                    me_flat_chunk = me_flat_chunk.dropna(subset=["methylation"])

                chunk_output_file_path = (
                    dataset_output_path / f"{me_df_path.stem}-{cpg_chunk_idx:05d}-{sample_chunk_idx:05d}.parquet"
                )
                # logging.info(f"Writing table to {chunk_output_file_path}")
                me_flat_chunk.to_parquet(
                    chunk_output_file_path, index=False
                )  # NOTE xk: index=False, as we do not need it in huggingface datasets.
                # logging.info(f"Finished writing table to {chunk_output_file_path}")
            pbar.set_description(
                f"Processed {processed_num_cpgs} CpGs and {processed_num_cpg_sample_pairs} CpG-sample pairs"
            )


if __name__ == "__main__":
    app.run(main)
