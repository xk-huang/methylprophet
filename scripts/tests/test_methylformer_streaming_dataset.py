import datetime
import json
import os
import pprint
import time
from pathlib import Path

from absl import app, flags, logging
from streaming import StreamingDataLoader, StreamingDataset


DDP_ENV_VARS = {
    "MASTER_ADDR": None,
    "MASTER_PORT": None,
    "NODE_RANK": None,
    "LOCAL_RANK": 0,
    "WORLD_SIZE": 1,
}

train_dataset_path = "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/train"
val_dataset_path = "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val"

gene_expr_df_path = "data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/gene_expr.filtered.parquet"

data_preprocess_config = {
    # NOTE: data_preprocessor_type will be popped in `src/trainer_data_module.py:create_data_preprocess`
    # choices: DatasetV2PreprocessorSCGPT, DatasetV2PreprocessorYF, DatasetV2PreprocessorPredictive
    "data_preprocessor_type": "DatasetV2PreprocessorSCGPT",
    "batched": False,
    "num_nbase": 1200,
    "num_gene_expr_bins": 51,
    "zero_gene_expr_filter": False,
    "cpg_nbase_type": "one_hot",  # ["one_hot", "tokenized"]
    "gene_expr_quantization": True,
}
train_shuffle_seed = 42
train_batch_size = 3
val_batch_size = 2


import pandas as pd

from src.data.data_preprocessor_predictive import DatasetV2PreprocessorPredictive
from src.data.data_preprocessor_scgpt import DatasetV2PreprocessorSCGPT
from src.data.data_preprocessor_yf import DatasetV2PreprocessorYF


def create_data_preprocessor(data_preprocessor_type, gene_expr_df_path, **kwargs):
    gene_expr_df = pd.read_parquet(gene_expr_df_path)
    if data_preprocessor_type == "DatasetV2PreprocessorSCGPT":
        data_preprocessor = DatasetV2PreprocessorSCGPT(gene_expr_df=gene_expr_df, **kwargs)
    elif data_preprocessor_type == "DatasetV2PreprocessorYF":
        data_preprocessor = DatasetV2PreprocessorYF(gene_expr_df=gene_expr_df, **kwargs)
    elif data_preprocessor_type == "DatasetV2PreprocessorPredictive":
        data_preprocessor = DatasetV2PreprocessorPredictive(gene_expr_df=gene_expr_df, **kwargs)
    return data_preprocessor


class MethylformerStreamingDataset(StreamingDataset):
    def __init__(self, *, data_preprocessor=None, group_idx_name_mapping_path=None, **kwargs):
        super().__init__(**kwargs)
        self.data_preprocessor = data_preprocessor

        if group_idx_name_mapping_path is not None:
            logging.info(f"Load group_idx_name_mapping from {group_idx_name_mapping_path}")
            with open(group_idx_name_mapping_path, "r") as f:
                self.group_idx_name_mapping = json.load(f)
        else:
            logging.info("group_idx_name_mapping is None")
            self.group_idx_name_mapping = None

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.data_preprocessor(data)


def main(_):
    ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
    logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")

    data_preprocessor_type = data_preprocess_config.pop("data_preprocessor_type")
    data_preprocessor = create_data_preprocessor(
        data_preprocessor_type,
        gene_expr_df_path,
        **data_preprocess_config,
    )
    logging.info(f"Data preprocess: {data_preprocess_config}")
    train_dataset = MethylformerStreamingDataset(
        data_preprocessor=data_preprocessor,
        group_idx_name_mapping_path=Path(train_dataset_path) / "group_idx_name_mapping.json",
        local=train_dataset_path,
        batch_size=train_batch_size,
        shuffle_seed=train_shuffle_seed,
        shuffle=False,
        epoch_size=12,
        sampling_method="fixed",
        sampling_granularity=12,
    )
    logging.info(f"Create train dataset: {train_dataset}")
    logging.info(f"len of train dataset: {len(train_dataset)}, size of train dataset: {train_dataset.size}")

    val_dataset = MethylformerStreamingDataset(
        data_preprocessor=data_preprocessor,
        group_idx_name_mapping_path=Path(val_dataset_path) / "group_idx_name_mapping.json",
        local=val_dataset_path,
        batch_size=val_batch_size,
        shuffle=False,
    )
    logging.info(f"Create val dataset: {val_dataset}")

    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=data_preprocessor.collate_fn,
        num_workers=1,
        persistent_workers=True,
    )
    logging.info(f"Create train dataloader: {train_dataloader}")
    val_dataloader = StreamingDataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=data_preprocessor.collate_fn,
        num_workers=1,
        persistent_workers=True,
    )
    logging.info(f"Create val dataloader: {val_dataloader}")

    for epoch in range(2):
        tic = time.time()
        for i, data in enumerate(train_dataloader):
            if tic is not None:
                toc = time.time()
                logging.info(f"Time: {datetime.timedelta(seconds=toc - tic)}")
                tic = None

            logging.info(i, data["cpg_idx"], data["sample_idx"], data["group_idx"])


if __name__ == "__main__":
    app.run(main)
