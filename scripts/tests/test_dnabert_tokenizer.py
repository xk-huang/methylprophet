import datetime
import json
import os
import pprint
import time
from pathlib import Path

from absl import app, flags, logging
from streaming import StreamingDataLoader, StreamingDataset
from transformers import AutoTokenizer

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


train_shuffle_seed = 42
train_batch_size = 3
val_batch_size = 2


class MethylformerStreamingDataset(StreamingDataset):
    def __init__(self, *, group_idx_name_mapping_path=None, **kwargs):
        super().__init__(**kwargs)

        if group_idx_name_mapping_path is not None:
            logging.info(f"Load group_idx_name_mapping from {group_idx_name_mapping_path}")
            with open(group_idx_name_mapping_path, "r") as f:
                self.group_idx_name_mapping = json.load(f)
        else:
            logging.info("group_idx_name_mapping is None")
            self.group_idx_name_mapping = None

        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data

    def collate_fn(self, batch):
        batch_sequence = []
        for data in batch:
            batch_sequence.append(data.pop("sequence").upper())
        batch_sequence_inputs = self.tokenizer(batch_sequence, padding=True, return_tensors="pt")
        breakpoint()

        batch = super().collate_fn(batch)
        return batch


def main(_):
    ddp_env_vars = {key: os.environ.get(key, value) for key, value in DDP_ENV_VARS.items()}
    logging.info(f"DDP Env args: {pprint.pformat(ddp_env_vars)}")

    train_dataset = MethylformerStreamingDataset(
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
        group_idx_name_mapping_path=Path(val_dataset_path) / "group_idx_name_mapping.json",
        local=val_dataset_path,
        batch_size=val_batch_size,
        shuffle=False,
    )
    logging.info(f"Create val dataset: {val_dataset}")

    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=0,
        # persistent_workers=True,
        collate_fn=train_dataset.collate_fn,
    )
    logging.info(f"Create train dataloader: {train_dataloader}")
    val_dataloader = StreamingDataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=0,
        # persistent_workers=True,
        collate_fn=val_dataset.collate_fn,
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
if __name__ == "__main__":
    app.run(main)
