"""
Dataset v2 playground

We use huggingface datasets library to load the dataset. The data should have this structure:
```
data/processed/encode_wgbs-240802
├── gene_expr.filtered.parquet
└── me_cpg_dataset.parquet
    ├── 00000-00000.parquet
    ├── 00001-00000.parquet
    ├── 00002-00000.parquet
    ├── ...
    └── 00099-00000.parquet
```
"""

from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import torch
import tqdm
from absl import app, flags, logging
from datasets import load_dataset
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate

from src.trainer_data_module import TrainerDataModule


class DummyMethylModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.module = torch.nn.Linear(1, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def forward(self, x):
        return self.module(x)

    def convert_data_type(self, data):
        if isinstance(data, torch.Tensor):
            if torch.is_floating_point(data):
                return data.float()
            return data
        elif isinstance(data, dict):
            return {key: self.convert_data_type(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_data_type(value) for value in data]
        else:
            return data

    def training_step(self, batch, batch_idx):
        rank = self.trainer.global_rank
        cpg_idx = batch["cpg_idx"]
        sample_idx = batch["sample_idx"]
        logging.info(f"train: rank={rank}, batch_idx={batch_idx}, cpg_idx={cpg_idx}, sample_idx={sample_idx}")

        # if self.trainer.is_global_zero:
        #     breakpoint()
        # self.trainer.strategy.barrier()
        batch = self.convert_data_type(batch)

        x = batch["methylation"].unsqueeze(-1)
        x = self.module(x).mean()

        return x

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        rank = self.trainer.global_rank
        cpg_idx = batch["cpg_idx"]
        sample_idx = batch["sample_idx"]
        name = self.trainer.val_dataloaders[dataloader_idx].dataset.name
        logging.info(
            f"val: name={dataloader_idx} rank={rank}, batch_idx={batch_idx}, cpg_idx={cpg_idx}, sample_idx={sample_idx}"
        )

        # if self.trainer.is_global_zero:
        #     breakpoint()
        # self.trainer.strategy.barrier()
        batch = self.convert_data_type(batch)

        x = batch["methylation"].unsqueeze(-1)
        x = self.module(x).mean()

        return x

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        rank = self.trainer.global_rank
        cpg_idx = batch["cpg_idx"]
        sample_idx = batch["sample_idx"]
        name = self.trainer.val_dataloaders[dataloader_idx].dataset.name
        logging.info(
            f"test: name={dataloader_idx} rank={rank}, batch_idx={batch_idx}, cpg_idx={cpg_idx}, sample_idx={sample_idx}"
        )

        # if self.trainer.is_global_zero:
        #     breakpoint()
        # self.trainer.strategy.barrier()
        batch = self.convert_data_type(batch)

        x = batch["methylation"].unsqueeze(-1)
        x = self.module(x).mean()

        return x


# flags.DEFINE_string("data_dir", "data/processed/encode_wgbs-240802", "Directory where the dataset is stored")
# flags.DEFINE_alias("d", "data_dir")
# flags.DEFINE_string("split_name", "debug", "Name of the split")
flags.DEFINE_integer("devices", 1, "Number of devices")

FLAGS = flags.FLAGS


class DatasetV2Preprocessor:
    def __init__(self, gene_expr_df, batched=False, num_gene_expr_bins=51, zero_gene_expr_filter=True):
        self.gene_expr_df = gene_expr_df
        self.batched = batched
        if batched is True:
            raise NotImplementedError("batched mode is not supported yet")

        self.num_gene_expr_bins = num_gene_expr_bins
        self.zero_gene_expr_filter = zero_gene_expr_filter
        logging.info(
            f"DatasetV2Preprocessor: num_gene_expr_bins={num_gene_expr_bins}, zero_gene_expr_filter={zero_gene_expr_filter}"
        )

    def add_gene_expr_to_data(self, data):
        sample_name = data["sample_name"]
        # sample_idx = data["sample_idx"]
        gene_expr = self.gene_expr_df[sample_name]
        data["gene_expr"] = gene_expr.to_numpy()
        return data

    nbase_mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": 4,
    }

    def tokenize_nbase(self, data):
        cpg_nbase_seq = data["sequence"]
        cpg_nbase_seq = cpg_nbase_seq.upper()
        cpg_nbase_seq = [self.nbase_mapping[c] for c in cpg_nbase_seq]
        # NOTE: torch.nn.Embedding requires the input to be int64 (LongTensor).
        cpg_nbase_seq = np.array(cpg_nbase_seq, dtype=np.int64)
        data["tokenized_sequence"] = cpg_nbase_seq
        return data

    def tokenize_gene(self, data):
        tokenized_gene = np.arange(len(data["gene_expr"]))
        data["tokenized_gene"] = tokenized_gene
        return data

    def quantize_gene_expr(self, data):
        # NOTE xk: scGPT takes binning techniques: quantize the values from 0-max per cell
        # https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/preprocess.py#L274
        gene_expr = data["gene_expr"]

        # Get quantile
        non_zero_gene_expr_slice = gene_expr > 0
        non_zero_gene_expr = gene_expr[non_zero_gene_expr_slice]
        bins = np.quantile(
            non_zero_gene_expr, np.linspace(0, 1, self.num_gene_expr_bins - 1)
        )  # NOTE xk: on zero values are quantized to [1 - self.num_gene_expr_bins - 1]
        quant_non_zero_gene_expr = quantize(non_zero_gene_expr, bins)
        quant_gene_expr = np.zeros_like(gene_expr, dtype=np.int64)
        quant_gene_expr[non_zero_gene_expr_slice] = quant_non_zero_gene_expr

        data["quantized_gene_expr"] = quant_gene_expr
        return data

    def check_me(self, data):
        if data["methylation"] is np.nan:
            raise ValueError("NaN values in methylation")
        if data["methylation"] is None:
            raise ValueError("None values in methylation")

    def check_gene_expr_df(self, gene_expr_df):
        if gene_expr_df.isnull().values.any():
            raise ValueError("NaN values in gene expression dataframe")

    def filter_zero_gene_expr(self, data):
        non_zero_gene_expr_slice = data["quantized_gene_expr"] != 0
        data["gene_expr"] = data["gene_expr"][non_zero_gene_expr_slice]
        data["quantized_gene_expr"] = data["quantized_gene_expr"][non_zero_gene_expr_slice]
        data["tokenized_gene"] = data["tokenized_gene"][non_zero_gene_expr_slice]
        data["gene_attention_mask"] = data["gene_attention_mask"][non_zero_gene_expr_slice]
        return data

    def apply_log1_hvg(self, data):
        raise NotImplementedError

    def add_attention_mask(self, data):
        attention_mask = np.ones_like(data["tokenized_gene"], dtype=np.int64)
        data["gene_attention_mask"] = attention_mask
        return data

    def __call__(self, data):
        self.check_me(data)
        self.check_gene_expr_df(self.gene_expr_df)

        data = self.add_gene_expr_to_data(data)
        data = self.tokenize_nbase(data)
        data = self.tokenize_gene(data)
        data = self.quantize_gene_expr(data)
        data = self.add_attention_mask(data)

        if self.zero_gene_expr_filter:
            data = self.filter_zero_gene_expr(data)

        return data


_KEYS_TO_PAD = ["quantized_gene_expr", "tokenized_gene", "gene_expr", "gene_attention_mask"]


def pad_gene_collate_fn(batch):
    # Pad gene expression and generate the attention mask for it.
    for key in _KEYS_TO_PAD:
        if key not in batch[0]:
            continue
        max_len = max(pad_sample[key].shape[0] for pad_sample in batch)
        for sample in batch:
            # fmt: off
            sample[key] = np.pad(sample[key], (0, max_len - sample[key].shape[0]),mode="constant", constant_values=0)

    return default_collate(batch)


def quantize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def create_me_cpg_gene_dataset(data_dir):
    me_cpg_datset_split_dir = Path(data_dir) / "me_cpg_dataset"
    logging.info(f"Loading dataset from {me_cpg_datset_split_dir}")

    # NOTE xk: path_to_me_cpg_dataset/{split_name}.parquet/xxxxx-yyyyy.parquet
    # Load multiple files: https://stackoverflow.com/a/78700397/9690936
    split_dirs = sorted(me_cpg_datset_split_dir.glob("*.parquet"))
    data_files = {}
    for split_dir in split_dirs:
        split_name = split_dir.stem
        # NOTE xk: split_name should not contains `-` as datasets library does not support it.
        split_name = split_name.replace("-", "_")
        data_files[split_name] = [
            f"{split_dir.name}/{parquet_file.name}" for parquet_file in sorted(split_dir.glob("*.parquet"))
        ]
    logging.info(f"Data files:\n{pformat(data_files)}")

    dataset = load_dataset(
        "parquet",
        data_dir=str(me_cpg_datset_split_dir),
        data_files=data_files,
        streaming=True,
    )

    # NOTE xk: assign `name` to each dataset, as eval needs it.
    for split_name in dataset.keys():
        dataset[split_name].name = split_name

    return dataset


def create_data_preprocess(data_dir):
    gene_expr_path = Path(data_dir) / "gene_expr.filtered.parquet"
    gene_expr_df = pd.read_parquet(gene_expr_path)
    data_preprocessor = DatasetV2Preprocessor(gene_expr_df)
    return data_preprocessor


def apply_data_preprocess(dataset, data_preprocessor):
    REMOVED_COLUMNS = ["cpg_chr_pos", "sequence", "sample_name", "gene_expr"]

    mapped_dataset = dataset.map(data_preprocessor, batched=False, remove_columns=REMOVED_COLUMNS)
    return mapped_dataset


def main(_):
    data_dir = Path(FLAGS.data_dir)
    devices = FLAGS.devices

    config = OmegaConf.create(
        {
            "data": {
                "data_dir": str(data_dir),
                "cache_dir": "data/hf_cache",
                "num_proc": 12,
            },
            "train_dataloader": {
                "batch_size": 3,
                "num_workers": 3,
            },
            "val_dataloader": {
                "batch_size": 3,
                "num_workers": 3,
            },
            "model": {
                "lr": 1e-3,
            },
        }
    )
    trainer_data_module = TrainerDataModule(config)
    model = DummyMethylModel(config.model)
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=1,
        devices=devices,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        limit_train_batches=5,
    )
    trainer.fit(model, trainer_data_module)


if __name__ == "__main__":
    app.run(main)
