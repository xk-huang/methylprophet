import json
from collections import defaultdict
from pathlib import Path

import torch
from absl import app, flags, logging
from streaming import StreamingDataLoader, StreamingDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from src.data.data_preprocessor import DataPreprocessor


class MethylformerStreamingDataset(StreamingDataset):
    def __init__(
        self,
        *,
        data_preprocessor=None,
        group_idx_name_mapping_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_preprocessor = data_preprocessor

        if group_idx_name_mapping_path is not None:
            logging.info(f"Load group_idx_name_mapping from {group_idx_name_mapping_path}")
            with open(group_idx_name_mapping_path, "r") as f:
                self.group_idx_name_mapping = json.load(f)
        else:
            logging.warning("group_idx_name_mapping is None")
            self.group_idx_name_mapping = None

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.data_preprocessor(data)

    REMOVED_KEYS = [
        "cpg_island_tuple",
        "sequence",
    ]
    TOKENIZED_KEYS = ["tokenized_sequence", "tokenized_cgi"]

    def collate_fn(self, batch):
        tokenized_dict = defaultdict(list)
        for data in batch:
            for removed_key in self.REMOVED_KEYS:
                if removed_key in data:
                    del data[removed_key]
            for tokenized_key in self.TOKENIZED_KEYS:
                if tokenized_key in data:
                    tokenized_dict[tokenized_key].append(torch.tensor(data.pop(tokenized_key), dtype=torch.int64))

        batch = default_collate(batch)
        # NOTE: list of float to tensor, the type of tensor is float64,
        # we need to convert it to float32
        batch["methylation"] = batch["methylation"].float()

        for key, token_list in tokenized_dict.items():
            padded_input_ids, attention_mask = self.batchify_tokens(token_list)
            batch[f"{key}_input_ids"] = padded_input_ids
            batch[f"{key}_attention_mask"] = attention_mask

        return batch

    def batchify_tokens(self, list_of_tokens):
        padded_input_ids = pad_sequence(list_of_tokens, batch_first=True)
        attention_mask = torch.zeros_like(padded_input_ids, dtype=torch.int64)
        for idx, seq in enumerate(list_of_tokens):
            attention_mask[idx, : len(seq)] = 1
        return padded_input_ids, attention_mask


def create_methylformer_streaming_dataset(
    # streamingdataset args
    local=None,
    remote=None,
    batch_size=None,
    shuffle_seed=9176,
    shuffle=False,
    epoch_size=None,
    shuffle_algo="py1e",
    sampling_method="balanced",
    sampling_granularity=1,
    # streamingdataset custom args
    group_idx_name_mapping_path=None,
    # data_preprocessor args
    gene_expr_df_path=None,
    sample_idx_path=None,
    num_nbase=1000,
    gene_expr_quantization=True,
    num_gene_expr_bins=51,
    dna_tokenizer_name="zhihan1996/DNABERT-2-117M",
    is_sequence_tokenized=False,
) -> MethylformerStreamingDataset:
    data_preprocessor = DataPreprocessor(
        gene_expr_df_path=gene_expr_df_path,
        sample_idx_path=sample_idx_path,
        num_nbase=num_nbase,
        gene_expr_quantization=gene_expr_quantization,
        num_gene_expr_bins=num_gene_expr_bins,
        dna_tokenizer_name=dna_tokenizer_name,
        is_sequence_tokenized=is_sequence_tokenized,
    )

    dataset = MethylformerStreamingDataset(
        data_preprocessor=data_preprocessor,
        group_idx_name_mapping_path=group_idx_name_mapping_path,
        local=local,
        remote=remote,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        shuffle=shuffle,
        epoch_size=epoch_size,
        shuffle_algo=shuffle_algo,
        sampling_method=sampling_method,
        sampling_granularity=sampling_granularity,
    )

    logging.info(
        f"Loading dataset from [local {local}]/[remote {remote}]: "
        f"size: {dataset.size:,}; len(dataset): {len(dataset):,}"
    )

    return dataset
