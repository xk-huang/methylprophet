import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from absl import logging
from transformers import AutoTokenizer

from src.data.constants import CHR_IDX_MAPPING


class DataPreprocessor:
    def __init__(
        self,
        gene_expr_df_path=None,
        sample_idx_path=None,
        num_nbase=1000,
        gene_expr_quantization=True,
        num_gene_expr_bins=51,
        dna_tokenizer_name="zhihan1996/DNABERT-2-117M",
        is_sequence_tokenized=False,
        **kwargs,
    ):
        # check args
        if gene_expr_df_path is None:
            raise ValueError("gene_expr_df_path is None")
        self.gene_expr_df = pd.read_parquet(gene_expr_df_path)
        if sample_idx_path is None:
            raise ValueError("sample_idx_path is None")
        sample_idx_df = pd.read_csv(sample_idx_path)
        self.sample_idx_to_name = pd.Series(
            sample_idx_df["sample_name"].values, index=sample_idx_df["sample_idx"]
        ).to_dict()

        self.num_nbase = num_nbase
        self.gene_expr_quantization = gene_expr_quantization
        self.num_gene_expr_bins = num_gene_expr_bins
        self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer_name, trust_remote_code=True)
        self.cgi_tokenizer = CGITokenizer()
        self.is_sequence_tokenized = is_sequence_tokenized

        _config = {
            "num_nbase": num_nbase,
            "gene_expr_quantization": gene_expr_quantization,
            "num_gene_expr_bins": num_gene_expr_bins,
            "dna_tokenizer_name": dna_tokenizer_name,
            "gene_expr_df_path": gene_expr_df_path,
        }
        logging.info(f"DataPreprocessor: {_config}")
        self.normalize_gene_expr_df()
        self.check_gene_expr_df()

    def normalize_gene_expr_df(self):
        if not self.gene_expr_quantization:
            self.gene_expr_df = (self.gene_expr_df - self.gene_expr_df.min()) / (
                self.gene_expr_df.max() - self.gene_expr_df.min()
            )
            logging.info("Normalized gene expression values to [0, 1] with min-max scaling")
        else:
            logging.info("Quantize and normalize gene expression to [0, 1]. Skip first normalization.")

    def check_gene_expr_df(self):
        if self.gene_expr_df.isnull().values.any():
            raise ValueError("NaN values in gene expression dataframe")

    def __call__(self, data):
        self.check_me(data)

        data = self.add_gene_expr_to_data(data)
        data = self.tokenize_nbase(data)
        data = self.add_chr_idx(data)

        if self.gene_expr_quantization:
            data = self.quantize_and_normalize_gene_expr(data)
        data = self.tokenize_cgi(data)
        return data

    def check_me(self, data):
        if data["methylation"] is np.nan:
            raise ValueError("NaN values in methylation")
        if data["methylation"] is None:
            raise ValueError("None values in methylation")

    def add_gene_expr_to_data(self, data):
        # [NOTE] sample_idx is not the same as the sample (column) index of gene_expr_df. We sort the column (column) name to build the sample_idx
        sample_idx = data["sample_idx"]
        sample_name = self.sample_idx_to_name[sample_idx]
        gene_expr = self.gene_expr_df[sample_name]
        data["gene_expr"] = gene_expr.to_numpy().astype("float32")  # NOTE xk: float32
        return data

    def tokenize_nbase(self, data):
        if self.is_sequence_tokenized:
            return data

        cpg_nbase_seq = data["sequence"]

        # NOTE xk: use the middle part of the sequence according to num_nbase
        if len(cpg_nbase_seq) < self.num_nbase:
            raise ValueError(f"sequence length is less than num_nbase: {len(cpg_nbase_seq)} < {self.num_nbase}")
        mid_pos = len(cpg_nbase_seq) // 2
        left_pos = mid_pos - self.num_nbase // 2
        right_pos = mid_pos + self.num_nbase // 2

        if left_pos < 0:
            raise ValueError(f"left_pos is less than 0: {left_pos}")
        if right_pos > len(cpg_nbase_seq):
            raise ValueError(f"right_pos is greater than sequence length: {right_pos} > {len(cpg_nbase_seq)}")

        cpg_nbase_seq = cpg_nbase_seq[left_pos:right_pos]

        cpg_nbase_seq = cpg_nbase_seq.upper()
        cpg_nbase_token_seq = self.dna_tokenizer.encode(cpg_nbase_seq, add_special_tokens=False)

        data["tokenized_sequence"] = cpg_nbase_token_seq
        return data

    def add_chr_idx(self, data):
        if "chr_idx" in data:
            return data

        cpg_chr_pos = data["cpg_chr_pos"]
        chr = cpg_chr_pos.split("_")[0]
        chr = chr.lower()
        if chr not in CHR_IDX_MAPPING:
            raise ValueError(f"Unknown chromosome: {chr} in {CHR_IDX_MAPPING.keys()}")
        data["chr_idx"] = CHR_IDX_MAPPING[chr]
        return data

    def quantize_and_normalize_gene_expr(self, data):
        # NOTE xk: scGPT takes binning techniques: quantize the values from 0-max per cell
        # https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/preprocess.py#L274
        gene_expr = data["gene_expr"]

        # Get quantile
        non_zero_gene_expr_slice = gene_expr > 0
        non_zero_gene_expr = gene_expr[non_zero_gene_expr_slice]
        bins = np.quantile(
            non_zero_gene_expr, np.linspace(0, 1, self.num_gene_expr_bins - 1)
        )  # NOTE xk: on zero values are quantized to [1 - self.num_gene_expr_bins - 1]
        quant_non_zero_gene_expr = self.quantize(non_zero_gene_expr, bins)
        quant_gene_expr = np.zeros_like(gene_expr, dtype=np.int64)
        quant_gene_expr[non_zero_gene_expr_slice] = quant_non_zero_gene_expr

        # Normalize to [0, 1]
        quant_gene_expr = quant_gene_expr / quant_gene_expr.max()

        data["gene_expr"] = quant_gene_expr.astype("float32")  # NOTE xk: float32
        return data

    @staticmethod
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

    def tokenize_cgi(self, data):
        cpg_island_tuple = data["cpg_island_tuple"]
        tokenized_cgi = self.cgi_tokenizer(cpg_island_tuple)
        data["tokenized_cgi"] = tokenized_cgi
        return data


class CGITokenizer:
    def __init__(
        self,
        num_cgi=28000,
    ):
        # NOTE: should be 1-based, thus +1 in the input.
        # add nan as the first token.
        self.num_cgi = num_cgi

        # NOTE: there should be no 0 index in `cpg_island_stats.json`
        # The index should be 1-based
        self.no_cgi_token_id = 0
        self.sep_token_id = num_cgi + 1

        self.cgi_types = [
            "cgi",
            "sea",
            "shelve",
            "upshore1",
            "upshore2",
            "upshore3",
            "upshore4",
        ]
        self.cgi_type_mapping = {loc: idx + 1 + self.sep_token_id for idx, loc in enumerate(self.cgi_types)}
        self.num_cgi_types = len(self.cgi_type_mapping)
        self.vocab_size = self.num_cgi + self.num_cgi_types + 1

    def __call__(self, cpg_island_tuple_list):
        # cpg_island_tuple_list: sea_-1,upshore2_12
        cpg_island_tuple_list = cpg_island_tuple_list.split(",")

        # format: <sep> <cgi_type> <cgi_index> <sep> ...
        tokenized_cgi = [self.sep_token_id]
        for cpg_island_tuple in cpg_island_tuple_list:
            tokenized_cgi.extend(self.encode(cpg_island_tuple) + [self.sep_token_id])
        return tokenized_cgi

    def encode(self, cpg_island_tuple):
        # cpg_island_tuple: sea_-1
        cgi_type, cgi_index_token_id = cpg_island_tuple.split("_")
        cgi_type_token_id = self.cgi_type_mapping[cgi_type]

        cgi_index_token_id = int(cgi_index_token_id)
        if cgi_index_token_id > self.num_cgi or cgi_index_token_id == 0 or cgi_index_token_id < -1:
            raise ValueError(f"Invalid CGI index: {cgi_index_token_id}, should be -1 or 1-{self.num_cgi}")

        if cgi_index_token_id == -1:
            cgi_index_token_id = self.no_cgi_token_id

        return [cgi_index_token_id, cgi_type_token_id]
