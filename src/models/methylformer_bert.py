from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from absl import logging

from src.models.bottleneck_mlp import get_mlp_model
from transformers import DistilBertConfig, DistilBertModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput


class MethylformerBertConfig(PretrainedConfig):
    model_type = "methylformer_bert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        _attn_implementation="eager",
        # bert config
        bert_config_dict=None,
        # embed
        cgi_vocab_size=20000,
        dna_vocab_size=4096,
        # sample gene mlp config
        sample_gene_mlp_config_dict=None,
        add_sample_gene_embeds_type="append",
        # chr embedder
        use_chr_embedder=False,
        num_chr_embeds=24,
        add_chr_embeds_type="append",
        # tissue embedder
        use_tissue_embedder=False,
        num_tissue_embeds=100,
        # bin logits
        use_bin_logits=False,
        num_bins=101,
        bin_min_val=0.0,
        bin_max_val=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._attn_implementation = _attn_implementation

        # embed
        self.cgi_vocab_size = cgi_vocab_size
        self.dna_vocab_size = dna_vocab_size

        # bert config
        if bert_config_dict is None:
            bert_config_dict = {
                "hidden_size": 256,
                "intermediate_size": 1024,
                "num_attention_heads": 4,
                "num_hidden_layers": 4,
            }
        bert_config_dict["vocab_size"] = 1  # NOTE: do not use the embed of bert
        self.bert_config_dict = bert_config_dict

        # sample gene mlp config
        if sample_gene_mlp_config_dict is None:
            sample_gene_mlp_config_dict = {
                "architecture": "B_6-Wi_1024",
                "dim_in": 20000,
                "dim_out": 256,
            }
        self.sample_gene_mlp_config_dict = sample_gene_mlp_config_dict
        self.add_sample_gene_embeds_type = add_sample_gene_embeds_type

        # chr embedder
        self.use_chr_embedder = use_chr_embedder
        self.num_chr_embeds = num_chr_embeds  # NOTE: 1-22, X, Y
        self.add_chr_embeds_type = add_chr_embeds_type  # NOTE: append, per_token_add

        # tissue embedder
        self.use_tissue_embedder = use_tissue_embedder
        self.num_tissue_embeds = num_tissue_embeds

        # bin logits
        self.use_bin_logits = use_bin_logits
        self.num_bins = num_bins
        self.bin_min_val = bin_min_val
        self.bin_max_val = bin_max_val

    ADD_CHR_EMBEDS_TYPES = ["append", "per_token_add"]

    def __post_init__(self):
        super().__post_init__()

        if self.add_chr_embeds_type not in self.ADD_CHR_EMBEDS_TYPES:
            raise ValueError(
                f"`chr_embedder_type` must be one of {self.ADD_CHR_EMBEDS_TYPES}. " f"Got {self.add_chr_embeds_type}"
            )


@dataclass
class MethylformerBertModelOutput(ModelOutput):
    output_value: torch.FloatTensor = None
    output_bin_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MethylformerBertModel(nn.Module):
    def __init__(self, config: MethylformerBertConfig):
        super().__init__()

        self.config = config
        logging.info(f"config: {config}")

        # Build bert
        bert_config_dict = self.config.bert_config_dict
        bert_config = DistilBertConfig(
            **bert_config_dict,
            _attn_implementation=config._attn_implementation,
        )
        self.bert = DistilBertModel(bert_config)

        # Build CLS embedder
        cls_embed = torch.nn.Embedding(1, bert_config.dim)
        cls_embed.weight.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.cls_embed = cls_embed

        # Build chr embedder
        self.chr_embedder = nn.Embedding(self.config.num_chr_embeds, bert_config.dim)
        self.use_chr_embedder = config.use_chr_embedder
        self.add_chr_embeds_type = config.add_chr_embeds_type
        logging.info(f"Use Chr Embedder: {self.use_chr_embedder}")
        if self.use_chr_embedder:
            logging.info(f"Chr Embedder: {self.chr_embedder}")
            logging.info(f"Add Chr Embeds Type: {self.add_chr_embeds_type}")

        # Build tissue embedder
        self.tissue_embedder = nn.Embedding(config.num_tissue_embeds, bert_config.dim)
        self.use_tissue_embedder = config.use_tissue_embedder
        logging.info(f"Use Tissue Embedder: {self.use_tissue_embedder}")
        if self.use_tissue_embedder:
            logging.info(f"Tissue Embedder: {self.tissue_embedder}")

        # Build dna embedder
        self.dna_embedder = nn.Embedding(config.dna_vocab_size, bert_config.dim)

        # Build cgi embedder
        self.cgi_embedder = nn.Embedding(config.cgi_vocab_size, bert_config.dim)

        # Build sample gene mlp
        sample_gene_mlp_config_dict = self.config.sample_gene_mlp_config_dict
        self.add_sample_gene_embeds_type = config.add_sample_gene_embeds_type
        self.sample_gene_mlp = get_mlp_model(**sample_gene_mlp_config_dict)

        # Regressor
        self.methylation_regressor = None
        self.methylation_bin_classifier = None
        self.use_bin_logits = config.use_bin_logits
        if not config.use_bin_logits:
            logging.info("Use Methylation Regressor")
            # NOTE xk: use nn.Linear + nn.Sigmoid for regression
            self.methylation_regressor = nn.Sequential(nn.Linear(bert_config.dim, 1), nn.Sigmoid())
            self.bin_values = None
        else:
            num_bins = config.num_bins
            bin_min_val = config.bin_min_val
            bin_max_val = config.bin_max_val
            logging.info(
                f"Use Methylation Bin Classifier: num_bins={config.num_bins}, bin_min_val={bin_min_val}, bin_max_val={bin_max_val}"
            )
            self.methylation_bin_classifier = nn.Linear(in_features=bert_config.dim, out_features=num_bins)
            bin_values = torch.linspace(bin_min_val, bin_max_val, config.num_bins)
            self.bin_values: torch.Tensor
            self.register_buffer("bin_values", bin_values, persistent=False)

    def forward(
        self,
        gene_expr,
        tokenized_sequence_input_ids,
        tokenized_sequence_attention_mask,
        tokenized_cgi_input_ids,
        tokenized_cgi_attention_mask,
        chr_idx,
        tissue_idx,
    ):
        """
        'gene_expr': torch.Size([1, 25052]),
        'tokenized_cgi_attention_mask': torch.Size([1, 4]),
        'tokenized_cgi_input_ids': torch.Size([1, 4]),
        'tokenized_sequence_attention_mask': torch.Size([1, 382]),
        'tokenized_sequence_input_ids': torch.Size([1, 382])
        'chr_idx': torch.Size([1])
        'tissue_idx': torch.Size([1])
        """
        sequence_input_embeds = self.dna_embedder(tokenized_sequence_input_ids)
        cgi_input_embeds = self.cgi_embedder(tokenized_cgi_input_ids)

        # gene_expr: (batch_size, gene_expr_dim)
        sample_gene_expr_embeds = self.sample_gene_mlp(gene_expr)
        # sample_gene_expr_embeds: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        sample_gene_expr_embeds = sample_gene_expr_embeds.unsqueeze(1)

        # NOTE: cls_embed, (1, hidden_size) -> (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        batch_size = sequence_input_embeds.shape[0]
        prefix_embeds = self.cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        if self.use_chr_embedder:
            # NOTE: chr_embeds, (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
            chr_embeds = self.chr_embedder(chr_idx)
            chr_embeds = chr_embeds.unsqueeze(1)
            if self.add_chr_embeds_type == "append":
                prefix_embeds = torch.cat((prefix_embeds, chr_embeds), dim=1)
            elif self.add_chr_embeds_type == "per_token_add":
                sequence_input_embeds = sequence_input_embeds + chr_embeds
            else:
                raise ValueError(f"Invalid add_chr_embeds_type: {self.add_chr_embeds_type}")

        if self.use_tissue_embedder:
            # NOTE tissue_embeds, (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
            tissue_embeds = self.tissue_embedder(tissue_idx)
            tissue_embeds = tissue_embeds.unsqueeze(1)
            prefix_embeds = torch.cat((prefix_embeds, tissue_embeds), dim=1)

        if self.add_sample_gene_embeds_type == "append":
            prefix_embeds = torch.cat((prefix_embeds, sample_gene_expr_embeds), dim=1)
        elif self.add_sample_gene_embeds_type == "per_token_add":
            sequence_input_embeds = sequence_input_embeds + sample_gene_expr_embeds
        else:
            raise ValueError(f"Invalid add_sample_embeds_type: {self.add_sample_gene_embeds_type}")

        input_embeds = torch.cat((prefix_embeds, cgi_input_embeds, sequence_input_embeds), dim=1)
        prefix_embeds_attention_mask = torch.ones(
            (batch_size, prefix_embeds.shape[1]), device=input_embeds.device, dtype=torch.long
        )
        attention_mask = torch.cat(
            (prefix_embeds_attention_mask, tokenized_cgi_attention_mask, tokenized_sequence_attention_mask),
            dim=1,
        )

        outputs = self.bert(inputs_embeds=input_embeds, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token_from_last_hidden_state = last_hidden_state[:, 0]

        output_value = self.methylation_regressor(cls_token_from_last_hidden_state)

        if not self.use_bin_logits:
            output_value = self.methylation_regressor(cls_token_from_last_hidden_state)
            output_bin_logits = None
        else:
            output_bin_logits = self.methylation_bin_classifier(cls_token_from_last_hidden_state)
            # NOTE: compute expected for logits
            output_bin_prob = torch.softmax(output_bin_logits, dim=1)
            output_value = torch.sum(output_bin_prob * self.bin_values, dim=1, keepdim=True)

        return MethylformerBertModelOutput(
            output_value=output_value,
            output_bin_logits=output_bin_logits,
            last_hidden_state=last_hidden_state,
        )
