"""
The config file of ml_collections is a python file that contains the configuration of the project.

https://github.com/google/ml_collections/tree/master?tab=readme-ov-file#config-flags
https://github.com/google/ml_collections/tree/master?tab=readme-ov-file#parameterising-the-get_config-function
"""

from src.config import create_config_dict


def get_config(config_type):
    BERT_CONFIG_DIST_MAPPING = {
        "debug": {
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
        },
        "small": {
            "hidden_size": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
        },
        "medium": {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 8,
        },
        "base": {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
        },
        "large": {
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
        },
    }

    model_config_dict_mapping = {}
    for bert_config_name, bert_config_dict in BERT_CONFIG_DIST_MAPPING.items():
        hidden_size = bert_config_dict["hidden_size"]
        num_attention_heads = bert_config_dict["num_attention_heads"]
        num_hidden_layers = bert_config_dict["num_hidden_layers"]

        model_config_dict = custom_create_config_dict(hidden_size, num_attention_heads, num_hidden_layers)
        model_config_dict_mapping[bert_config_name] = model_config_dict

    return model_config_dict_mapping[config_type]


def custom_create_config_dict(hidden_size, num_attention_heads, num_hidden_layers):
    model_config_dict = create_config_dict(
        model_class="MethylformerBertModel",
        model_config_class="MethylformerBertConfig",
        _attn_implementation="eager",
        # bert config
        bert_config_dict=create_config_dict(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=2048,
        ),
        # embed
        cgi_vocab_size=28010,
        dna_vocab_size=4096,
        # sample gene mlp config
        sample_gene_mlp_config_dict=create_config_dict(
            architecture="B_6-Wi_1024",
            dim_in=25052,  # change it according to the num genes
            dim_out=hidden_size,
        ),
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
    )

    return model_config_dict
