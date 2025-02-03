import torch
from absl import app, logging

from src.utils import get_model_complexity_info
from transformers import DistilBertConfig, DistilBertModel


def main(_):
    logging.info("Analyzing MLP model")

    hidden_size = 512
    intermediate_size = hidden_size * 4
    num_hidden_layers = 8
    num_attention_heads = 8

    # hidden_size = 768
    # intermediate_size = hidden_size * 4
    # num_hidden_layers = 12
    # num_attention_heads = 12

    seq_length = 2000

    bert_config_dict = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
    }
    model_config = DistilBertConfig(
        **bert_config_dict,
    )
    model = DistilBertModel(model_config)
    model.resize_position_embeddings(seq_length)
    # NOTE xk: resizing position embeddings does not resize the position embeddings
    model.embeddings.position_ids = torch.arange(seq_length).expand((1, -1))

    batch_size = 1
    seq_length = 1004
    input_data = torch.randn(batch_size, seq_length, hidden_size)

    analysis_results = get_model_complexity_info(model, inputs=(None, None, None, input_data))
    logging.info(analysis_results["out_table"])

    output = model(inputs_embeds=input_data)
    logging.info(f"input shape: {input_data.shape}, output shape: {output.last_hidden_state.shape}")


if __name__ == "__main__":
    app.run(main)
