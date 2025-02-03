import torch
from absl import app, logging

from src.models.convnextv2_1d import ConvNextV21DConfig, ConvNextV21DModel
from src.utils import get_model_complexity_info


def main(_):
    logging.info("Analyzing MLP model")

    num_channels = 5
    # convnext_config_dict = {
    #     "num_stages": 4,
    #     "depths": [2, 2, 6, 2],
    #     "hidden_sizes": [40, 80, 160, 320],
    #     # "hidden_sizes": [64, 128, 256, 512],
    # }
    convnext_config_dict = {
        "num_channels": num_channels,
        "patch_size": 5,
        "num_stages": 2,
        "depths": [3, 3],
        # "depths": [3, 3, 9, 3],
        "hidden_sizes": [192, 384],
        # "hidden_sizes": [96, 192],
        # "hidden_sizes": [96, 192, 384, 768],
        "downsampling_in_stem": False,
    }
    model_config = ConvNextV21DConfig(
        **convnext_config_dict,
    )
    model = ConvNextV21DModel(model_config)

    batch_size = 1
    seq_length = 2000
    input_data = torch.randn(batch_size, seq_length, num_channels)

    # (batch_size, seq_length, num_channels) -> (batch_size, num_channels, seq_length)
    input_data = input_data.transpose(1, 2)
    analysis_results = get_model_complexity_info(model, inputs=input_data)
    logging.info(analysis_results["out_table"])

    output = model(input_data)
    logging.info(f"input shape: {input_data.shape}, output shape: {output.last_hidden_state.shape}")


if __name__ == "__main__":
    app.run(main)
