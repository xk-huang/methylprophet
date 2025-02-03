import torch
from absl import app, logging

from src.models.bottleneck_mlp import get_mlp_model
from src.utils import get_model_complexity_info


def main(_):
    logging.info("Analyzing MLP model")
    dim_in = 17000
    out_dim = 512
    # model_name = "B_6-Wi_512"
    model_name = "B_6-Wi_1024"
    # model_name = "B_12-Wi_1024"
    model = get_mlp_model(model_name, dim_in, out_dim)

    batch_size = 1
    input_data = torch.randn(batch_size, dim_in)

    analysis_results = get_model_complexity_info(model, inputs=input_data)
    logging.info(analysis_results["out_table"])


if __name__ == "__main__":
    app.run(main)
