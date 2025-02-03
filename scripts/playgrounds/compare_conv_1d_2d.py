import math

import torch
import torch.nn as nn
from absl import logging

from src.utils import get_model_complexity_info


def main(_=None):
    batch_size = 1
    seq_length = 10000
    input_channels = 1
    output_channels = 64
    round_sqrt_seq_length = math.ceil(seq_length**0.5)
    print(f"batch_size: {batch_size}, seq_length: {seq_length}, round_sqrt_seq_length: {round_sqrt_seq_length}")

    input_1d = torch.randn(batch_size, input_channels, seq_length)
    input_2d = torch.randn(batch_size, input_channels, round_sqrt_seq_length, round_sqrt_seq_length)

    print("Input 1D shape: {}".format(input_1d.shape))
    print("Input 2D shape: {}".format(input_2d.shape))

    conv_1d = nn.Conv1d(input_channels, output_channels, 3, padding=1)
    conv_2d = nn.Conv2d(input_channels, output_channels, 3, padding=1)

    analysis_results_1d = get_model_complexity_info(conv_1d, inputs=input_1d)
    analysis_results_2d = get_model_complexity_info(conv_2d, inputs=input_2d)
    print(analysis_results_1d["out_table"])
    print(analysis_results_2d["out_table"])
    # fmt: off
    # fmt: on


if __name__ == "__main__":
    main()
