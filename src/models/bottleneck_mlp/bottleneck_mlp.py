"""
Copied from https://github.com/gregorbachmann/scaling_mlps/blob/main/models/networks.py
"""

import numpy as np
from torch import nn


NORMS = {"layer": nn.LayerNorm, "batch": nn.BatchNorm1d, "none": nn.Identity}

ACT = {"gelu": nn.GELU(), "relu": nn.ReLU()}


class StandardMLP(nn.Module):
    def __init__(self, dim_in, dim_out, widths):
        super(StandardMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.widths = widths
        self.linear_in = nn.Linear(self.dim_in, self.widths[0])
        self.linear_out = nn.Linear(self.widths[-1], self.dim_out)
        self.layers = []
        self.layer_norms = []
        for i in range(len(self.widths) - 1):
            self.layers.append(nn.Linear(self.widths[i], self.widths[i + 1]))
            self.layer_norms.append(nn.LayerNorm(widths[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.layernorms = nn.ModuleList(self.layer_norms)

    def forward(self, x):
        z = self.linear_in(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            z = norm(z)
            z = nn.GELU()(z)
            z = layer(z)

        out = self.linear_out(z)

        return out


class BottleneckMLP(nn.Module):
    def __init__(self, dim_in, dim_out, block_dims, norm="layer", checkpoint=None, name=None):
        super(BottleneckMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.block_dims = block_dims
        self.norm = NORMS[norm]
        self.checkpoint = checkpoint

        self.name = name
        self.linear_in = nn.Linear(self.dim_in, self.block_dims[0][1])
        self.linear_out = nn.Linear(self.block_dims[-1][1], self.dim_out)
        blocks = []
        layernorms = []

        for block_dim in self.block_dims:
            wide, thin = block_dim
            blocks.append(BottleneckBlock(thin=thin, wide=wide))
            layernorms.append(self.norm(thin))

        self.blocks = nn.ModuleList(blocks)
        self.layernorms = nn.ModuleList(layernorms)

        if self.checkpoint is not None:
            self.load(self.checkpoint)

    def forward(self, x):
        x = self.linear_in(x)

        for block, norm in zip(self.blocks, self.layernorms):
            x = x + block(norm(x))

        out = self.linear_out(x)

        return out

    def load(self, name, checkpoint_path="./checkpoints/"):
        raise NotImplementedError("There is no case to use it.")


class BottleneckBlock(nn.Module):
    def __init__(self, thin, wide, act=nn.GELU()):
        super(BottleneckBlock, self).__init__()

        self.block = nn.Sequential(nn.Linear(thin, wide), act, nn.Linear(wide, thin))

    def forward(self, x):
        out = self.block(x)

        return out


def B_12_Wi_1024(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 1024, 1024] for _ in range(12)]
    return BottleneckMLP(
        dim_in=dim_in,
        dim_out=dim_out,
        norm="layer",
        block_dims=block_dims,
        checkpoint=checkpoint,
        name="B_" + str(len(block_dims)) + "-Wi_" + str(block_dims[0][1]) + "_res_" + str(int(np.sqrt(dim_in / 3))),
    )


def B_12_Wi_512(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 512, 512] for _ in range(12)]
    return BottleneckMLP(
        dim_in=dim_in,
        dim_out=dim_out,
        norm="layer",
        block_dims=block_dims,
        checkpoint=checkpoint,
        name="B_" + str(len(block_dims)) + "-Wi_" + str(block_dims[0][1]) + "_res_" + str(int(np.sqrt(dim_in / 3))),
    )


def B_6_Wi_1024(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 1024, 1024] for _ in range(6)]
    return BottleneckMLP(
        dim_in=dim_in,
        dim_out=dim_out,
        norm="layer",
        block_dims=block_dims,
        checkpoint=checkpoint,
        name="B_" + str(len(block_dims)) + "-Wi_" + str(block_dims[0][1]) + "_res_" + str(int(np.sqrt(dim_in / 3))),
    )


def B_6_Wi_512(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 512, 512] for _ in range(6)]
    return BottleneckMLP(
        dim_in=dim_in,
        dim_out=dim_out,
        norm="layer",
        block_dims=block_dims,
        checkpoint=checkpoint,
        name="B_" + str(len(block_dims)) + "-Wi_" + str(block_dims[0][1]) + "_res_" + str(int(np.sqrt(dim_in / 3))),
    )


def B_18_Wi_1024(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 1024, 1024] for _ in range(18)]
    return BottleneckMLP(
        dim_in=dim_in,
        dim_out=dim_out,
        norm="layer",
        block_dims=block_dims,
        checkpoint=checkpoint,
        name="B_" + str(len(block_dims)) + "-Wi_" + str(block_dims[0][1]) + "_res_" + str(int(np.sqrt(dim_in / 3))),
    )


def B_24_Wi_1024(dim_in, dim_out, checkpoint=None):
    block_dims = [[4 * 1024, 1024] for _ in range(24)]
    return BottleneckMLP(
        dim_in=dim_in,
        dim_out=dim_out,
        norm="layer",
        block_dims=block_dims,
        checkpoint=checkpoint,
        name="B_" + str(len(block_dims)) + "-Wi_" + str(block_dims[0][1]) + "_res_" + str(int(np.sqrt(dim_in / 3))),
    )


MLP_MODEL_LIST = {
    "B_24-Wi_1024": B_24_Wi_1024,
    "B_18-Wi_1024": B_18_Wi_1024,
    "B_12-Wi_1024": B_12_Wi_1024,
    "B_12-Wi_512": B_12_Wi_512,
    "B_6-Wi_1024": B_6_Wi_1024,
    "B_6-Wi_512": B_6_Wi_512,
}


def get_mlp_model(architecture, dim_in, dim_out, checkpoint=None):
    if architecture not in MLP_MODEL_LIST:
        raise ValueError(f"Architecture {architecture} not found in mlp_model")

    return MLP_MODEL_LIST[architecture](dim_in=dim_in, dim_out=dim_out, checkpoint=checkpoint)
