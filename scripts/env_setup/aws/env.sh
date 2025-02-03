#!/bin/bash
set -e

source activate base
conda create -y -n xiaoke-methylformer python=3.10
source activate
conda activate xiaoke-methylformer
which python
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

root_dir=/opt/dlami/nvme
base_dir=$root_dir/xiaoke/codes/methylformer
cd $base_dir
pip install -r requirements.txt
pip install tensorboard scikit-learn
pip install packaging ninja
pip uninstall -y ninja && pip install ninja
pip install flash-attn --no-build-isolation

echo finished env installation for $(hostname)