#!/bin/bash
set -e

root_dir=/opt/dlami/nvme && \
base_dir=$root_dir/xiaoke/codes/methylformer && \
cd $base_dir

source activate base && \
conda activate xiaoke-methylformer

NUM_WORKERS=120 bash scripts/data_preprocessing/241213-encode.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-encode.sh
