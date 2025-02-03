#!/bin/bash
set -e

root_dir=/opt/dlami/nvme && \
base_dir=$root_dir/xiaoke/codes/methylformer && \
cd $base_dir

source activate base && \
conda activate xiaoke-methylformer

# export HF_TOKEN=hf_?
huggingface-cli whoami


REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/tar/241213-encode_wgbs

REPO_URL=xk-huang/250111_184413-241213-encode_wgbs

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}

# decompress
# prefix directory: data/parquet/241213-encode_wgbs
PARQUET_DIR=data/parquet
cat ${LOCAL_DIR}/*.tar.gz.part_??? | tar -x --use-compress-program=pigz --strip-components=2 -C ${PARQUET_DIR}
