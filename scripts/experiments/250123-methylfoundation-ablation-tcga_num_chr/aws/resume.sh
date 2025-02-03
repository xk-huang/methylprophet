#!/bin/bash
set -e



source activate base
conda activate xiaoke-methylformer
which python

if [[ -z "${master_addr}" ]]; then
    echo "master_addr is not set"
    exit 1
fi
if [[ -z "${master_port}" ]]; then
    echo "master_port is not set"
    exit 1
fi
# if num_nodes / num_processes is null, then it will be set to 1
if [[ -z "${num_nodes}" ]]; then
    num_nodes=1
fi
if [[ -z "${num_processes}" ]]; then
    num_processes=1
fi


export OMP_NUM_THREADS=10
if [[ -z "${resume_training_config_path}" ]]; then
    echo "resume_training_config_path is not set"
    exit 1
fi
if [[ -z "${ckpt_path}" ]]; then
    echo "ckpt_path is not set"
    exit 1
fi


# --rdzv_id= \
# torchrun \
torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$num_processes \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
-m src.main \
--main.resume_training_config_path="${resume_training_config_path}" --main.ckpt_path="${ckpt_path}" \
$@
