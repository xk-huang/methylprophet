#!/bin/bash

#SBATCH --job-name=resume
#SBATCH --output=outputs/resume.log
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=220gb
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=gpu
#SBATCH --nodes=16

# same args for srun: srun --pty -t 0-03:00 -c 10 --mem=20gb --gres=gpu:L40S:2 --partition=gpu bash -i

# NOTE xk: Eval takes a lot of memory due to make statistics, make it big.
set -e


module load conda/3
source activate base
conda activate xiaoke-methylformer
which python

# source activate base
# conda activate xiaoke-methylformer
# which python

# if [[ -z "${master_addr}" ]]; then
#     echo "master_addr is not set"
#     exit 1
# fi
# if [[ -z "${master_port}" ]]; then
#     echo "master_port is not set"
#     exit 1
# fi
# # if num_nodes / num_processes is null, then it will be set to 1
# if [[ -z "${num_nodes}" ]]; then
#     num_nodes=1
# fi
# if [[ -z "${num_processes}" ]]; then
#     num_processes=1
# fi
export master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export master_port=32345
export num_nodes=$SLURM_NNODES
export num_processes=2 # 2 GPUs per node


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
srun torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$num_processes \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
-m src.main \
--main.resume_training_config_path="${resume_training_config_path}" --main.ckpt_path="${ckpt_path}" \
$@
