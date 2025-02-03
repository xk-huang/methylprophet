#!/bin/bash

#SBATCH --job-name=241229-resume-train-encode-base-12xl40s
#SBATCH --output=outputs/241229-resume-train-encode-base-12xl40s.log
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120gb
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=gpu
#SBATCH --nodes=6

# same args for srun: srun --pty -t 0-03:00 -c 10 --mem=20gb --gres=gpu:L40S:2 --partition=gpu bash -i

# NOTE xk: Eval takes a lot of memory due to make statistics, make it big.

module load conda/3
source activate base
conda activate xiaoke-methylformer
which python

# Set MASTER_ADDR to the first node's hostname
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# Set a MASTER_PORT (ensure this port is free and allowed by firewall)
export MASTER_PORT=32345

# Print job information (optional, useful for debugging)
echo "Job ID: $SLURM_JOB_ID"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Hostname: $(hostname)"

output_dir=outputs
exp_name=241229-methylformer_bert-ins
job_name=241229-train-encode-base-12xl40s


# max_steps=1047173  # 1 epoch, 1608459072 / 256 / 12 = 523586
max_steps=214843  # 1 epoch, 660000000 / 256 / 12 = 214843 (Not full before 241231)
scheduler_num_training_steps=214843
scheduler_num_warmup_steps=1000
val_check_interval=2000
num_sanity_val_steps=2
learning_rate=0.0001
accumulate_grad_batches=1
weight_decay=0.00001
gradient_checkpointing=True


model=src/configs/models/methylformer_bert.py:base
use_chr_embedder=True
add_chr_embeds_type=append
add_sample_gene_embeds_type=append
_attn_implementation=flash_attention_2


use_bin_logits_cls_loss=False
use_bin_logits=False


train_batch_size=256  # base size, batch size 360 w/ grad ckpt, memory 80GB
val_batch_size=256
train_drop_last=True
train_shuffle=True
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True


export OMP_NUM_THREADS=5
NUM_PROCESS=2
train_num_workers=5
val_num_workers=5

NUM_PROCESS=2

RESUME_TRAINING_CONFIG_PATH="outputs/241229-methylformer_bert-ins/241229-train-encode-base-12xl40s/ckpt/version_29/config.yaml"
CKPT_PATH="outputs/241229-methylformer_bert-ins/241229-train-encode-base-12xl40s/ckpt/version_29/step=82000.ckpt"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$NUM_PROCESS \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
-m src.main \
--main.resume_training_config_path="${RESUME_TRAINING_CONFIG_PATH}" --main.ckpt_path="${CKPT_PATH}"

