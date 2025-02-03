#!/bin/bash

#SBATCH --job-name=tcga_mix_chr1-bs_512-w_tissue_embedder-c2b2
#SBATCH --output=outputs/tcga_mix_chr1-bs_512-w_tissue_embedder-c2b2.log
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=220gb
#SBATCH --gres=gpu:L40S:2
#SBATCH --partition=gpu
#SBATCH --nodes=8

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


output_dir=outputs
exp_name=250125-methylfoundation-tcga
job_name=tcga_mix_chr1-bs_512-w_tissue_embedder-c2b2


max_steps=55524 # 1 epoch, 454,857,221 / 512 / 16 = 55,524
scheduler_num_training_steps=55524
scheduler_num_warmup_steps=2000
val_check_interval=2000
num_sanity_val_steps=2
learning_rate=0.0001
accumulate_grad_batches=1
weight_decay=0.001
gradient_checkpointing=True
betas='(0.9,0.95)'


model=src/configs/models/methylformer_bert.py:base
use_chr_embedder=True
add_chr_embeds_type=append
add_sample_gene_embeds_type=append
_attn_implementation=flash_attention_2
gene_dim=25017
use_tissue_embedder=True


use_bin_logits_cls_loss=False
use_bin_logits=False


data_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_dataset.cfg
train_batch_size=512  # base size, batch size 360 w/ grad ckpt, memory 80GB
val_batch_size=256
train_drop_last=True
train_shuffle=True
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True


export OMP_NUM_THREADS=10
train_num_workers=10
val_num_workers=10


# --rdzv_id= \
# torchrun \
srun torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$num_processes \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
-m src.main \
--trainer.devices=${num_processes} \
--trainer.num_nodes=${num_nodes} \
--trainer.accelerator=gpu \
--trainer.max_steps=${max_steps} \
--trainer.log_every_n_steps=100 \
--trainer.val_check_interval=${val_check_interval} \
--trainer.num_sanity_val_steps=${num_sanity_val_steps} \
--trainer.gradient_clip_val=1.0 \
--trainer.accumulate_grad_batches=${accumulate_grad_batches} \
--trainer.precision=bf16 \
--trainer_model.full_eval=${full_eval} \
--trainer_model.save_eval_results=${save_eval_results} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--trainer_model.scheduler_type='cosine_with_min_lr' \
--trainer_model.learning_rate=${learning_rate} \
--trainer_model.weight_decay=${weight_decay} \
--trainer_model.scheduler_num_training_steps=${scheduler_num_training_steps} \
--trainer_model.scheduler_num_warmup_steps=${scheduler_num_warmup_steps} \
--trainer_model.gradient_checkpointing=${gradient_checkpointing} \
--trainer_model.betas=${betas} \
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--main.job_name=${job_name} \
--model=${model} \
--model._attn_implementation=${_attn_implementation} \
--model.add_sample_gene_embeds_type=${add_sample_gene_embeds_type} \
--model.add_chr_embeds_type=${add_chr_embeds_type} --model.use_chr_embedder=${use_chr_embedder} \
--model.use_tissue_embedder=${use_tissue_embedder} \
--model.use_bin_logits=${use_bin_logits} --trainer_model.use_bin_logits_cls_loss=${use_bin_logits_cls_loss} \
--model.sample_gene_mlp_config_dict.dim_in=${gene_dim} \
--flagfile ${data_flagfile} \
--data.train_dataset.batch_size=${train_batch_size} \
--data.train_dataloader.batch_size=${train_batch_size} \
--data.train_dataloader.num_workers=${train_num_workers} \
--data.train_dataloader.drop_last=${train_drop_last} \
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
$@
