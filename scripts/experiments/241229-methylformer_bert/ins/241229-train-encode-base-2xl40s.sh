#!/bin/bash

#SBATCH --job-name=241229-train-encode-base-2xl40s
#SBATCH --output=outputs/241229-train-encode-base-2xl40s.log
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=200gb
#SBATCH --gres=gpu:2
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --account=houlab

# same args for srun: srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=180gb --gpus=a100:4 --time=3-00:00:00 --partition=gpu --pty bash

# NOTE xk: Eval takes a lot of memory due to make statistics, make it big.

output_dir=outputs
exp_name=241229-methylformer_bert-ins
job_name=241229-train-encode-base-2xl40s


max_steps=3141521  # 1 epoch, 1608459072 / 256 / 2 = 3,141,521
scheduler_num_training_steps=3141521
scheduler_num_warmup_steps=1000
val_check_interval=10000
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


torchrun --nproc_per_node ${NUM_PROCESS} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 11454 \
-m src.main \
--trainer.devices=${NUM_PROCESS} \
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
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--main.job_name=${job_name} \
--model=${model} \
--model._attn_implementation=${_attn_implementation} \
--model.add_sample_gene_embeds_type=${add_sample_gene_embeds_type} --model.add_chr_embeds_type=${add_chr_embeds_type} --model.use_chr_embedder=${use_chr_embedder} \
--model.use_bin_logits=${use_bin_logits} --trainer_model.use_bin_logits_cls_loss=${use_bin_logits_cls_loss} \
--flagfile src/configs/cfg/data/dataset.cfg \
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
