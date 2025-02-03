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
# if num_nodes / num_process is null, then it will be set to 1
if [[ -z "${num_nodes}" ]]; then
    num_nodes=1
fi
if [[ -z "${num_process}" ]]; then
    num_process=1
fi


output_dir=outputs
exp_name=241229-methylformer_bert-ins
job_name=250116-train-tcga_chr1-base-b12_wi1024_mlp-wd_1e_5-16xl40s


max_steps=111049  # 1 epoch, 454,860,485 / 256 / 12 = 111,049
scheduler_num_training_steps=111049
scheduler_num_warmup_steps=2000
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
gene_dim=25939
gene_mlp_architecture=B_12-Wi_1024


use_bin_logits_cls_loss=False
use_bin_logits=False


data_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_dataset.cfg
train_batch_size=256  # base size, batch size 360 w/ grad ckpt, memory 80GB
val_batch_size=256
train_drop_last=True
train_shuffle=True
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True


export OMP_NUM_THREADS=5
train_num_workers=5
val_num_workers=5


# --rdzv_id= \
torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$num_process \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
-m src.main \
--trainer.devices=${num_process} \
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
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--main.job_name=${job_name} \
--model=${model} \
--model._attn_implementation=${_attn_implementation} \
--model.add_sample_gene_embeds_type=${add_sample_gene_embeds_type} --model.add_chr_embeds_type=${add_chr_embeds_type} --model.use_chr_embedder=${use_chr_embedder} \
--model.use_bin_logits=${use_bin_logits} --trainer_model.use_bin_logits_cls_loss=${use_bin_logits_cls_loss} \
--model.sample_gene_mlp_config_dict.dim_in=${gene_dim} \
--model.sample_gene_mlp_config_dict.architecture=${gene_mlp_architecture} \
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
