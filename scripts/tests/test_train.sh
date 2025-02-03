export WANDB_MODE=disabled

master_addr=localhost
master_port=29500
num_nodes=1
num_process=1

output_dir=outputs
exp_name=debug
job_name=fast_dev_run

val_num_workers=20
val_batch_size=256
train_num_workers=20
train_batch_size=256
num_nbase=1000

full_eval=True
plot_eval_results=True
dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg

gene_dim=25939

val_epoch_size=65536 # 256 * 256
train_epoch_size=65536 # 256 * 256
eval_save_batch_interval=100

max_steps=20
val_check_interval=10

torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$num_process \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
-m src.main \
--trainer.devices=${num_process} \
--trainer.num_nodes=${num_nodes} \
--trainer.accelerator=gpu \
--trainer.precision=bf16 \
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--flagfile=${dataset_flagfile} \
--data.train_dataset.batch_size=${train_batch_size} \
--data.train_dataloader.batch_size=${train_batch_size} \
--data.train_dataloader.num_workers=${train_num_workers} \
--data.train_dataloader.drop_last=True \
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
--main.job_name=${job_name} \
--trainer_model.full_eval=${full_eval} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--trainer_model.eval_save_batch_interval=${eval_save_batch_interval} \
--model.sample_gene_mlp_config_dict.dim_in=${gene_dim} \
--data.val_dataset.epoch_size=${val_epoch_size} \
--data.train_dataset.epoch_size=${train_epoch_size} \
--trainer.max_steps=${max_steps} \
--trainer.val_check_interval=${val_check_interval} \