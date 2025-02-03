export WANDB_MODE=disabled

master_addr=localhost
master_port=29500
num_nodes=1
num_process=1

output_dir=outputs
exp_name=debug
job_name=eval_52_batches_save_every_20_batches

val_num_workers=20
val_batch_size=256
num_nbase=1000

full_eval=True
plot_eval_results=True
dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg

gene_dim=25939

val_epoch_size=65536 # 256 * 256
limit_test_batches=256
eval_save_batch_interval=100
eval_dir=outputs/debug/eval/eval_256_batches_save_every_100_batches

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
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
--main.test_only \
--main.job_name=${job_name} \
--trainer_model.full_eval=${full_eval} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--trainer.limit_test_batches=${limit_test_batches} \
--trainer_model.eval_save_batch_interval=${eval_save_batch_interval} \
--trainer_model.eval_dir=${eval_dir} \
--model.sample_gene_mlp_config_dict.dim_in=${gene_dim} \
--data.val_dataset.epoch_size=${val_epoch_size}



eval_dir=outputs/debug/eval/eval_256_batches_save_every_100_batches_break_at_100

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
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
--main.test_only \
--main.job_name=${job_name} \
--trainer_model.full_eval=${full_eval} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--trainer.limit_test_batches=${limit_test_batches} \
--trainer_model.eval_save_batch_interval=${eval_save_batch_interval} \
--trainer_model.eval_dir=${eval_dir} \
--model.sample_gene_mlp_config_dict.dim_in=${gene_dim} \
--data.val_dataset.epoch_size=${val_epoch_size}

num_process=2
eval_dir=outputs/debug/eval/eval_256_batches_save_every_100_batches_gpus_2
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
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
--main.test_only \
--main.job_name=${job_name} \
--trainer_model.full_eval=${full_eval} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--trainer.limit_test_batches=${limit_test_batches} \
--trainer_model.eval_save_batch_interval=${eval_save_batch_interval} \
--trainer_model.eval_dir=${eval_dir} \
--model.sample_gene_mlp_config_dict.dim_in=${gene_dim} \
--data.val_dataset.epoch_size=${val_epoch_size}

