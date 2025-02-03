export master_addr=localhost
export master_port=32345
export num_nodes=1
export num_processes=1 # 1 GPUs per node


output_dir=outputs
exp_name=example_data
job_name=encode_wgbs

job_name="eval-${job_name}"

val_num_workers=20
# h100: 20 workers, 350w
# l40s: 10 workers, 350w

val_batch_size=256
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True

model_config_path=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/config.yaml
weight_path=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/finished.ckpt

dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg
eval_save_batch_interval=2000

local=data/examples/encode_wgbs/val_10_shards
group_idx_name_mapping_path=data/examples/encode_wgbs/val_10_shards/group_idx_name_mapping.json
gene_expr_df_path=data/examples/encode_wgbs/gene_expr.filtered.parquet
sample_idx_path=data/examples/encode_wgbs/sample_tissue_count_with_idx.csv


torchrun \
    --nnodes=$num_nodes \
    --nproc_per_node=$num_processes \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
-m src.main \
--trainer.devices=${num_processes} \
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
--trainer_model.save_eval_results=${save_eval_results} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--main.model_config_path="${model_config_path}" --main.weight_path="${weight_path}" \
--trainer_model.eval_save_batch_interval=${eval_save_batch_interval} \
--data.train_dataset.local=${local} \
--data.val_dataset.local=${local} \
--data.train_dataset.group_idx_name_mapping_path=${group_idx_name_mapping_path} \
--data.val_dataset.group_idx_name_mapping_path=${group_idx_name_mapping_path} \
--data.train_dataset.gene_expr_df_path=${gene_expr_df_path} \
--data.val_dataset.gene_expr_df_path=${gene_expr_df_path} \
--data.train_dataset.sample_idx_path=${sample_idx_path} \
--data.val_dataset.sample_idx_path=${sample_idx_path}
