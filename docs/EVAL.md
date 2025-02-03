# EVAL

## Inference (and Eval) a Model Checkpoint

Slurm template

```bash
# Set MASTER_ADDR to the first node's hostname
export master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

# Set a MASTER_PORT (ensure this port is free and allowed by firewall)
export master_port=32345


output_dir=outputs
exp_name=241229-methylformer_bert-ins
job_name=250110-train-tcga_chr1-base-half_data-12xl40s

job_name="eval-${job_name}"

# if num_nodes / num_process is null, then it will be set to 1
if [[ -z "${num_nodes}" ]]; then
    num_nodes=1
fi
if [[ -z "${num_process}" ]]; then
    num_process=1
fi
val_num_workers=20
# h100: 20 workers, 350w
# l40s: 10 workers, 350w

val_batch_size=256
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True

model_config_path=outputs/c2b2/250110-train-tcga_chr1-base-half_data-12xl40s/ckpt/version_0/config.yaml
weight_path=outputs/c2b2/250110-train-tcga_chr1-base-half_data-12xl40s/ckpt/version_0/finished.ckpt

dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/val_dataset.cfg

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
--trainer_model.save_eval_results=${save_eval_results} \
--trainer_model.plot_eval_results=${plot_eval_results} \
--main.model_config_path="${model_config_path}" --main.weight_path="${weight_path}"
```

### Resume eval prediction

We predict the results by chunk.
By set the `trainer_model.eval_dir` to the saved directory, we can resume predictions.

Another arg is `trainer_model.eval_save_batch_interval`, which controls the saving chunk size

## Compute metrics

```bash
input_result_csv=
input_group_idx_name_mapping_json=
output_dir=
pcc_backend=pandas
# pcc_backend=torchmetrics_cuda
python scripts/tools/eval/eval_pcc.py \
    --input_result_csv "${input_result_csv}" \
    --input_group_idx_name_mapping_json "${input_group_idx_name_mapping_json}" \
    --output_dir "${output_dir}" \
    --pcc_backend "${pcc_backend}"
```