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


output_dir=outputs
if [[ -z "${exp_name}" ]]; then
    echo "exp_name is not set"
    exit 1
fi
if [[ -z "${job_name}" ]]; then
    echo "job_name is not set"
    exit 1
fi

job_name="eval-${job_name}"

val_num_workers=20
val_batch_size=256
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True

if [[ -z "${model_config_path}" ]]; then
    echo "model_config_path is not set"
    exit 1
fi
if [[ -z "${weight_path}" ]]; then
    echo "weight_path is not set"
    exit 1
fi

# data/mds_tokenized/241213-tcga-mix-chr123/val_new_cpg	135,167,779 / 256 / 32 = 16,499
dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr123/tokenized_val_dataset.cfg
if [[ -z "${eval_save_batch_interval}" ]]; then
    echo "eval_save_batch_interval is not set"
    exit 1
fi

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
--trainer_model.eval_save_batch_interval=${eval_save_batch_interval}
