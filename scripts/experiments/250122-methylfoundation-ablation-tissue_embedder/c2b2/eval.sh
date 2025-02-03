#!/bin/bash

#SBATCH --job-name=eva
#SBATCH --output=outputs/eval.log
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


output_dir=outputs
# exp_name=exp_name
if [[ -z "${exp_name}" ]]; then
    echo "exp_name is not set"
    exit 1
fi
# job_name=250116-train-tcga_chr1-base-wd_1e_5-half_data-16xl40s
if [[ -z "${job_name}" ]]; then
    echo "job_name is not set"
    exit 1
fi

job_name="eval-${job_name}"

val_num_workers=20
# h100: 20 workers, 350w
# l40s: 10 workers, 350w

val_batch_size=256
num_nbase=1000


full_eval=True
save_eval_results=True
plot_eval_results=True

# model_config_path=outputs/241229-methylformer_bert-ins/250116-train-tcga_chr1-base-wd_1e_5-half_data-16xl40s/ckpt/version_1/config.yaml
# weight_path=outputs/241229-methylformer_bert-ins/250116-train-tcga_chr1-base-wd_1e_5-half_data-16xl40s/ckpt/version_1/finished.ckpt
if [[ -z "${model_config_path}" ]]; then
    echo "model_config_path is not set"
    exit 1
fi
if [[ -z "${weight_path}" ]]; then
    echo "weight_path is not set"
    exit 1
fi

dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg
# data/mds_tokenized/241213-tcga-mix-chr1/val: 91,923,132 / 256 / 32 = 11,221
# eval_save_batch_interval=2000
if [[ -z "${eval_save_batch_interval}" ]]; then
    echo "eval_save_batch_interval is not set"
    exit 1
fi

srun torchrun \
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
