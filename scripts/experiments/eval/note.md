# Eval

## Commands

c2b2
- Exclude `c0103`, it raise `RuntimeError: CUDA error: uncorrectable ECC error encountered`

```bash
# On c2b2, num_processes=2, each node has 2 L40s GPUs
export num_nodes=16

export exp_name="?"
export job_name="?"
# export job_name="tcga_array_chr1-32xl40s-c2b2"
# export job_name="tcga_array_epic_chr1-32xl40s-c2b2"

export model_config_path="?"
# export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2/ckpt/version_1/config.yaml"
# export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2/ckpt/version_4/config.yaml"
export weight_path="?"
# export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2/ckpt/version_1/finished.ckpt"
# export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2/ckpt/version_4/finished.ckpt"

export dataset_flagfile="?"
export eval_save_batch_interval=2000

sbatch --exclude=c0103 --export=ALL --nodes=$num_nodes scripts/experiments/eval/eval_c2b2.sh
```


aws
```bash
export master_addr="?"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="?"
export job_name="?"

export model_config_path="?"
export weight_path="?"

export dataset_flagfile="?"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh
```



## 250123-methylfoundation-ablation-tcga_data_mix

aws
```bash
# model: tcga_array_chr1
export master_addr="172.31.21.59"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250123-methylfoundation-ablation-tcga_data_mix"
export job_name="tcga_array_chr1-32xl40s-c2b2"

export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2/ckpt/version_1/config.yaml"
export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2/ckpt/version_1/finished.ckpt"

export dataset_flagfile="src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh


# model: tcga_array_epic_chr1
export master_addr="172.31.21.59"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250123-methylfoundation-ablation-tcga_data_mix"
export job_name="tcga_array_epic_chr1-32xl40s-c2b2"

export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2/ckpt/version_4/config.yaml"
export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2/ckpt/version_4/finished.ckpt"

export dataset_flagfile="src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh



# model: tcga_array_wgbs_chr1
export master_addr="172.31.21.59"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250123-methylfoundation-ablation-tcga_data_mix"
export job_name="tcga_array_wgbs_chr1-32xl40s-c2b2"

export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_wgbs_chr1-32xl40s-c2b2/ckpt/version_4/config.yaml"
export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_wgbs_chr1-32xl40s-c2b2/ckpt/version_4/finished.ckpt"

export dataset_flagfile="src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh
```


## 250123-methylfoundation-ablation-tcga_num_chr

aws
```bash
# model tcga-mix-chr123
# data tcga_chr1
export master_addr="172.31.31.20"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250123-methylfoundation-ablation-tcga_num_chr"
export job_name="tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr1"

export model_config_path="outputs/250123-methylfoundation-ablation-tcga_num_chr/tcga-mix-chr123-bs_512-32xl40s-aws/ckpt/version_13/config.yaml"
export weight_path="outputs/250123-methylfoundation-ablation-tcga_num_chr/tcga-mix-chr123-bs_512-32xl40s-aws/ckpt/version_13/finished.ckpt"

# tcga_chr1
export dataset_flagfile="src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh



# model tcga-mix-chr123
# data tcga_chr123
export master_addr="172.31.31.20"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250123-methylfoundation-ablation-tcga_num_chr"
export job_name="tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123"

export model_config_path="outputs/250123-methylfoundation-ablation-tcga_num_chr/tcga-mix-chr123-bs_512-32xl40s-aws/ckpt/version_13/config.yaml"
export weight_path="outputs/250123-methylfoundation-ablation-tcga_num_chr/tcga-mix-chr123-bs_512-32xl40s-aws/ckpt/version_13/finished.ckpt"

# tcga_chr123
export dataset_flagfile="src/configs/cfg/data/dataset/tcga_chr123/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh


# model tcga-mix-chr1
# data tcga_chr123
export master_addr="172.31.31.20"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250123-methylfoundation-ablation-tcga_num_chr"
export job_name="tcga-mix-chr1-prev-eval_on_tcga_chr123"

export model_config_path="outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/ckpt/version_18/config.yaml"
export weight_path="outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/ckpt/version_18/finished.ckpt"

# tcga_chr123
export dataset_flagfile="src/configs/cfg/data/dataset/tcga_chr123/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh
```

## 250125-methylfoundation-encode

aws
```bash
export master_addr="172.31.21.59"
export master_port=51994
export num_nodes=4
export num_processes=8

export exp_name="250125-methylfoundation-encode"
export job_name="encode_wgbs-bs_512-64xl40s-aws"

export model_config_path="outputs/250125-methylfoundation-encode/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/config.yaml"
export weight_path="outputs/250125-methylfoundation-encode/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/finished.ckpt"

# encode
export dataset_flagfile="src/configs/cfg/data/dataset/encode/tokenized_val_dataset.cfg"
export eval_save_batch_interval=2000

bash scripts/experiments/eval/eval_aws.sh
```


## Upload and Download Eval Results from HF

Upload
```bash
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR="?"
REPO_URL="?"

# exp_name: 250123-methylfoundation-ablation-tcga_data_mix
# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/eval-tcga_array_chr1-32xl40s-c2b2/eval/version_13"
# REPO_URL="eval-tcga_array_chr1-32xl40s-c2b2"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/eval-tcga_array_epic_chr1-32xl40s-c2b2/eval/version_9"
# REPO_URL="eval-tcga_array_epic_chr1-32xl40s-c2b2"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/eval-tcga_array_wgbs_chr1-32xl40s-c2b2/eval/version_21"
# REPO_URL="eval-tcga_array_wgbs_chr1-32xl40s-c2b2"


# exp_name: 250123-methylfoundation-ablation-tcga_num_chr
# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_num_chr/eval-tcga-mix-chr1-prev-eval_on_tcga_chr123/eval/version_21"
# REPO_URL="eval-tcga-mix-chr1-prev-eval_on_tcga_chr123"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_num_chr/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr1/eval/version_13"
# REPO_URL="eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr1"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_num_chr/eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123/eval/version_11"
# REPO_URL="eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123"

# LOCAL_DIR=outputs/250125-methylfoundation-encode/eval-encode_wgbs-bs_512-64xl40s-aws/eval/version_18
# REPO_URL=eval-encode_wgbs-bs_512-64xl40s-aws

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```

Download
```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR="outputs/eval/?"

REPO_URL="xk-huang/?"

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```