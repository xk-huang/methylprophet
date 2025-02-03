# 250123-methylfoundation-ablation-tcga_data_mix

## Train

c2b2
```bash
sbatch --exclude=c0103 scripts/experiments/250123-methylfoundation-ablation-tcga_data_mix/c2b2/tcga_array_chr1.sh
sbatch --exclude=c0103 scripts/experiments/250123-methylfoundation-ablation-tcga_data_mix/c2b2/tcga_array_epic_chr1.sh
sbatch --exclude=c0103 scripts/experiments/250123-methylfoundation-ablation-tcga_data_mix/c2b2/tcga_array_wgbs_chr1.sh
```

# Eval

c2b2
- Exclude `c0103`, it raise `RuntimeError: CUDA error: uncorrectable ECC error encountered`

```bash
# On c2b2, num_processes=2, each node has 2 L40s GPUs
export num_nodes=16
export exp_name=250123-methylfoundation-ablation-tcga_data_mix
export job_name="?"
# export job_name="tcga_array_chr1-32xl40s-c2b2"
# export job_name="tcga_array_epic_chr1-32xl40s-c2b2"

export model_config_path="?"
# export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2/ckpt/version_1/config.yaml"
# export model_config_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2/ckpt/version_4/config.yaml"

export weight_path="?"
# export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2/ckpt/version_1/finished.ckpt"
# export weight_path="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2/ckpt/version_4/finished.ckpt"


export eval_save_batch_interval=2000

sbatch --exclude=c0103 --export=ALL --nodes=$num_nodes scripts/experiments/250123-methylfoundation-ablation-tcga_data_mix/c2b2/eval.sh
```

## Resume

resume
```bash
sbatch --export=ALL,resume_training_config_path=?,ckpt_path=? scripts/experiments/250123-methylfoundation-ablation-tcga_data_mix/c2b2/resume.sh
```

## Misc

Beaware of `srun torchrun` when using slurm multiple node.
If you do not append `srun` ahead, the `torchrun` just never runs.

## Upload HF

Model
```bash
REPO_TYPE=model # model, dataset
NUM_WORKERS=8

LOCAL_DIR="?"
REPO_URL="?"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2"
# REPO_URL="tcga_array_chr1-32xl40s-c2b2"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2"
# REPO_URL="tcga_array_epic_chr1-32xl40s-c2b2"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_wgbs_chr1-32xl40s-c2b2/"
# REPO_URL="tcga_array_wgbs_chr1-32xl40s-c2b2"

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```

Eval results
```bash
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR="?"
REPO_URL="?"


huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```

## Download

Model
```bash
REPO_TYPE=model

LOCAL_DIR="?"
REPO_URL="xk-huang/?"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_chr1-32xl40s-c2b2"
# REPO_URL="xk-huang/tcga_array_chr1-32xl40s-c2b2"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_epic_chr1-32xl40s-c2b2"
# REPO_URL="xk-huang/tcga_array_epic_chr1-32xl40s-c2b2"

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_data_mix/tcga_array_wgbs_chr1-32xl40s-c2b2/"
# REPO_URL="xk-huang/tcga_array_wgbs_chr1-32xl40s-c2b2"

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

Eval results
```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR="outputs/eval/?"

REPO_URL="xk-huang/?"

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```