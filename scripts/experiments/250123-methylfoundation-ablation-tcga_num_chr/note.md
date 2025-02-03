# 250123-methylfoundation-ablation-tcga_num_chr

## Train

```bash
master_addr=172.31.17.80 \
master_port=51344 \
num_nodes=4 \
num_processes=8 \
bash scripts/experiments/250123-methylfoundation-ablation-tcga_num_chr/aws/tcga-mix-chr123-bs_512.sh




master_addr=172.31.17.80 \
master_port=51345 \
num_nodes=4 \
num_processes=8 \
bash scripts/experiments/250123-methylfoundation-ablation-tcga_num_chr/aws/tcga-mix-chr123-bs_256.sh
```

## Eval

aws
```bash
export master_addr="?"
export master_port=51994
export num_nodes=4
export num_processes=8
export exp_name=250123-methylfoundation-ablation-tcga_num_chr
export job_name="?"

export model_config_path="?"

export weight_path="?"



export eval_save_batch_interval=2000

bash scripts/experiments/250123-methylfoundation-ablation-tcga_num_chr/aws/eval.sh
```

## Resume

```bash
export master_addr="?"
export master_port=51994
export num_nodes=4
export num_processes=8

export resume_training_config_path=outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/ckpt/version_5/config.yaml
export ckpt_path=outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/ckpt/version_5/step=52000.ckpt 

bash scripts/experiments/250123-methylfoundation-ablation-tcga_num_chr/aws/resume.sh
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

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_num_chr/tcga-mix-chr123-bs_512-32xl40s-aws/"
# REPO_URL="tcga-mix-chr123-bs_512-32xl40s-aws"

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

# LOCAL_DIR="outputs/250123-methylfoundation-ablation-tcga_num_chr/tcga-mix-chr123-bs_512-32xl40s-aws/"
# REPO_URL="xk-huang/tcga-mix-chr123-bs_512-32xl40s-aws"


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