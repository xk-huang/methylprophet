# 250122-methylfoundation-ablation-tissue_embedder

## Eval

aws
```bash
export master_addr="?"
export master_port=51894
export num_nodes=4
export num_processes=8
export exp_name=250122-methylfoundation-ablation-tissue_embedder
export job_name=wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

export model_config_path="outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/ckpt/version_18/config.yaml"

export weight_path="outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/ckpt/version_18/finished.ckpt"

# "outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/ckpt/version_18/step=50000.ckpt"


export eval_save_batch_interval=2000

bash scripts/experiments/250122-methylfoundation-ablation-tissue_embedder/aws/eval.sh
```

c2b2
- Exclude `c0103`, it raise `RuntimeError: CUDA error: uncorrectable ECC error encountered`

```bash
# On c2b2, num_processes=2, each node has 2 L40s GPUs
export num_nodes=16
export exp_name=250122-methylfoundation-ablation-tissue_embedder
export job_name=w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2

export model_config_path="outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/ckpt/version_5/config.yaml"

export weight_path="outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/ckpt/version_5/finished.ckpt"


export eval_save_batch_interval=2000

sbatch --exclude=c0103 --export=ALL --nodes=$num_nodes scripts/experiments/250122-methylfoundation-ablation-tissue_embedder/c2b2/eval.sh
```

## Resume

resume
```bash
sbatch --export=ALL,resume_training_config_path=outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/ckpt/version_5/config.yaml,ckpt_path=outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/ckpt/version_5/step=52000.ckpt bash scripts/experiments/250122-methylfoundation-ablation-tissue_embedder/c2b2/resume.sh
```

## Misc

Beaware of `srun torchrun` when using slurm multiple node.
If you do not append `srun` ahead, the `torchrun` just never runs.

## Upload

aws
```bash
REPO_TYPE=model # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws
REPO_URL=wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"



REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs/250122-methylfoundation-ablation-tissue_embedder/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/eval/version_19
REPO_URL=eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```


c2b2
```bash
REPO_TYPE=model # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/
REPO_URL=w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"



REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs/250122-methylfoundation-ablation-tissue_embedder/eval-w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2/eval/version_1/
REPO_URL=eval-w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```

## Download

Model
```bash
REPO_TYPE=model

LOCAL_DIR="?"
REPO_URL="xk-huang/?"

# LOCAL_DIR="outputs/250122-methylfoundation-ablation-tissue_embedder/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws"
# REPO_URL="xk-huang/wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws"

# LOCAL_DIR="outputs/250122-methylfoundation-ablation-tissue_embedder/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2"
# REPO_URL="xk-huang/w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2"


mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

Eval results
```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

REPO_URL=xk-huang/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}



REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/eval/eval-w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2

REPO_URL=xk-huang/eval-w_tissue_embedder-tcga_mix_chr1-32xl40s-c2b2

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}



# cpg, sample, tissue idx mapping
tar -czf outputs/eval/idx_to_name.250123.tar.gz \
    --transform 's/^data\/processed\///' \
    --transform 's/^data\/parquet\///' \
    data/processed/241231-tcga/sample_tissue_count_with_idx.csv \
    data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/sample_tissue_count_with_idx.csv \
    data/parquet/241213-encode_wgbs/metadata/cpg_per_chr_stats/cpg_chr_pos_df.parquet \
    data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix/cpg_chr_pos_df.parquet
```