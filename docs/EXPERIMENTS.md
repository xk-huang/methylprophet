# EXPERIMENTS

## Reproducing Experiments

To reproduce the models in [docs/MODEL.md](./MODEL.md), run scripts in `scripts/experiments/250125-methylfoundation-tcga` and `scripts/experiments/250125-methylfoundation-encode`

To reproduce all the other experiments in our paper, run scripts in `scripts/experiments/`.

After finishing training, run eval according the [eval docs](../scripts/experiments/eval/note.md).


## Results

The prediction results:

| Model              | Data                          | Url        |
|--------------------|-------------------------------|------------|
| MethylProphet-base | ENCODE (WGBS)                 | [HF Dataset](https://huggingface.co/datasets/xk-huang/eval-encode_wgbs-bs_512-64xl40s-aws) |
| MethylProphet-base | TCGA (Array+EPIC+WGBS, Chr 1) | [HF Dataset](https://huggingface.co/datasets/xk-huang/eval-tcga_mix_chr1-bs_512-c2b2) |

Download the results manually or with huggingfacehub-cli:

```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/eval/eval-encode_wgbs-bs_512-64xl40s-aws

REPO_URL=xk-huang/eval-encode_wgbs-bs_512-64xl40s-aws

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}


REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/eval/eval-tcga_mix_chr1-bs_512-c2b

REPO_URL=xk-huang/eval-tcga_mix_chr1-bs_512-c2b

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

We provide the `cpg_idx` to `cpg_chr_pos` mapping in `cpg_chr_pos_df.parquet`, and `sample_idx` to `sample_name` and `tissue_name` in `sample_tissue_count_with_idx.csv`.

Those two files are either in `data/parquet/*/metadata/{sample_split,cpg_per_chr_stats}` or examples data ([docs/CUSTOMIZED.md](./CUSTOMIZED.md)).
