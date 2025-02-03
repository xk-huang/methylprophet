# MODEL ZOO

| Model              | Data                          | Url      |
|--------------------|-------------------------------|----------|
| MethylProphet-base | ENCODE (WGBS)                 | [HF Model](https://huggingface.co/xk-huang/encode_wgbs-bs_512-64xl40s-aws) |
| MethylProphet-base | TCGA (Array+EPIC+WGBS, Chr 1) | [HF Model](https://huggingface.co/xk-huang/tcga_mix_chr1-bs_512-c2b2) |


Download the models manually or with huggingfacehub-cli:

```bash
REPO_TYPE=model # model, dataset
LOCAL_DIR=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws

REPO_URL=xk-huang/encode_wgbs-bs_512-64xl40s-aws

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}


REPO_TYPE=model # model, dataset
LOCAL_DIR=outputs/ckpts/tcga_mix_chr1-bs_512-c2b2

REPO_URL=xk-huang/tcga_mix_chr1-bs_512-c2b2

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```