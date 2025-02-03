# HF

## Set up

```shell
pip install -U huggingface_hub hf_transfer

git config --global credential.helper store

# Login huggingface hub
# huggingface-cli login
# export HF_TOKEN=hf_YOUR_TOKEN

# Check login
huggingface-cli whoami
```


## Upload to Huggingface Hub

```shell
# Speed up with hf_transfer
# export HF_HUB_ENABLE_HF_TRANSFER=1 

REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=
REPO_URL=

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```

If you encounter `Failed to preupload LFS: Error while uploading huggingface`, then disable `hf_transfer` by `HF_HUB_ENABLE_HF_TRANSFER=0`.

```shell
# Single file
REPO_TYPE=dataset # model, dataset
NUM_WORKERS=1

FILE_PATH=outputs/241229-methylformer_bert-ins/eval-241229-train-encode-base-12xl40s/eval/eval-241229-train-encode-base-12xl40s.tar.gz
REPO_URL=xk-huang/$(date +%y%m%d-%H%M%S)-$(basename FILE_PATH)
huggingface-cli upload "$REPO_URL" --repo-type="${REPO_TYPE}" ${FILE_PATH}
```

## Download from HF

```shell
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/ckpts

REPO_URL=

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```


## Compress


### Raw

```bash
SAVE_DIR_NAME=241231-tcga
DATED_SAVE_DIR_NAME=$(date +%y%m%d_%H%M%S)-${SAVE_DIR_NAME}-raw
mkdir -p data/tar/${DATED_SAVE_DIR_NAME}
tar -c --use-compress-program=pigz data/extracted/${SAVE_DIR_NAME}* | \
split -b 500M -d --suffix-length=3 - "data/tar/${DATED_SAVE_DIR_NAME}/${SAVE_DIR_NAME}.tar.gz.part_"

mkdir -p data/tar
tar -cvf data/tar/$(date +%y%m%d_%H%M%S)-tcga_cpg-raw.tar.gz -T <(find data/extracted/ -name 'note*')


SAVE_DIR_NAME=241213-encode
DATED_SAVE_DIR_NAME=$(date +%y%m%d_%H%M%S)-${SAVE_DIR_NAME}-raw
mkdir -p data/tar/${DATED_SAVE_DIR_NAME}
tar -c --use-compress-program=pigz data/extracted/${SAVE_DIR_NAME}* | \
split -b 500M -d --suffix-length=3 - "data/tar/${DATED_SAVE_DIR_NAME}/${SAVE_DIR_NAME}.tar.gz.part_"
```

### Parquet
```
30G     data/parquet/241213-encode_wgbs
3.3G    data/parquet/241231-tcga
56G     data/parquet/241231-tcga_array
18G     data/parquet/241231-tcga_epic
6.1G    data/parquet/241231-tcga_wgbs
1.5G    data/parquet/grch38_hg38
```


TCGA
```bash
SAVE_DIR_NAME=241231-tcga
DATED_SAVE_DIR_NAME=$(date +%y%m%d_%H%M%S)-${SAVE_DIR_NAME}
mkdir -p data/compressed/${DATED_SAVE_DIR_NAME}
tar -c --use-compress-program=pigz data/parquet/${SAVE_DIR_NAME}* | \
split -b 500M -d --suffix-length=3 - "data/compressed/${DATED_SAVE_DIR_NAME}/${SAVE_DIR_NAME}.tar.gz.part_"
```

ENCODE
```bash
SAVE_DIR_NAME=241213-encode_wgbs
DATED_SAVE_DIR_NAME=$(date +%y%m%d_%H%M%S)-${SAVE_DIR_NAME}
mkdir -p data/compressed/${DATED_SAVE_DIR_NAME}
tar -c --use-compress-program=pigz data/parquet/${SAVE_DIR_NAME}* | \
split -b 500M -d --suffix-length=3 - "data/compressed/${DATED_SAVE_DIR_NAME}/${SAVE_DIR_NAME}.tar.gz.part_"
```

grch38_hg38
```bash
SAVE_DIR_NAME=grch38_hg38
DATED_SAVE_DIR_NAME=$(date +%y%m%d_%H%M%S)-${SAVE_DIR_NAME}
mkdir -p data/compressed/${DATED_SAVE_DIR_NAME}
tar -c --use-compress-program=pigz data/parquet/${SAVE_DIR_NAME}* | \
split -b 500M -d --suffix-length=3 - "data/compressed/${DATED_SAVE_DIR_NAME}/${SAVE_DIR_NAME}.tar.gz.part_"
```

eval_results-test.csv
```bash
EVAL_DIR=outputs/241229-methylformer_bert-ins/eval-241229-train-encode-base-12xl40s/eval/version_12
SAVE_DIR=${EVAL_DIR}/tar
mkdir -p ${SAVE_DIR}
tar -c --use-compress-program=pigz ${EVAL_DIR}/group_idx_name_mapping-test.json ${EVAL_DIR}/eval_results-test.csv | \
split -b 500M -d --suffix-length=3 - "$SAVE_DIR/eval_results-test.tar.gz.part_"
```

## decompress
```bash
# extract split tar files
cat data/raw/tcga_tar_gz/tcga-241213_150726.tar.gz.part_??? | tar -x --use-compress-program=pigz -C data/raw/tcga_tar_gz-decompressed
```


### Mapping

```bash
# sample idx to names
data/processed/241231-tcga/sample_tissue_count_with_idx.csv
data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/sample_tissue_count_with_idx.csv

# cpg idx to names
data/parquet/241213-encode_wgbs/metadata/cpg_per_chr_stats/cpg_chr_pos_df.parquet
data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix/cpg_chr_pos_df.parquet

tar -czf misc/idx_to_name.tar.gz \
    --transform 's/^data\/processed\///' \
    --transform 's/^data\/parquet\///' \
    data/processed/241231-tcga/sample_tissue_count_with_idx.csv \
    data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/sample_tissue_count_with_idx.csv \
    data/parquet/241213-encode_wgbs/metadata/cpg_per_chr_stats/cpg_chr_pos_df.parquet \
    data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix/cpg_chr_pos_df.parquet
```
