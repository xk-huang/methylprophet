#!/bin/bash

mkdir -p data/examples/{encode_wgbs,tcga_mix_chr1}


rsync -avP data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards data/examples/encode_wgbs/
rsync -avP data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/gene_expr.filtered.parquet data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/sample_tissue_count_with_idx.csv data/parquet/241213-encode_wgbs/metadata/cpg_per_chr_stats/cpg_chr_pos_df.parquet data/examples/encode_wgbs/


# REPO_TYPE=dataset # model, dataset
# NUM_WORKERS=8

# LOCAL_DIR=data/examples/encode_wgbs/
# REPO_URL=xk-huang/methylprophet-example_data-encode_wgbs

# huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"


# REPO_TYPE=dataset # model, dataset
# LOCAL_DIR=data/examples/encode_wgbs/

# REPO_URL=xk-huang/methylprophet-example_data-encode_wgbs

# mkdir -p $LOCAL_DIR
# huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}


rsync -avP data/mds_tokenized/241213-tcga-mix-chr1/val_10_shards data/examples/tcga_mix_chr1/
rsync -avP data/processed/241231-tcga/gene_expr.filtered.parquet data/processed/241231-tcga/sample_tissue_count_with_idx.csv data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix/cpg_chr_pos_df.parquet data/examples/tcga_mix_chr1/


# REPO_TYPE=dataset # model, dataset
# NUM_WORKERS=8

# LOCAL_DIR=data/examples/tcga_mix_chr1/
# REPO_URL=xk-huang/methylprophet-example_data-tcga_mix_chr1

# huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"


# REPO_TYPE=dataset # model, dataset
# LOCAL_DIR=data/examples/tcga_mix_chr1/

# REPO_URL=xk-huang/methylprophet-example_data-tcga_mix_chr1

# mkdir -p $LOCAL_DIR
# huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
