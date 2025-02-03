#!/bin/bash
set -e

SECONDS=0
SCRIPT_PATH=$(readlink -f "$0")

function time_duration_and_reset(){
    task_name=$1
    
    duration=$SECONDS
    h=$((duration / 3600))
    m=$(( (duration % 3600) / 60 ))
    s=$((duration % 60))
    
    printf "[%s] %s Execution time: %02d:%02d:%02d\n" "$(date '+%Y.%m.%d-%H:%M:%S')" "$task_name" $h $m $s | tee -a $SCRIPT_PATH.time.log
    
    SECONDS=0
}



# Whether to skip the generation of val_cpg data
if [[ NO_VAL_CPG_DATA -eq "" ]]; then
    echo -e "${RED}NO_VAL_CPG_DATA is not set, set to 0${RESET}"
    NO_VAL_CPG_DATA=0
else
    echo -e "${GREEN}NO_VAL_CPG_DATA is set to ${NO_VAL_CPG_DATA}${RESET}"
fi

# Define colors
RED='\e[31m'
GREEN='\e[32m'
BLUE='\e[34m'
RESET='\e[0m'

if [[ NUM_WORKERS -eq "" ]]; then
    echo -e "${RED}NUM_WORKERS is not set, set to 20${RESET}"
    NUM_WORKERS=20
else
    echo -e "${GREEN}NUM_WORKERS is set to ${NUM_WORKERS}${RESET}"
fi



# Convert parquet to mds
python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/train_cpg-train_sample.parquet,data/processed/241231-tcga_epic-index_files-all_tissue-non_nan/me_cpg_bg/train_cpg-train_sample.parquet,data/processed/241231-tcga_wgbs-index_files-all_tissue/me_cpg_bg/train_cpg-train_sample.parquet" \
    --output_dir data/mds_tokenized/241213-tcga-mix-chr123/train \
    --num_workers "${NUM_WORKERS}" \
    --remove_unused_columns \
    --filter_by_chr chr1,chr2,chr3 \
    --tokenize_sequence

python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/train_cpg-val_sample.parquet,data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/val_cpg-train_sample.parquet,data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/val_cpg-val_sample.parquet" \
    --output_dir data/mds_tokenized/241213-tcga-mix-chr123/val \
    --num_workers "${NUM_WORKERS}" \
    --remove_unused_columns \
    --filter_by_chr chr1,chr2,chr3 \
    --tokenize_sequence

python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/train_cpg-val_sample.parquet,data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/val_cpg-train_sample.parquet,data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/val_cpg-val_sample.parquet" \
    --num_shards_list "10,10,10" \
    --output_dir data/mds_tokenized/241213-tcga-mix-chr123/val_10_shards \
    --num_workers 10 \
    --remove_unused_columns \
    --filter_by_chr chr1,chr2,chr3 \
    --tokenize_sequence

python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/val_cpg-train_sample.parquet,data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/me_cpg_bg/val_cpg-val_sample.parquet" \
    --output_dir data/mds_tokenized/241213-tcga-mix-chr123/val_new_cpg \
    --num_workers "${NUM_WORKERS}" \
    --remove_unused_columns \
    --filter_by_chr chr1,chr2,chr3 \
    --tokenize_sequence


time_duration_and_reset "TCGA ARRAY, EPIC, and WGBS data preprocessing"
