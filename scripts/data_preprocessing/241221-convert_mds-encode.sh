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


# ENCODE WGBS (Healthy)
# 28,301,739 (CpG) * 95 (sample)

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


PARQUET_DIR_NAME=241213-encode_wgbs
CPG_SPLIT_NAME="train_0_9_val_0_1"
SAMLE_SPLIT_NAME="ind_tissue"


# Convert parquet to mds
python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/train_cpg-train_sample.parquet" \
    --output_dir data/mds/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/train \
    --num_workers "${NUM_WORKERS}" \
    --remove_unused_columns

python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/train_cpg-val_sample.parquet,data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/val_cpg-train_sample.parquet,data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/val_cpg-val_sample.parquet" \
    --output_dir data/mds/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/val \
    --num_workers "${NUM_WORKERS}" \
    --remove_unused_columns

python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/train_cpg-val_sample.parquet,data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/val_cpg-train_sample.parquet,data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/val_cpg-val_sample.parquet" \
    --num_shards_list "10,10,10" \
    --output_dir data/mds/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/val_10_shards \
    --num_workers 10 \
    --remove_unused_columns

python scripts/tools/data_preprocessing/convert_parquet_to_mds.py \
    --input_parquet_dir_list "data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/val_cpg-train_sample.parquet,data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg/val_cpg-val_sample.parquet" \
    --output_dir data/mds/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/val_new_cpg \
    --num_workers "${NUM_WORKERS}" \
    --remove_unused_columns


time_duration_and_reset "Encode WGBS data preprocessing"
