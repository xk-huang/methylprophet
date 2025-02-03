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


# TCGA WGBS
# 23,047,052 (CpG) * 32 (sample)

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


# convert bg files to parquet
python scripts/tools/data_preprocessing/convert_grch38_hg3_to_parquet.py \
    --grch38_file data/raw/grch38.v41.gtf \
    --hg38_file data/raw/hg38.fa.gz \
    --save_dir data/parquet/grch38_hg38


# convert raw methylation and gene expression csv to parquet
PARQUET_DIR_NAME=241231-tcga_epic
ROW_CHUNK_SIZE=10000

# convert raw methylation csv to parquet
OUTPUT_FILE_NAME=me.parquet
python scripts/tools/data_preprocessing/convert_raw_csv_to_parquet.py \
    --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
    --file_name me_rownamesloc.csv \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
    --output_file_name "${OUTPUT_FILE_NAME}" \
    --row_chunk_size "${ROW_CHUNK_SIZE}"

# convert raw gene expression csv to parquet
OUTPUT_FILE_NAME=gene_expr.parquet
python scripts/tools/data_preprocessing/convert_raw_csv_to_parquet.py \
    --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
    --file_name ge.csv \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
    --output_file_name "${OUTPUT_FILE_NAME}" \
    --row_chunk_size "${ROW_CHUNK_SIZE}"

# convert raw cpg island info csv to parquet
OUTPUT_FILE_NAME=cpg_island.parquet
python scripts/tools/data_preprocessing/convert_raw_csv_to_parquet.py \
    --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
    --file_name cpg_names_cpg_location.txt \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
    --output_file_name "${OUTPUT_FILE_NAME}" \
    --row_chunk_size "${ROW_CHUNK_SIZE}" \
    --sep " "


# convert cpg bg to parquet, add nbase sequence
# ~320 GB memory with 20 workers
INPUT_FILE_NAME=me.parquet
OUTPUT_FILE_NAME=cpg_bg.parquet
HG38_PARQUET_FILE=data/parquet/grch38_hg38/hg38.fa.parquet
CPG_BG_NBASE_LENGTH=1000
python scripts/tools/data_preprocessing/convert_cpg_bg_to_parquet.py \
    --input_me_parquet_dir "data/parquet/${PARQUET_DIR_NAME}/${INPUT_FILE_NAME}" \
    --hg38_parquet_file "${HG38_PARQUET_FILE}" \
    --cpg_bg_nbase_length "${CPG_BG_NBASE_LENGTH}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
    --output_file_name "${OUTPUT_FILE_NAME}"


# check and stats nan in parquet
OUTPUT_FILE_NAME=me.parquet
python scripts/tools/data_preprocessing/check_nan_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_nan"

OUTPUT_FILE_NAME=gene_expr.parquet
python scripts/tools/data_preprocessing/check_nan_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_nan"

OUTPUT_FILE_NAME=cpg_bg.parquet
python scripts/tools/data_preprocessing/check_nan_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_nan"

OUTPUT_FILE_NAME=cpg_island.parquet
python scripts/tools/data_preprocessing/check_nan_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_nan"


# check uniqueness in parquet
OUTPUT_FILE_NAME=me.parquet
INDEX_COLUMN_NAME='Unnamed: 0'
python scripts/tools/data_preprocessing/check_unique_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --index_column_name "${INDEX_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_unique"

OUTPUT_FILE_NAME=gene_expr.parquet
INDEX_COLUMN_NAME='Unnamed: 0'
python scripts/tools/data_preprocessing/check_unique_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --index_column_name "${INDEX_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_unique"

OUTPUT_FILE_NAME=cpg_bg.parquet
INDEX_COLUMN_NAME='CpG_location'
python scripts/tools/data_preprocessing/check_unique_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --index_column_name "${INDEX_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_unique"

OUTPUT_FILE_NAME=cpg_island.parquet
INDEX_COLUMN_NAME='cpg'
python scripts/tools/data_preprocessing/check_unique_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --index_column_name "${INDEX_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_unique"


# Convert cpg island to tuple due to cpg may have multiple island info
OUTPUT_FILE_NAME=cpg_island_tuple.parquet
python scripts/tools/data_preprocessing/convert_cpg_island_tuple.py \
    --input_cpg_island_parquet "data/parquet/${PARQUET_DIR_NAME}/cpg_island.parquet" \
    --input_me_parquet "data/parquet/${PARQUET_DIR_NAME}/me.parquet" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
    --output_file_name "${OUTPUT_FILE_NAME}" \
    --row_chunk_size "${ROW_CHUNK_SIZE}"

# Stats cpg island info
OUTPUT_FILE_NAME=cpg_island.parquet
python scripts/tools/data_preprocessing/stats_cpg_island.py \
    --input_cpg_island_parquet "data/parquet/${PARQUET_DIR_NAME}/${OUTPUT_FILE_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/cpg_island_stats"


# check me and cpg_bg has the same index
INPUT_ME_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/me.parquet"
INPUT_CPG_BG_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_bg.parquet"
INPUT_ME_INDEX_NAME='Unnamed: 0'
INPUT_CPG_BG_INDEX_NAME="CpG_location"
python scripts/tools/data_preprocessing/check_me_cpg_bg_parquet_same_index.py \
    --input_me_parquet_file "${INPUT_ME_PARQUET_FILE}" \
    --input_cpg_bg_parquet_file "${INPUT_CPG_BG_PARQUET_FILE}" \
    --input_me_index_name "${INPUT_ME_INDEX_NAME}" \
    --input_cpg_bg_index_name "${INPUT_CPG_BG_INDEX_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_same_index" \
    --output_file_name "me_vs_cpg_bg"

# check me and cpg island tuple has the same index
INPUT_ME_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/me.parquet"
INPUT_CPG_BG_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_island_tuple.parquet"
INPUT_ME_INDEX_NAME='Unnamed: 0'
INPUT_CPG_BG_INDEX_NAME="cpg"
python scripts/tools/data_preprocessing/check_me_cpg_bg_parquet_same_index.py \
    --input_me_parquet_file "${INPUT_ME_PARQUET_FILE}" \
    --input_cpg_bg_parquet_file "${INPUT_CPG_BG_PARQUET_FILE}" \
    --input_me_index_name "${INPUT_ME_INDEX_NAME}" \
    --input_cpg_bg_index_name "${INPUT_CPG_BG_INDEX_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_same_index" \
    --output_file_name "me_vs_cpg_island_tuple"


# check me and gene expre has the same samples
python scripts/tools/data_preprocessing/check_stats_samples.py \
    --input_me_paruqet data/parquet/${PARQUET_DIR_NAME}/me.parquet/00000.parquet \
    --input_gene_expr_parquet data/parquet/${PARQUET_DIR_NAME}/gene_expr.parquet/00000.parquet \
    --output_dir data/parquet/${PARQUET_DIR_NAME}/metadata/check_stats_samples


# convert cancer type csv to parquet
OUTPUT_FILE_NAME=cancer_type.parquet
python scripts/tools/data_preprocessing/convert_cancer_type_to_parquet.py \
    --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
    --file_name project.csv \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
    --output_file_name "${OUTPUT_FILE_NAME}" \
    --row_chunk_size "${ROW_CHUNK_SIZE}"

OUTPUT_FILE_NAME=cancer_type.parquet
python scripts/tools/data_preprocessing/check_nan_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_nan"

OUTPUT_FILE_NAME=cancer_type.parquet
INDEX_COLUMN_NAME='Sample'
python scripts/tools/data_preprocessing/check_unique_in_parquet.py \
    --input_parquet_file "data/parquet/${PARQUET_DIR_NAME}/$OUTPUT_FILE_NAME" \
    --index_column_name "${INDEX_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/check_unique"


python scripts/tools/data_preprocessing/subset_merge_sample_tissue_count_with_idx.py \
    --input_me_parquet_file data/parquet/${PARQUET_DIR_NAME}/me.parquet/00000.parquet \
    --sample_tissue_count_with_idx_file data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_dir data/parquet/${PARQUET_DIR_NAME}/metadata/ \
    --output_file_name subset_sample_split

# No val split for TCGA WGBS
# ind-cancer
# INPUT_SAMPLE_TISSUE_COUNT_WITH_IDX_FILE="data/parquet/${PARQUET_DIR_NAME}/metadata/sample_split/sample_tissue_count_with_idx.csv"
# OUTPUT_DIR="data/parquet/${PARQUET_DIR_NAME}/metadata/sample_split"
# OUTPUT_SAMPLE_SPLIT_TYPE="ind_cancer"
# python scripts/tools/data_preprocessing/split_sample_tcga.py \
#     --input_sample_tissue_count_with_idx_file "${INPUT_SAMPLE_TISSUE_COUNT_WITH_IDX_FILE}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --output_sample_split_type "${OUTPUT_SAMPLE_SPLIT_TYPE}" \
#     --val_split_ratio 0.0


# Stats cpg per chr
INPUT_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_bg.parquet"
INPUT_COLUMN_NAME="CpG_location"
python scripts/tools/data_preprocessing/stats_cpg_per_chr.py \
    --input_parquet_file "${INPUT_PARQUET_FILE}" \
    --input_column_name "${INPUT_COLUMN_NAME}" \
    --output_dir "data/parquet/${PARQUET_DIR_NAME}/metadata/cpg_per_chr_stats" \
    --input_reindex_cpg_idx_file data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix/cpg_chr_pos_df.parquet

# Split cpg
INPUT_CHR_DF_PARQUET="data/parquet/${PARQUET_DIR_NAME}/metadata/cpg_per_chr_stats/cpg_chr_pos_df.parquet"
OUTPUT_DIR="data/parquet/${PARQUET_DIR_NAME}/metadata/cpg_split/index_files/"

python scripts/tools/data_preprocessing/split_cpg_by_index_files.py \
    --input_chr_df_parquet "${INPUT_CHR_DF_PARQUET}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file_name "train.parquet" \
    --index_method="all" \
    --shuffle


# Stats cpg split nbase overlap
# No overlap

# create me cpg bg parquet with cpg and sample splits
INPUT_ME_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/me.parquet"
INPUT_CPG_BG_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_bg.parquet"
INPUT_CPG_ISLAND_TUPLE_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_island_tuple.parquet"

# train cpg, train sample
CPG_SPLIT_NAME="index_files"
CPG_SPLIT_FILE_NAME="train.parquet"  # ? * 0.9 = ?
INPUT_CPG_SPLIT_FILE="data/parquet/${PARQUET_DIR_NAME}/metadata/cpg_split/${CPG_SPLIT_NAME}/${CPG_SPLIT_FILE_NAME}"

SAMLE_SPLIT_NAME="all_tissue"
# SAMPLE_SPLIT_FILE_NAME="train_sample_tissue_count_with_idx.csv"
SAMPLE_SPLIT_FILE_NAME="sample_tissue_count_with_idx.csv"
# INPUT_SAMPLE_SPLIT_FILE="data/parquet/${PARQUET_DIR_NAME}/metadata/sample_split/${SAMLE_SPLIT_NAME}/${SAMPLE_SPLIT_FILE_NAME}"
INPUT_SAMPLE_SPLIT_FILE="data/parquet/${PARQUET_DIR_NAME}/metadata/subset_sample_split/${SAMPLE_SPLIT_FILE_NAME}"

OUTPUT_DIR="data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg"
OUTPUT_FILE_NAME="train_cpg-train_sample.parquet"
OUTPUT_CHUNK_SIZE=1000  # ? / 100 = 3035
python scripts/tools/data_preprocessing/create_me_cpg_bg_parquet_by_splits.py \
    --input_me_parquet_file "${INPUT_ME_PARQUET_FILE}" \
    --input_cpg_bg_parquet_file "${INPUT_CPG_BG_PARQUET_FILE}" \
    --input_cpg_island_tuple_parquet_file "${INPUT_CPG_ISLAND_TUPLE_PARQUET_FILE}" \
    --input_cpg_split_file "${INPUT_CPG_SPLIT_FILE}" \
    --input_sample_split_file "${INPUT_SAMPLE_SPLIT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file_name "${OUTPUT_FILE_NAME}" \
    --output_chunk_size "${OUTPUT_CHUNK_SIZE}" \
    # --num_workers "${NUM_WORKERS}"

# stats shards
INPUT_PROCESSED_DATASET_DIR="data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/"
python scripts/tools/data_preprocessing/stats_processed_dataset_shards.py \
    --input_processed_dataset_dir "${INPUT_PROCESSED_DATASET_DIR}" \
    --num_workers "${NUM_WORKERS}"

# create gene expr parquet
# TODO: the format is not align with previous ones
# TODO: figure out the threshold
python scripts/tools/data_preprocessing/create_filtered_gene_expr_parquet.py \
    --input_gene_expr_parquet_file "data/parquet/${PARQUET_DIR_NAME}/gene_expr.parquet" \
    --input_grch38_parquet_file "data/parquet/grch38_hg38/grch38.v41.parquet"  \
    --output_dir "data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/" \
    --output_file_name "gene_expr.filtered.parquet" \
    --nofilter_non_protein_coding_gene \
    # --mean_threshold 0.5 \
    # --std_threshold 0.5


# Split nan and non-nan me cpg bg parquet
# num_workers=10, memory: ~630GB
INPUT_PROCESSED_DATASET_DIR="data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/"
python scripts/tools/data_preprocessing/split_nan_in_me_cpg_bg_parquets_by_splits.py \
    --input_me_cpg_bg_dataset_dir ${INPUT_PROCESSED_DATASET_DIR} \
    --num_workers 10

# stats shards
INPUT_PROCESSED_DATASET_DIR="data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}-nan/"
python scripts/tools/data_preprocessing/stats_processed_dataset_shards.py \
    --input_processed_dataset_dir "${INPUT_PROCESSED_DATASET_DIR}" \
    --num_workers "${NUM_WORKERS}"

INPUT_PROCESSED_DATASET_DIR="data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}-non_nan/"
python scripts/tools/data_preprocessing/stats_processed_dataset_shards.py \
    --input_processed_dataset_dir "${INPUT_PROCESSED_DATASET_DIR}" \
    --num_workers "${NUM_WORKERS}"


time_duration_and_reset "TCGA EPIC data preprocessing"
