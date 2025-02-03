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


# convert bg files to parquet
python scripts/tools/data_preprocessing/convert_grch38_hg3_to_parquet.py \
    --grch38_file data/raw/grch38.v41.gtf \
    --hg38_file data/raw/hg38.fa.gz \
    --save_dir data/parquet/grch38_hg38

ROW_CHUNK_SIZE=10000

# convert raw gene expression csv to parquet
PARQUET_DIR_NAME_ARRAY=(241231-tcga_array 241231-tcga_epic 241231-tcga_wgbs)

for PARQUET_DIR_NAME in "${PARQUET_DIR_NAME_ARRAY[@]}"; do
    # convert gene expression csv to parquet
    OUTPUT_FILE_NAME=gene_expr.parquet
    python scripts/tools/data_preprocessing/convert_raw_csv_to_parquet.py \
        --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
        --file_name ge.csv \
        --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
        --output_file_name "${OUTPUT_FILE_NAME}" \
        --row_chunk_size "${ROW_CHUNK_SIZE}"

    # convert cancer type csv to parquet
    OUTPUT_FILE_NAME=cancer_type.parquet
    python scripts/tools/data_preprocessing/convert_cancer_type_to_parquet.py \
        --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
        --file_name project.csv \
        --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
        --output_file_name "${OUTPUT_FILE_NAME}" \
        --row_chunk_size "${ROW_CHUNK_SIZE}"

    # convert cpg bg
    OUTPUT_FILE_NAME=me.parquet
    python scripts/tools/data_preprocessing/convert_raw_csv_to_parquet.py \
        --data_dir "data/extracted/${PARQUET_DIR_NAME}" \
        --file_name me_rownamesloc.csv \
        --output_dir "data/parquet/${PARQUET_DIR_NAME}" \
        --output_file_name "${OUTPUT_FILE_NAME}" \
        --row_chunk_size "${ROW_CHUNK_SIZE}"

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

    INPUT_PARQUET_FILE="data/parquet/${PARQUET_DIR_NAME}/cpg_bg.parquet"
    INPUT_COLUMN_NAME="CpG_location"
    # NOTE xk: output_dir is different
    python scripts/tools/data_preprocessing/stats_cpg_per_chr.py \
        --input_parquet_file "${INPUT_PARQUET_FILE}" \
        --input_column_name "${INPUT_COLUMN_NAME}" \
        --output_dir "data/parquet/241231-tcga/metadata/cpg_per_chr_stats/${PARQUET_DIR_NAME}/"
done


# merge gene
python scripts/tools/data_preprocessing/merge_gene_expr_parquet.py \
    --input_gene_expr_parquet_list data/parquet/241231-tcga_array/gene_expr.parquet,data/parquet/241231-tcga_epic/gene_expr.parquet,data/parquet/241231-tcga_wgbs/gene_expr.parquet \
    --output_dir data/parquet/241231-tcga

python scripts/tools/data_preprocessing/create_filtered_gene_expr_parquet.py \
    --input_gene_expr_parquet_file data/parquet/241231-tcga/gene_expr.parquet \
    --input_grch38_parquet_file data/parquet/grch38_hg38/grch38.v41.parquet \
    --output_dir data/processed/241231-tcga/ \
    --output_file_name gene_expr.filtered.parquet \
    --nofilter_non_protein_coding_gene \
    --mean_threshold 0.5 \
    --std_threshold 0.5


python scripts/tools/compare_two_df.py --a data/parquet/241231-tcga_array/cancer_type.parquet --b data/parquet/241231-tcga_epic/cancer_type.parquet

python scripts/tools/compare_two_df.py --a data/parquet/241231-tcga_array/cancer_type.parquet --b data/parquet/241231-tcga_wgbs/cancer_type.parquet

# Now create merged sample names from gene expression and cancer type
python scripts/tools/data_preprocessing/stats_sample_tcga_by_filtered_gene_expr_parquet.py \
    --input_parquet_file data/processed/241231-tcga/gene_expr.filtered.parquet \
    --input_cancer_type_file data/parquet/241231-tcga_array/cancer_type.parquet \
    --output_dir data/parquet/241231-tcga/metadata \
    --output_file_name sample_split \
    --assign_na_tissue_to_unknown_samples

# Just copy the sample_split files to processed directory
python scripts/tools/data_preprocessing/merge_sample_tissue_count_with_idx.py \
    --input_sample_tissue_counts_iwth_idx_files data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_path data/processed/241231-tcga/sample_tissue_count_with_idx.csv


# merge cpg
python scripts/tools/data_preprocessing/merge_cpg_and_stats.py \
    --cpg_chr_pos_files data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_array/cpg_chr_pos_df.parquet,data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_epic/cpg_chr_pos_df.parquet,data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_wgbs/cpg_chr_pos_df.parquet \
    --output_dir data/parquet/241231-tcga/metadata/cpg_per_chr_stats/241231-tcga_mix

# Stats N in CpG DNA sequence
python scripts/tools/data_preprocessing/stats_n_in_cpg_dna_seq.py \
    --input_cpg_bg_parquet_dir_list data/parquet/241231-tcga_array/cpg_bg.parquet,data/parquet/241231-tcga_epic/cpg_bg.parquet,data/parquet/241231-tcga_wgbs/cpg_bg.parquet \
    --output_dir data/parquet/241231-tcga/metadata \
    --output_file_name cpg_nda_seq_n_stats \


time_duration_and_reset "TCGA Array, EPIC, WGBS: Merge gene expression, filtering, and merge index mapping"

# NUM_WORKERS=120 bash scripts/data_preprocessing/241231-tcga_array.sh
# NUM_WORKERS=120 bash scripts/data_preprocessing/241231-tcga_epic.sh
# NUM_WORKERS=120 bash scripts/data_preprocessing/241231-tcga_wgbs.sh

