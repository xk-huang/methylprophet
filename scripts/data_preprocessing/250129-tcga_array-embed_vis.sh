python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-BRCA,TCGA-LAML,TCGA-GBM \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx.csv

python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-BRCA \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx-brca.csv

python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-LAML \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx-laml.csv

python scripts/tools/eval/subset_sample_idx_by_tissue_name.py \
    --input_sample_idx_csv data/parquet/241231-tcga_array/metadata/subset_sample_split/ind_cancer/val_sample_tissue_count_with_idx.csv \
    --filtered_tissue_names TCGA-GBM \
    --output_dir data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/ \
    --output_filename val_sample_tissue_count_with_idx-gbm.csv

PARQUET_DIR_NAME=241231-tcga_embed_vis
ROW_CHUNK_SIZE=10000

INPUT_PARQUET_DIR_NAME=241231-tcga_array
INPUT_ME_PARQUET_FILE="data/parquet/${INPUT_PARQUET_DIR_NAME}/me.parquet"
INPUT_CPG_BG_PARQUET_FILE="data/parquet/${INPUT_PARQUET_DIR_NAME}/cpg_bg.parquet"
INPUT_CPG_ISLAND_TUPLE_PARQUET_FILE="data/parquet/${INPUT_PARQUET_DIR_NAME}/cpg_island_tuple.parquet"

INPUT_CPG_SPLIT_FILE="data/parquet/241231-tcga_array/metadata/embed_vis-cpg_split/val.parquet"
INPUT_SAMPLE_SPLIT_FILE="data/parquet/241231-tcga_array/metadata/embed_vis_sample_split/ind_cancer/val_sample_tissue_count_with_idx-brca.csv"

CPG_SPLIT_NAME=embed_vis
SAMLE_SPLIT_NAME=embed_vis
OUTPUT_DIR="data/processed/${PARQUET_DIR_NAME}-${CPG_SPLIT_NAME}-${SAMLE_SPLIT_NAME}/me_cpg_bg"
OUTPUT_FILE_NAME="val_cpg-sample_brca.parquet"
OUTPUT_CHUNK_SIZE=100  # ? / 100 = ?
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