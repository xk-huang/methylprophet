set -e

# TCGA, Array, Chr1
processed_dir=data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/
output_dir=data/me_parquets_after_mds/241213-tcga-array-chr1

split_array=(
    train_cpg-train_sample.parquet
    train_cpg-val_sample.parquet
    val_cpg-train_sample.parquet
    val_cpg-val_sample.parquet
)
for split in "${split_array[@]}"; do
    input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
    output_split_dir=${output_dir}/${split}
    python scripts/tools/eval/save_idx_and_me.py \
        --input_parquet_dir_list $input_parquet_dir_list \
        --output_dir $output_split_dir \
        --filter_by_chr chr1 \
        --num_workers 20
done

# TCGA, EPIC, Chr1
processed_dir=data/processed/241231-tcga_epic-index_files-all_tissue-non_nan
output_dir=data/me_parquets_after_mds/241213-tcga-epic-chr1
split=train_cpg-train_sample.parquet

input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
output_split_dir=${output_dir}/${split}
python scripts/tools/eval/save_idx_and_me.py \
    --input_parquet_dir_list $input_parquet_dir_list \
    --output_dir $output_split_dir \
    --filter_by_chr chr1 \
    --num_workers 20

# TCGA, WGBS, Chr1
processed_dir=data/processed/241231-tcga_wgbs-index_files-all_tissue
output_dir=data/me_parquets_after_mds/241213-tcga-wgbs-chr1
split=train_cpg-train_sample.parquet

input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
output_split_dir=${output_dir}/${split}
python scripts/tools/eval/save_idx_and_me.py \
    --input_parquet_dir_list $input_parquet_dir_list \
    --output_dir $output_split_dir \
    --filter_by_chr chr1 \
    --num_workers 20



# TCGA, Array, Chr123
processed_dir=data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/
output_dir=data/me_parquets_after_mds/241213-tcga-array-chr123

split_array=(
    train_cpg-train_sample.parquet
    train_cpg-val_sample.parquet
    val_cpg-train_sample.parquet
    val_cpg-val_sample.parquet
)
for split in "${split_array[@]}"; do
    input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
    output_split_dir=${output_dir}/${split}
    python scripts/tools/eval/save_idx_and_me.py \
        --input_parquet_dir_list $input_parquet_dir_list \
        --output_dir $output_split_dir \
        --filter_by_chr chr1,chr2,chr3 \
        --num_workers 20
done

# TCGA, EPIC, Chr123
processed_dir=data/processed/241231-tcga_epic-index_files-all_tissue-non_nan
output_dir=data/me_parquets_after_mds/241213-tcga-epic-chr123

split=train_cpg-train_sample.parquet
input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
output_split_dir=${output_dir}/${split}
python scripts/tools/eval/save_idx_and_me.py \
    --input_parquet_dir_list $input_parquet_dir_list \
    --output_dir $output_split_dir \
    --filter_by_chr chr1,chr2,chr3 \
    --num_workers 20

# TCGA, WGBS, Chr123
processed_dir=data/processed/241231-tcga_wgbs-index_files-all_tissue
output_dir=data/me_parquets_after_mds/241213-tcga-wgbs-chr123

split=train_cpg-train_sample.parquet
input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
output_split_dir=${output_dir}/${split}
python scripts/tools/eval/save_idx_and_me.py \
    --input_parquet_dir_list $input_parquet_dir_list \
    --output_dir $output_split_dir \
    --filter_by_chr chr1,chr2,chr3 \
    --num_workers 20

# ENCODE, WGBS
processed_dir=data/processed/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/
output_dir=data/me_parquets_after_mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue

split_array=(
    train_cpg-train_sample.parquet
    train_cpg-val_sample.parquet
    val_cpg-train_sample.parquet
    val_cpg-val_sample.parquet
)
for split in "${split_array[@]}"; do
    input_parquet_dir_list=${processed_dir}/me_cpg_bg/${split}
    output_split_dir=${output_dir}/${split}
    python scripts/tools/eval/save_idx_and_me.py \
        --input_parquet_dir_list $input_parquet_dir_list \
        --output_dir $output_split_dir \
        --num_workers 20
done


