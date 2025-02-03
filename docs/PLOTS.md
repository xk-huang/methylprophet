# PLOTS

## Download All results

```bash
REPO_NAME_ARRAY=(
    eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws
    eval-tcga_array_chr1-32xl40s-c2b2
    eval-tcga_array_epic_chr1-32xl40s-c2b2
    eval-tcga_array_wgbs_chr1-32xl40s-c2b2
    eval-tcga-mix-chr1-prev-eval_on_tcga_chr123
    eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr1
    eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123
    eval-encode_wgbs-bs_512-64xl40s-aws
)
for REPO_NAME in ${REPO_NAME_ARRAY[@]}; do
    REPO_TYPE=dataset # model, dataset
    LOCAL_DIR="outputs/eval/${REPO_NAME}"

    REPO_URL="xk-huang/${REPO_NAME}"

    mkdir -p $LOCAL_DIR
    huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
done
```

## stats_sample_name

```bash
python scripts/tools/plot/plot_sample_type_stats.py \
    --input_sample_tissue_count_csv data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_dir data/parquet/241231-tcga/metadata/sample_split/ \
    --output_plot_name sample_tissue_count-tcga \
    --overwrite

python scripts/tools/plot/plot_sample_type_stats.py \
    --input_sample_tissue_count_csv data/parquet/241213-encode_wgbs/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_dir data/parquet/241213-encode_wgbs/metadata/sample_split/ \
    --output_plot_name sample_tissue_count-encode \
    --overwrite \
    --rename_encode_tissue
```


## save pcc and me std

```bash
REPO_NAME_ARRAY=(
    eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws
    eval-tcga_array_chr1-32xl40s-c2b2
    eval-tcga_array_epic_chr1-32xl40s-c2b2
    eval-tcga_array_wgbs_chr1-32xl40s-c2b2
    eval-tcga-mix-chr1-prev-eval_on_tcga_chr123
    eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr1
    eval-tcga-mix-chr123-bs_512-32xl40s-aws-eval_on_tcga_chr123
    # eval-encode_wgbs-bs_512-64xl40s-aws
)
for REPO_NAME in ${REPO_NAME_ARRAY[@]}; do
    EVAL_DIR="outputs/eval/${REPO_NAME}"
    python scripts/tools/eval/save_pcc_and_me_std.py \
        --input_result_df ${EVAL_DIR}/eval_results-test.parquet \
        --input_group_idx_name_mapping_json ${EVAL_DIR}/group_idx_name_mapping-test.json \
        --output_dir ${EVAL_DIR}/pcc_and_me_std \
        --pcc_backend pandas --overwrite
    python scripts/tools/plot/plot_pcc.py \
        --input_eval_pcc_dir ${EVAL_DIR}/pcc_and_me_std \
        --output_dir ${EVAL_DIR}/plots/pcc/ \
        --output_plot_name ${REPO_NAME}
done


REPO_NAME=eval-encode_wgbs-bs_512-64xl40s-aws
EVAL_DIR=outputs/eval/${REPO_NAME}
python scripts/tools/eval/save_pcc_and_me_std.py \
    --input_result_df ${EVAL_DIR}/eval_results-test.parquet \
    --input_group_idx_name_mapping_json ${EVAL_DIR}/group_idx_name_mapping-test.json \
    --output_dir ${EVAL_DIR}/pcc_and_me_std \
    --pcc_backend pandas
REPO_NAME=eval-encode_wgbs-bs_512-64xl40s-aws
EVAL_DIR=outputs/eval/${REPO_NAME}
python scripts/tools/plot/plot_pcc.py \
    --input_eval_pcc_dir ${EVAL_DIR}/pcc_and_me_std \
    --output_dir ${EVAL_DIR}/plots/pcc/ \
    --output_plot_name ${REPO_NAME}


find outputs/eval -name '*.pdf' | tar -czf outputs/eval/plots-pcc.tar.gz --transform 's/^outputs\/eval\///' -T -
```