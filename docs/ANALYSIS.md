# ANALYSIS

## Prediction 

250114

```
TCGA Chr1:

half_data: `/insomnia001/depts/houlab/users/xh2689/codes/methylformer/outputs/241229-methylformer_bert-ins/eval-250110-train-tcga_chr1-base-half_data-12xl40s/eval/version_0`
full_data: `/insomnia001/depts/houlab/users/xh2689/codes/methylformer/outputs/241229-methylformer_bert-ins/eval-250112-eval-tcga_chr1-base-12xl40s/eval/version_0`


ENCODE:

half_data: `outputs/241229-methylformer_bert-ins/eval-241229-train-encode-base-12xl40s/eval/version_12`
```

## Folder Structure and File Format.

A typicall eval folder consists of:
```
outputs/241229-methylformer_bert-ins/eval-250110-train-tcga_chr1-base-half_data-12xl40s/eval/version_0
├── eval
│   ├── log_dict-train_cpg-val_sample.parquet.json
│   ├── log_dict-val_cpg-train_sample.parquet.json
│   ├── log_dict-val_cpg-val_sample.parquet.json
│   ├── metrics_summary.csv
│   ├── metrics_summary.md
│   ├── train_cpg-val_sample.parquet--me_pcc_by_cpg_id-boxenplot.png
│   ├── train_cpg-val_sample.parquet--me_pcc_by_cpg_id-boxplot.png
│   ├── train_cpg-val_sample.parquet--me_pcc_by_sample_id-boxenplot.png
│   ├── train_cpg-val_sample.parquet--me_pcc_by_sample_id-boxplot.png
│   ├── train_cpg-val_sample.parquet--pcc_gt_std_by_cpg_id-scatter.png
│   ├── train_cpg-val_sample.parquet--pcc_pred_std_by_cpg_id-scatter.png
│   ├── val_cpg-train_sample.parquet--me_pcc_by_cpg_id-boxenplot.png
│   ├── val_cpg-train_sample.parquet--me_pcc_by_cpg_id-boxplot.png
│   ├── val_cpg-train_sample.parquet--me_pcc_by_sample_id-boxenplot.png
│   ├── val_cpg-train_sample.parquet--me_pcc_by_sample_id-boxplot.png
│   ├── val_cpg-train_sample.parquet--pcc_gt_std_by_cpg_id-scatter.png
│   ├── val_cpg-train_sample.parquet--pcc_pred_std_by_cpg_id-scatter.png
│   ├── val_cpg-val_sample.parquet--me_pcc_by_cpg_id-boxenplot.png
│   ├── val_cpg-val_sample.parquet--me_pcc_by_cpg_id-boxplot.png
│   ├── val_cpg-val_sample.parquet--me_pcc_by_sample_id-boxenplot.png
│   ├── val_cpg-val_sample.parquet--me_pcc_by_sample_id-boxplot.png
│   ├── val_cpg-val_sample.parquet--pcc_gt_std_by_cpg_id-scatter.png
│   └── val_cpg-val_sample.parquet--pcc_pred_std_by_cpg_id-scatter.png
├── eval_results-test.csv
├── group_idx_name_mapping-test.json
└── tar
    ├── eval_results-test.tar.gz.part_000
    ├── eval_results-test.tar.gz.part_001
    ├── eval_results-test.tar.gz.part_002
    └── eval_results-test.tar.gz.part_003
```

`eval_results-test.csv`: The predicted methylation, ground-truth methylation, mse loss, sample index, cpg index, and group_idx: `mse_loss_per_point, pred_methyl, gt_methyl, cpg_idx, sample_idx, group_idx`.

`group_idx_name_mapping-test.json`: The name (path) of each group, i.e., the val splits: 1) train_cpg-val_sample, 2) val_cpg-train_sample, 3) val_cpg-val_sample.

`eval/`: The PCC eval results.

`tar`: compressed `eval_results-test.csv` to upload and backup them to HuggingFace.
`