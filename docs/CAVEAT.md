# CAVEAT

## TCGA WGBS and Array, cpg_idx is not unique

However, we unique the sample_idx


## There are NaN in the tables

CpG (CpG Me value), Gene (in gene_id).

## The name of gene can be different

```python
(Pdb) self.gene_id_keys
Index([       'TSPAN6;ENSG00000000003',          'TNMD;ENSG00000000005',
                'DPM1;ENSG00000000419',         'SCYL3;ENSG00000000457',
            'C1orf112;ENSG00000000460',           'FGR;ENSG00000000938',
                 'CFH;ENSG00000000971',         'FUCA2;ENSG00000001036',
                'GCLC;ENSG00000001084',          'NFYA;ENSG00000001167',
       ...
              'PAUPAR;ENSG00000281880',    'AL512506.3;ENSG00000281883',
       'GIMAP1-GIMAP5;ENSG00000281887',    'AC018638.8;ENSG00000281896',
           'LINC02246;ENSG00000281903',    'AC233263.6;ENSG00000281904',
             'HERC2P7;ENSG00000281909',      'SNORA50A;ENSG00000281910',
           'LINC01144;ENSG00000281912',    'AC007389.5;ENSG00000281920'],
      dtype='object', name='gene_id', length=55503)
(Pdb) self.gene_bg_df.index
Index(['ENSG00000186092', 'ENSG00000187634', 'ENSG00000187961',
       'ENSG00000187583', 'ENSG00000187608', 'ENSG00000188157',
       'ENSG00000162571', 'ENSG00000176022', 'ENSG00000162572',
       'ENSG00000169972',
       ...
       'ENSG00000169953', 'ENSG00000012817', 'ENSG00000244395',
       'ENSG00000242389', 'ENSG00000169807', 'ENSG00000169800',
       'ENSG00000188120', 'ENSG00000172352', 'ENSG00000187191',
       'ENSG00000185894'],
      dtype='object', name='gene_id', length=20017)
```


## The column name in cpg background can be different

e.g., "sequence" -> "DNA_sequence"


## The cpg bg file has the same cpg_id, but diff cpg_chr_pos

```python
(Pdb)
cpg_bg_series
       cpg_chr_pos                                       DNA_sequence
cpg_id
10484   chr1_10484  CCTAACCCCTAACCCTAACCCTAACCCTAACCCTCGCGGTACCCTC...
10484   chr2_10484  CCCAACCCTAACCCCTCACCCTCACCCTCGACCCCCGACCCCCGAC...
```


## On UF HPG, the code may stuck due to error in dataloading

Not reported, we need to debug it with single card and no workers.


### sbatch with output log path

The log dir should exists


## On AWS

Use `torchrun` to start distributed jobs.

Use `python` with pytorch lighting leads to block when gathering tensor across distribution.


## Efficient Distributed Steaming with Datasets

Make sure `num_processes` can divide the number of shards (even after distributed split).
While the `num_workers` may not be dividable.

See "https://huggingface.co/docs/datasets/v3.0.0/en/use_with_pytorch#distributed"

> For iterable datasets:
> 
> If the dataset has a number of shards that is a factor of world_size (i.e. if dataset.n_shards % world_size == 0), then the shards are evenly assigned across the nodes, which is the most optimized. Otherwise, each node keeps 1 example out of world_size, skipping the other examples.
> 
>This can also be combined with a torch.utils.data.DataLoader if you want each node to use multiple workers to load the data.


## Re-arrange TCGA ARRAY parquet

Each of them have different amount of pair. Make them the same. It may solve the block of loading in TCGA


## Compatibility w/ HuggingFace Datasets `split_data_by_node`

DDP strategy in Lightning has handled the data split for map-style datasets.
While for iterable-style one, we need `split_data_by_node` to avoid data repetition.


## Dataloader Crush due to too many open files

RuntimeError: Too many open files. Communication with the workers is no longer possible. Please increase the limit using `ulimit -n` in the shell or change the sharing strategy by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code.

A solution is to use docker. Set `ulimit` when running containers.
- The `inifity` in `ulimit`: https://github.com/moby/moby/issues/44547#issuecomment-1334125338

Check `ulimit -a` first at the very beginning.


## use the first val dataloader as the monitor key to avoid the error of no metrics for **resume training**

`src/main.py:226-228`.


## Change the `monitor` name in `pl_callbacks.ModelCheckpoint` according to val dataset name.

If the dataset name in the dataset dict is changes, we need to change the `monitor` name in `pl_callbacks.ModelCheckpoint` in `src/main.py:228`.


## When streaming, number of shards should be divisible by the the number of process

When streaming `datasets.load_dataset(streaming=True)`, `data.*_num_shards` should be divisible by `NUM_NODES`.

Otherwise the loading speed would be slow.


## Stuck due to uneven tensor dim gathering in DDP

Apparently, if the tensor are not in the same size, after gathering, we got stuck in moving tensor from GPU to CPU. 

The real problem is that we cannot even access the gathered-concated tensor in `pdb`

Again, the number of shards should be divisible by the number of processes.
And each of them should have the same amount of samples.


## absl + ml_collections paramter overwrite

For those args are initialized with `placeholder` (`None`), we need to override them with `=`.

e.g., Use `--data.train_dataset.epoch_size=100` instead of `--data.train_dataset.epoch_size 100`. The latter one leads to make `config.data.train_dataset.epoch_size=True`.

## Slurm Sbatch with torchrun

Use `srun torchrun ...`. Otherwise `torchrun` get stuck.

## Fix TCGA Tissue mapping

(This fix is temporary, when process data from scratch, no need to use this.)

Fix sample tissue mapping:
```bash
mv data/processed/241231-tcga/sample_tissue_count_with_idx.csv data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv

python scripts/tools/data_preprocessing/stats_sample_tcga_by_filtered_gene_expr_parquet.py \
    --input_parquet_file data/processed/241231-tcga/gene_expr.filtered.parquet \
    --input_cancer_type_file data/parquet/241231-tcga_array/cancer_type.parquet \
    --output_dir data/parquet/241231-tcga/metadata \
    --output_file_name sample_split \
    --assign_na_tissue_to_unknown_samples

python scripts/tools/data_preprocessing/merge_sample_tissue_count_with_idx.py \
    --input_sample_tissue_counts_iwth_idx_files data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv \
    --output_path data/processed/241231-tcga/sample_tissue_count_with_idx.csv
```

Correct:
```bash
data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv
data/processed/241231-tcga/sample_tissue_count_with_idx.csv
grep 'TCGA-FY-A3TY' data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv
grep 'TCGA-FY-A3TY' data/processed/241231-tcga/sample_tissue_count_with_idx.csv
# index,tissue_name,sample_idx,sample_name,sample_tissue_same,tissue_idx,count
# 7704,TCGA-THCA,7704,TCGA-FY-A3TY,False,22,502
```

Wrong:
```bash
data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv
grep 'TCGA-FY-A3TY' data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv
# index,tissue_name,sample_idx,sample_name,sample_tissue_same,tissue_idx,count
# 7917,TCGA-BRCA,7917,TCGA-FY-A3TY,False,14,781
```

fix eval-results

```bash
input_dir=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/eval_results-test.parquet
correct_sample_idx_file=data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv
wrong_sample_idx_file=data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv
output_dir=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/fix_tissue_map-eval_results-test.parquet

python scripts/tools/eval/fix_sample_tissue_in_eval_results.py \
    --input_eval_result_dir=$input_dir \
    --output_eval_result_dir=$output_dir \
    --correct_sample_idx_csv=$correct_sample_idx_file \
    --wrong_sample_idx_csv=$wrong_sample_idx_file


REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/
REPO_URL=eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"


REPO_TYPE=dataset # model, dataset
LOCAL_DIR=outputs/eval/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws/

REPO_URL=xk-huang/eval-wo_tissue_embedder-tcga_mix_chr1-32xl40s-aws

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

Fix TCGA train_cpg-train_sample.parquet
```bash
input_dir=data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/train_cpg-train_sample.parquet
correct_sample_idx_file=data/parquet/241231-tcga/metadata/sample_split/sample_tissue_count_with_idx.csv
wrong_sample_idx_file=data/processed/241231-tcga/sample_tissue_count_with_idx-wrong_tissue_mapping.csv
output_dir=data/processed/241231-tcga_array-index_files-ind_cancer-non_nan/fix_tissue_map-train_cpg-train_sample.parquet

python scripts/tools/eval/fix_sample_tissue_in_eval_results.py \
    --input_eval_result_dir=$input_dir \
    --output_eval_result_dir=$output_dir \
    --correct_sample_idx_csv=$correct_sample_idx_file \
    --wrong_sample_idx_csv=$wrong_sample_idx_file --num_processes=20
```