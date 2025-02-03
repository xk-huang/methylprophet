# CUSTOMIZED

## Predicting with Customized Data

Download the models from [docs/MODEL_ZOO.md](./MODEL_ZOO.md)

Download the example data:

| Dataset | URL |
|----------|-----|
| ENCODE (WGBS) Val Example Data | [xk-huang/methylprophet-example_data-encode_wgbs](https://huggingface.co/datasets/xk-huang/methylprophet-example_data-encode_wgbs) |
| TCGA (Array+EPIC+WGBS) Val Example Data | [xk-huang/methylprophet-example_data-tcga_mix_chr1](https://huggingface.co/datasets/xk-huang/methylprophet-example_data-tcga_mix_chr1) |

```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/examples/encode_wgbs/

REPO_URL=xk-huang/methylprophet-example_data-encode_wgbs

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}


REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/examples/tcga_mix_chr1/

REPO_URL=xk-huang/methylprophet-example_data-tcga_mix_chr1

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}
```

The new genes should have the same rows as `data/examples/encode_wgbs/gene_expr.filtered.parquet`.
The columns are the samples you would like to predict.

For new CpG sites, to get the CpG-specific DNA sequence, you can use the human genome template from https://huggingface.co/datasets/xk-huang/250111_191630-grch38_hg38.

We provide an example commands in `scripts/examples/predict_example_data.py`.
The results will be in `outputs/example_data/eval-encode_wgbs/eval`.


We also provide an example python script in `scripts/examples/predict_example_data.py`, function `run_inference`.
The input and output format is displayed in the function logging:

```python
def run_inference(trainer_model, trainer_data_module, config):
    # Prepare model
    dtype = config.trainer.precision
    device = "cuda"
    trainer_model.to(dtype=DTYPE_MAPPING[dtype], device=device)

    trainer_data_module.setup("val")
    val_dataloader = trainer_data_module.val_dataloader()
    batch = next(iter(val_dataloader))
    batch = convert_data_type(batch, DTYPE_MAPPING[dtype], device)

    logging.info(f"Batch: {pprint.pformat(batch)}")
    trainer_model.eval()
    with torch.no_grad():
        sample_idx = batch.pop("sample_idx", None)
        if sample_idx is None:
            raise ValueError("sample_idx should not be None.")

        cpg_idx = batch.pop("cpg_idx", None)
        if cpg_idx is None:
            raise ValueError("cpg_idx should not be None.")

        group_idx = batch.pop("group_idx", None)
        if group_idx is None:
            raise ValueError("group_idx should not be None")

        gt_me = batch.pop("methylation", None)

        outputs = trainer_model.model(**batch)
        pred_me = outputs["output_value"]

    outputs = {
        "pred_me": pred_me,
        "gt_me": gt_me,
    }
    logging.info(f"Outputs: {pprint.pformat(outputs)}")
```

How to run the code:
```bash
output_dir=outputs
exp_name=example_data
job_name=encode_wgbs

job_name="try-${job_name}"

val_num_workers=20

val_batch_size=12
num_nbase=1000

model_config_path=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/config.yaml
weight_path=outputs/ckpts/encode_wgbs-bs_512-64xl40s-aws/ckpt/version_3/finished.ckpt

dataset_flagfile=src/configs/cfg/data/dataset/tcga_chr1/tokenized_val_dataset.cfg

local=data/examples/encode_wgbs/val_10_shards
group_idx_name_mapping_path=data/examples/encode_wgbs/val_10_shards/group_idx_name_mapping.json
gene_expr_df_path=data/examples/encode_wgbs/gene_expr.filtered.parquet
sample_idx_path=data/examples/encode_wgbs/sample_tissue_count_with_idx.csv

python scripts/examples/predict_example_data.py \
--trainer.devices=1 \
--trainer.num_nodes=1 \
--trainer.accelerator=gpu \
--trainer.precision=bf16 \
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--flagfile=${dataset_flagfile} \
--data.val_dataset.batch_size=${val_batch_size} \
--data.val_dataloader.batch_size=${val_batch_size} \
--data.val_dataloader.num_workers=${val_num_workers} \
--data.val_dataloader.drop_last=False \
--data.train_dataset.num_nbase=${num_nbase} \
--data.val_dataset.num_nbase=${num_nbase} \
--main.test_only \
--main.job_name=${job_name} \
--main.model_config_path="${model_config_path}" --main.weight_path="${weight_path}" \
--data.train_dataset.local=${local} \
--data.val_dataset.local=${local} \
--data.train_dataset.group_idx_name_mapping_path=${group_idx_name_mapping_path} \
--data.val_dataset.group_idx_name_mapping_path=${group_idx_name_mapping_path} \
--data.train_dataset.gene_expr_df_path=${gene_expr_df_path} \
--data.val_dataset.gene_expr_df_path=${gene_expr_df_path} \
--data.train_dataset.sample_idx_path=${sample_idx_path} \
--data.val_dataset.sample_idx_path=${sample_idx_path}
```

We provide the `cpg_idx` to `cpg_chr_pos` mapping in `cpg_chr_pos_df.parquet`, and `sample_idx` to `sample_name` and `tissue_name` in `sample_tissue_count_with_idx.csv`.


## Training with Customized Data

[Option 1] Prepare the data like files in `xk-huang/methylprophet-example_data-encode_wgbs` by yourself.

[Option 2] First download the data [docs/DATA.md](./DATA.md).
Check the scripts in `scripts/data_preprocessing` and [docs/DATA.md](./DATA.md). The required raw data should have the same format as `data/extracted/*/`. Then large portion of the data preprocessing scripts in `scripts/data_preprocessing` can be reused.
