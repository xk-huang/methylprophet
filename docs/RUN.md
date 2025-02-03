# RUN


## Env Vars

Create `./.env` env var file:

```shell
WANDB_API_KEY=$your_own_wandb_api_key
WANDB_PROJECT=methylformer
WANDB_MODE=online  # online, offline, disabled; ref: https://docs.wandb.ai/ref/python/init/
```


## Resume Training

Update: check `scripts/tools/generate_eval_resume_config.py`.

```shell
# Resume
NUM_PROCESS=

RESUME_TRAINING_CONFIG_PATH=
CKPT_PATH=

torchrun --nproc_per_node ${NUM_PROCESS} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 11453 \
-m src.main \
--main.resume_training_config_path="${RESUME_TRAINING_CONFIG_PATH}" --main.ckpt_path="${CKPT_PATH}"
```

If you want to update the flags in the config:

```shell
# Resume
NUM_PROCESS=

RESUME_TRAINING_CONFIG_PATH=
CKPT_PATH=

UPDATE_CONFIG_BY_DOTLIST=
# Use single quote like `'a.b=c,a.b.f=g'` to escape comma

torchrun --nproc_per_node ${NUM_PROCESS} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 11453 \
-m src.main \
--main.resume_training_config_path="${RESUME_TRAINING_CONFIG_PATH}" --main.ckpt_path="${CKPT_PATH}" \
--main.update_config_by_dotlist="${UPDATE_CONFIG_BY_DOTLIST}"
```


## Prediction with Model

Update: check `scripts/tools/generate_eval_resume_config.py`.


Usually we predict the full val set. Then the amount of cpg-sample pair is too large to compute and plot the eval.

```shell
# Eval
JOB_NAME="eval-${JOB_NAME}"

NUM_PROCESS=
VAL_NUM_WORKERS=
VAL_NUM_SHARDS_TRAIN_CPG_TRAIN_SAMPLE=8  # 2824, reduce as it is train
VAL_NUM_SHARDS_TRAIN_CPG_VAL_SAMPLE=2824
VAL_NUM_SHARDS_VAL_CPG_TRAIN_SAMPLE=2824
VAL_NUM_SHARDS_VAL_CPG_VAL_SAMPLE=2824

FULL_EVAL=True
SAVE_EVAL_RESULTS=True
PLOT_EVAL_RESULTS=True

MODEL_CONFIG_PATH=
WEIGHT_PATH=

torchrun --nproc_per_node ${NUM_PROCESS} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 11453 \
-m src.main \
# Other flags... \
--main.test_only \
--main.job_name=${JOB_NAME} \
--val_dataloader.num_workers=${VAL_NUM_WORKERS} \
--data.val_streaming=True \
--data.val_num_shards.train_cpg_train_sample=${VAL_NUM_SHARDS_TRAIN_CPG_TRAIN_SAMPLE} \
--data.val_num_shards.train_cpg_val_sample=${VAL_NUM_SHARDS_TRAIN_CPG_VAL_SAMPLE} \
--data.val_num_shards.val_cpg_train_sample=${VAL_NUM_SHARDS_VAL_CPG_TRAIN_SAMPLE} \
--data.val_num_shards.val_cpg_val_sample=${VAL_NUM_SHARDS_VAL_CPG_VAL_SAMPLE} \
--trainer_model.full_eval=${FULL_EVAL} \
--trainer_model.save_eval_results=${SAVE_EVAL_RESULTS} \
--trainer_model.plot_eval_results=${PLOT_EVAL_RESULTS} \
--main.model_config_path="${MODEL_CONFIG_PATH}" --main.weight_path="${WEIGHT_PATH}"
```

To update flags in config, see the "Resume Training".

Do not use resuming with flag updates, otherwise the saved config would be changed as the save ckpt dir is the old one.

## Find Batch Size

The last line matters.

```shell
output_dir=outputs/240830-slurm-conv_model-flash_attn
exp_name=bert_small

max_steps=200000
train_batch_size=28
val_batch_size=8
learning_rate=0.0001

DATASET=encode_wgbs-240802

python -m src.main \
--trainer.accelerator=gpu \
--trainer.max_steps=${max_steps} \
--trainer.log_every_n_steps=100 \
--trainer.val_check_interval=500 \
--trainer.gradient_clip_val=1.0 \
--model=src/configs/models/methyl_bert_conv_gene.py:medium \
--data=src/configs/data.py:${DATASET} \
--flagfile=src/configs/dataloader/default.cfg \
--main.output_dir=${output_dir} \
--main.exp_name=${exp_name} \
--train_dataloader.batch_size=${train_batch_size} \
--val_dataloader.batch_size=${val_batch_size} \
--trainer.accumulate_grad_batches=1 \
--trainer_model.full_eval=True \
--trainer_model.scheduler_type='constant' \
--trainer_model.learning_rate=${learning_rate} \
--trainer_model.scheduler_num_training_steps=${max_steps} \
--trainer_model.scheduler_num_warmup_steps=1000 \
--trainer.precision=bf16 \
--model._attn_implementation=flash_attention_2 \
--trainer.devices=1 --main.find_batch_size=True --trainer_model.full_eval=False --trainer.num_sanity_val_steps=0
```

## Profiling Model

Update: check `scripts/tools/generate_eval_resume_config.py`.

```shell
# ...
NUM_PROCESS=1
_attn_implementation=eager
job_name="analyze_model-${job_name}"

python \
-m src.analyze_model \
# ...
```

## Upload/Download Results from Huggingface Hub

### Set up

```shell
pip install -U huggingface_hub hf_transfer

git config --global credential.helper store

# Login huggingface hub
# huggingface-cli login
# export HF_TOKEN=hf_YOUR_TOKEN

# Check login
huggingface-cli whoami
```

### Download from Huggingface Hub

Fill the `REPO_ID` and change `LOCAL_DIR` to whatever you want

Result

```shell
LOCAL_DIR=results/
mkdir -p $LOCAL_DIR

REPO_OWNER_NAME=xk-huang
REPO_ID=

huggingface-cli download --repo-type dataset --local-dir $LOCAL_DIR/${REPO_ID} "${REPO_OWNER_NAME}/${REPO_ID}"
```

Model

```shell
LOCAL_DIR=results/
mkdir -p $LOCAL_DIR

REPO_OWNER_NAME=xk-huang
REPO_ID=

huggingface-cli download --repo-type model --local-dir $LOCAL_DIR/${REPO_ID} "${REPO_OWNER_NAME}/${REPO_ID}" --include 'ckpt/**'
```

### Upload to Huggingface Hub

Results

```shell
# Speed up with hf_transfer
# export HF_HUB_ENABLE_HF_TRANSFER=1 

REPO_TYPE=dataset # model, dataset
NUM_WORKERS=8

LOCAL_DIR=
REPO_URL=

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}"
```


Model

```shell
# Speed up with hf_transfer
# export HF_HUB_ENABLE_HF_TRANSFER=1 

REPO_TYPE=model # model, dataset
NUM_WORKERS=8

LOCAL_DIR=
REPO_URL=

huggingface-cli upload-large-folder "$REPO_URL" --repo-type="${REPO_TYPE}" "$LOCAL_DIR" --num-workers="${NUM_WORKERS}" --include 'ckpt/**'
```
