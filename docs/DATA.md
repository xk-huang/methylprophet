# DATA

## Statistics

The scale of DNAm data included in MethylProphet.

| Data Source \& Sequen. | \# CpG Sites | \# Tissues / Cancers | \# Samples | \# CpG-Sample Pairs | \# Pairs w/ Me. |
|------------------------|-------------:|---------------------:|-----------:|--------------------:|----------------:|
| ENCODE WGBS            |   27,078,450 |                   57 |         95 |       2,572,452,750 |   2,572,452,750 |
| TCGA Array             |      408,399 |                   33 |      9,194 |       3,754,820,406 |   3,684,770,086 |
| TCGA EPIC              |      740,296 |                    4 |      1,706 |       1,262,944,976 |   1,188,102,524 |
| TCGA WGBS              |   23,047,052 |                   17 |         32 |         737,505,664 |     737,505,664 |


## Protocols

The data statistics among all the data source and splits in our experiments. The number of tokens is estimated by the average sequence length (i.e., 200) of the input embeddings of the transformer encoder.


<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Dataset</th>
    <th class="tg-dvpl">Chr.</th>
    <th class="tg-dvpl">Sequen.</th>
    <th class="tg-dvpl">Split</th>
    <th class="tg-dvpl"># CpG Sites</th>
    <th class="tg-dvpl"># Tissues</th>
    <th class="tg-dvpl"># Samples</th>
    <th class="tg-dvpl"># Pairs w/ Me.</th>
    <th class="tg-dvpl"># Tokens</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="4">ENCODE</td>
    <td class="tg-dvpl" rowspan="4">1-22</td>
    <td class="tg-dvpl" rowspan="4">WGBS</td>
    <td class="tg-dvpl">Train: Train CpG - Train Sample</td>
    <td class="tg-dvpl">24,363,170</td>
    <td class="tg-dvpl">57</td>
    <td class="tg-dvpl">66</td>
    <td class="tg-dvpl">1,607,969,220</td>
    <td class="tg-dvpl">321,593,844,000</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Train CpG - Val Sample</td>
    <td class="tg-dvpl">24,363,170</td>
    <td class="tg-dvpl">22</td>
    <td class="tg-dvpl">29</td>
    <td class="tg-dvpl">706,531,930</td>
    <td class="tg-dvpl">141,306,386,000</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Val CpG - Train Sample</td>
    <td class="tg-dvpl">2,707,033</td>
    <td class="tg-dvpl">57</td>
    <td class="tg-dvpl">66</td>
    <td class="tg-dvpl">178,664,178</td>
    <td class="tg-dvpl">35,732,835,600</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Val CpG - Val Sample</td>
    <td class="tg-dvpl">2,707,033</td>
    <td class="tg-dvpl">22</td>
    <td class="tg-dvpl">29</td>
    <td class="tg-dvpl">78,503,957</td>
    <td class="tg-dvpl">15,700,791,400</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="6">TCGA</td>
    <td class="tg-dvpl" rowspan="6">1</td>
    <td class="tg-dvpl">Array</td>
    <td class="tg-dvpl" rowspan="3">Train: Train CpG - Train Sample</td>
    <td class="tg-dvpl">33,885</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">8,258</td>
    <td class="tg-dvpl">275,018,849</td>
    <td class="tg-dvpl">55,003,769,800</td>
  </tr>
  <tr>
    <td class="tg-dvpl">EPIC</td>
    <td class="tg-dvpl">71,748</td>
    <td class="tg-dvpl">4</td>
    <td class="tg-dvpl">1,706</td>
    <td class="tg-dvpl">115,856,100</td>
    <td class="tg-dvpl">23,171,220,000</td>
  </tr>
  <tr>
    <td class="tg-dvpl">WGBS</td>
    <td class="tg-dvpl">1,999,446</td>
    <td class="tg-dvpl">17</td>
    <td class="tg-dvpl">32</td>
    <td class="tg-dvpl">63,982,272</td>
    <td class="tg-dvpl">12,796,454,400</td>
  </tr>
  <tr>
    <td class="tg-dvpl" rowspan="3">Array</td>
    <td class="tg-dvpl">Val: Train CpG - Val Sample</td>
    <td class="tg-dvpl">33,885</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">920</td>
    <td class="tg-dvpl">30,638,464</td>
    <td class="tg-dvpl">6,127,692,800</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Val CpG - Train Sample</td>
    <td class="tg-dvpl">6,742</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">8,258</td>
    <td class="tg-dvpl">55,141,308</td>
    <td class="tg-dvpl">11,028,261,600</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Val CpG - Val Sample</td>
    <td class="tg-dvpl">6,742</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">920</td>
    <td class="tg-dvpl">6,143,360</td>
    <td class="tg-dvpl">1,228,672,000</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="6">TCGA</td>
    <td class="tg-dvpl" rowspan="6">1-3</td>
    <td class="tg-dvpl">Array</td>
    <td class="tg-dvpl" rowspan="3">Train: Train CpG - Train Sample</td>
    <td class="tg-dvpl">78,211</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">8,258</td>
    <td class="tg-dvpl">632,281,133</td>
    <td class="tg-dvpl">126,456,226,600</td>
  </tr>
  <tr>
    <td class="tg-dvpl">EPIC</td>
    <td class="tg-dvpl">172,722</td>
    <td class="tg-dvpl">4</td>
    <td class="tg-dvpl">1,706</td>
    <td class="tg-dvpl">276,181,739</td>
    <td class="tg-dvpl">55,236,347,800</td>
  </tr>
  <tr>
    <td class="tg-dvpl">WGBS</td>
    <td class="tg-dvpl">5,396,193</td>
    <td class="tg-dvpl">17</td>
    <td class="tg-dvpl">32</td>
    <td class="tg-dvpl">172,678,176</td>
    <td class="tg-dvpl">34,535,635,200</td>
  </tr>
  <tr>
    <td class="tg-dvpl" rowspan="3">Array</td>
    <td class="tg-dvpl">Val: Train CpG - Val Sample</td>
    <td class="tg-dvpl">78,211</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">920</td>
    <td class="tg-dvpl">70,443,801</td>
    <td class="tg-dvpl">14,088,760,200</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Val CpG - Train Sample</td>
    <td class="tg-dvpl">14,893</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">8,258</td>
    <td class="tg-dvpl">121,617,682</td>
    <td class="tg-dvpl">24,323,536,400</td>
  </tr>
  <tr>
    <td class="tg-dvpl">Val: Val CpG - Val Sample</td>
    <td class="tg-dvpl">14,893</td>
    <td class="tg-dvpl">33</td>
    <td class="tg-dvpl">920</td>
    <td class="tg-dvpl">13,550,097</td>
    <td class="tg-dvpl">2,710,019,400</td>
  </tr>
</tbody></table>

## Prepare from Raw Data


### From HF

Download raw data from huggingface.

| Dataset | URL |
|----------|-----|
| ENCODE (WGBS) | [xk-huang/250116_191844-241213-encode-raw](https://huggingface.co/datasets/xk-huang/250116_191844-241213-encode-raw) |
| TCGA (Array+EPIC+WGBS) | [xk-huang/250116_191533-241231-tcga-raw](https://huggingface.co/datasets/xk-huang/250116_191533-241231-tcga-raw) |
| CpG List of TCGA | [xk-huang/250116_192054-tcga_cpg-raw](https://huggingface.co/datasets/xk-huang/250116_192054-tcga_cpg-raw) |


ENCODE
```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/tar/241213-encode-raw

REPO_URL=xk-huang/250116_191844-241213-encode-raw

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}

# decompress
# prefix directory: data/extracted/...
PARQUET_DIR=data/extracted
mkdir -p $PARQUET_DIR
cat ${LOCAL_DIR}/*.tar.gz.part_??? | tar -x --use-compress-program=pigz --strip-components=2 -C ${PARQUET_DIR}
```


TCGA

```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/tar/241231-tcga-raw

REPO_URL=xk-huang/250116_191533-241231-tcga-raw

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}

# decompress
# prefix directory: data/extracted/...
PARQUET_DIR=data/extracted
mkdir -p $PARQUET_DIR
cat ${LOCAL_DIR}/*.tar.gz.part_??? | tar -x --use-compress-program=pigz --strip-components=2 -C ${PARQUET_DIR}
```

TCGA CpG List

```bash
REPO_TYPE=dataset # model, dataset
LOCAL_DIR=data/tar/tcga_cpg-raw

REPO_URL=xk-huang/250116_192054-tcga_cpg-raw

mkdir -p $LOCAL_DIR
huggingface-cli download --repo-type $REPO_TYPE --local-dir $LOCAL_DIR ${REPO_URL}

# decompress
# prefix directory: data/extracted/...
PARQUET_DIR=data/extracted
tar -xvf ${LOCAL_DIR}/250116_192054-tcga_cpg-raw.tar.gz --strip-components=2 -C ${PARQUET_DIR}
```



### From Local

<details>
<summary>From Local (deprecated)</summary>
Get raw data

```bash
mkdir -p data/raw
cd data/raw
gdown 1f_xVUIw9_3CSd9tAHizcSwTbCH4_s7_p  # grch38.v41.zip
unzip grch38.v41.zip
gdown 1PaTUnYVGf6rwv-1QQ280bdBGNPrusrWp  # hg38.fa.gz
cd -

SOURCE_DIR=/insomnia001/depts/houlab/users/wh2526/methylformer/data/encode/
EXTRACTED_DIR=data/extracted/241213-encode_wgbs
mkdir -p $EXTRACTED_DIR
rsync -avP $SOURCE_DIR $EXTRACTED_DIR
mv ${EXTRACTED_DIR}/ge_commongene.csv ${EXTRACTED_DIR}/ge.csv

SOURCE_DIR=/insomnia001/depts/houlab/users/wh2526/methylformer/data/tcga/hg38/2024/
EXTRACTED_BASE_DIR=data/extracted/
EXTRACTED_DIR=data/extracted/241231-tcga
mkdir -p $EXTRACTED_DIR
rsync -avP $SOURCE_DIR $EXTRACTED_DIR
mkdir -p $EXTRACTED_BASE_DIR/{241231-tcga_array,241231-tcga_epic,241231-tcga_wgbs}

mv $EXTRACTED_DIR/450k* $EXTRACTED_BASE_DIR/241231-tcga_array
mv $EXTRACTED_DIR/ge*450k* $EXTRACTED_BASE_DIR/241231-tcga_array
mv $EXTRACTED_BASE_DIR/241231-tcga_array/450k.csv $EXTRACTED_BASE_DIR/241231-tcga_array/me_rownamesloc.csv
mv $EXTRACTED_BASE_DIR/241231-tcga_array/ge_for_450k.csv $EXTRACTED_BASE_DIR/241231-tcga_array/ge.csv
mv $EXTRACTED_BASE_DIR/241231-tcga_array/450k_cpg_names_cgi_location.txt $EXTRACTED_BASE_DIR/241231-tcga_array/cpg_names_cpg_location.txt
mv $EXTRACTED_BASE_DIR/241231-tcga_array/450k_cpgnames.txt $EXTRACTED_BASE_DIR/241231-tcga_array/cpgnames.txt
mv $EXTRACTED_BASE_DIR/241231-tcga_array/450k_samples.txt $EXTRACTED_BASE_DIR/241231-tcga_array/samples.txt
cp $EXTRACTED_DIR/project.csv $EXTRACTED_BASE_DIR/241231-tcga_array

mv $EXTRACTED_DIR/EPIC.csv $EXTRACTED_BASE_DIR/241231-tcga_epic
mv $EXTRACTED_DIR/epic* $EXTRACTED_BASE_DIR/241231-tcga_epic
mv $EXTRACTED_DIR/ge*epic* $EXTRACTED_BASE_DIR/241231-tcga_epic
mv $EXTRACTED_BASE_DIR/241231-tcga_epic/EPIC.csv $EXTRACTED_BASE_DIR/241231-tcga_epic/me_rownamesloc.csv
mv $EXTRACTED_BASE_DIR/241231-tcga_epic/ge_for_epic.csv $EXTRACTED_BASE_DIR/241231-tcga_epic/ge.csv
mv $EXTRACTED_BASE_DIR/241231-tcga_epic/epic_cpg_names_cgi_location.txt $EXTRACTED_BASE_DIR/241231-tcga_epic/cpg_names_cpg_location.txt
mv $EXTRACTED_BASE_DIR/241231-tcga_epic/epic_cpgnames.txt $EXTRACTED_BASE_DIR/241231-tcga_epic/cpgnames.txt
mv $EXTRACTED_BASE_DIR/241231-tcga_epic/epic_samples.txt $EXTRACTED_BASE_DIR/241231-tcga_epic/samples.txt
cp $EXTRACTED_DIR/project.csv $EXTRACTED_BASE_DIR/241231-tcga_epic

mv $EXTRACTED_DIR/wgbs* $EXTRACTED_BASE_DIR/241231-tcga_wgbs
mv $EXTRACTED_DIR/ge*wgbs* $EXTRACTED_BASE_DIR/241231-tcga_wgbs
mv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/wgbs.csv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/me_rownamesloc.csv
mv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/ge_for_wgbs.csv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/ge.csv
mv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/wgbs_cpg_names_cgi_location.txt $EXTRACTED_BASE_DIR/241231-tcga_wgbs/cpg_names_cpg_location.txt
mv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/wgbs_cpgnames.txt $EXTRACTED_BASE_DIR/241231-tcga_wgbs/cpgnames.txt
mv $EXTRACTED_BASE_DIR/241231-tcga_wgbs/wgbs_samples.txt $EXTRACTED_BASE_DIR/241231-tcga_wgbs/samples.txt
cp $EXTRACTED_DIR/project.csv $EXTRACTED_BASE_DIR/241231-tcga_wgbs
```
</details>

## Process

You can choose to tokenize sequence or not:
- No sequence tokenization, 1x time, but 3x more space
- Tokenize sequence, takes 3x more time, but use 1x space

ENCODE

```bash
# scripts/data_preprocessing/convert_data_full.sh

NUM_WORKERS=120 bash scripts/data_preprocessing/241213-encode.sh

# No sequence tokenization, 1x time but 3x space
# NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-encode.sh

# Tokenize sequence, takes 3x more time, but use 1x space
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-encode.sh
```

TCGA

```bash
# scripts/data_preprocessing/convert_data_full.sh

# NOTE xk: We first merge the sample names to get unique sample idx
bash scripts/data_preprocessing/250102-tcga-merge_gene_expr_filtering_and_idx_mapping.sh

NUM_WORKERS=120 bash scripts/data_preprocessing/241231-tcga_array.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241231-tcga_epic.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241231-tcga_wgbs.sh

# No sequence tokenization, 1x time but 3x space
# NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tcga-chr1.sh
# NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tcga.sh

# Tokenize sequence, takes 3x more time
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-tcga-mix-chr1.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-tcga-array-chr1.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-tcga-array_epic-chr1.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-tcga-array_wgbs-chr1.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-tcga-mix-chr123.sh
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-tcga-mix.sh
```

Get mds dataset size:

```bash
MDS_DIR=data/mds_tokenized/241213-tcga-mix-chr1
find $MDS_DIR -maxdepth 1 -mindepth 1 -type d | sort | xargs -I {} python scripts/tools/get_mds_dataset_size.py --d={} --output_dir=$MDS_DIR


MDS_DIR_ARRAY=(
    data/mds_tokenized/241213-tcga-mix-chr1
    data/mds_tokenized/241213-tcga-array-chr1
    data/mds_tokenized/241213-tcga-array_epic-chr1
    data/mds_tokenized/241213-tcga-array_wgbs-chr1
    data/mds_tokenized/241213-tcga-mix
    data/mds_tokenized/241213-tcga-mix-chr123
    data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue
)
for MDS_DIR in "${MDS_DIR_ARRAY[@]}"; do
    find $MDS_DIR -maxdepth 1 -mindepth 1 -type d | sort | xargs -I {} python scripts/tools/get_mds_dataset_size.py --d={} --output_dir=$MDS_DIR
done


find data/mds_tokenized -maxdepth 2 -name '*.tsv' | sort | xargs -I {} cat {} > misc/mds_dataset_size.tsv

# remove all tsv
find data/mds_tokenized -maxdepth 2 -name '*.tsv' -exec rm {} \;
```

Get me statistics:

```bash
ME_DIR=data/parquet/241213-encode_wgbs
python scripts/tools/data_preprocessing/stats_me_parquets.py \
    --i "${ME_DIR}/me.parquet" \
    --output_dir "${ME_DIR}/metadata/me_stats"

ME_DIR_ARRAY=(
    data/parquet/241213-encode_wgbs
    data/parquet/241231-tcga_array
    data/parquet/241231-tcga_epic
    data/parquet/241231-tcga_wgbs
)
for ME_DIR in "${ME_DIR_ARRAY[@]}"; do 
    python scripts/tools/data_preprocessing/stats_me_parquets.py \
        --i "${ME_DIR}/me.parquet" \
        --output_dir "${ME_DIR}/metadata/me_stats"
done

find data/parquet -maxdepth 4 -path '*/metadata/me_stats/*' -name 'me_stats.tsv' | sort | xargs -I {} cat {} > misc/me_stats.tsv
```

## Dataset Storage

### Sequence tokenized

ENCODE
```
1.2T    data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue
```

TCGA
```
2.7T    data/mds_tokenized/241213-tcga-mix
265G    data/mds_tokenized/241213-tcga-mix-chr1
```

### Sequence not tokenzied

ENCODE
```
2.8T    data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue
```

TCGA
```
6.2T    data/mds_tokenized/241213-tcga-mix
602G    data/mds/241213-tcga-mix-chr1
```


### Other files
ENCODE
```
└── data [144G]
    ├── raw [2.5G]
    ├── processed [65G]
    │   └── 241213-encode_wgbs-train_0_9_val_0_1-ind_tissue [65G]
    ├── parquet [31G]
    │   ├── 241213-encode_wgbs [30G]
    │   └── grch38_hg38 [1.5G]
    └── extracted [45G]
        └── 241213-encode_wgbs [45G]
```

TCGA
```
└── data [454G]
    ├── processed [236G]
    │   ├── 241231-tcga_epic-index_files-all_tissue [20G]
    │   ├── 241231-tcga_array-index_files-ind_cancer [56G]
    │   ├── 241231-tcga_epic-index_files-all_tissue-non_nan [19G]
    │   ├── 241213-encode_wgbs-train_0_9_val_0_1-ind_tissue [65G]
    │   ├── 241231-tcga_array-index_files-ind_cancer-non_nan [56G]
    │   ├── 241231-tcga_wgbs-index_files-all_tissue [17G]
    │   ├── 241231-tcga_epic-index_files-all_tissue-nan [1.3G]
    │   ├── 241231-tcga [2.2G]
    │   └── 241231-tcga_array-index_files-ind_cancer-nan [2.7G]
    ├── raw [2.5G]
    ├── extracted [101G]
    │   ├── 241231-tcga_epic [22G]
    │   ├── 241231-tcga_array [69G]
    │   ├── 241231-tcga_wgbs [4.3G]
    │   └── 241231-tcga [6.7G]
    └── parquet [114G]
        ├── 241213-encode_wgbs [30G]
        ├── 241231-tcga_epic [18G]
        ├── 241231-tcga_array [57G]
        ├── grch38_hg38 [1.5G]
        ├── 241231-tcga_wgbs [6.1G]
        └── 241231-tcga [3.3G]
```

## Check Dataloading

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=16 \
    --master-addr=localhost \
    --master-port=52408 \
src/check_data_loading.py --flagfile=src/configs/cfg/data/dataset/encode/tokenized_dataset.cfg \
--main.exp_name=check_data_load \
--main.job_name=encode \
--trainer.devices=16 \
--data.train_dataloader.num_workers=10 \
--data.val_dataloader.num_workers=10 \
--data.train_dataloader.batch_size=128 \
--data.val_dataloader.batch_size=128 \
--data.train_dataset.batch_size=128 \
--data.val_dataset.batch_size=128 \
--data.train_dataset.shuffle=False
--data.train_dataset.local=data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/train
--data.val_dataset.local=data/mds_tokenized/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val
# --trainer_model.eval_save_batch_interval=1000 \
# --data.train_dataset.epoch_size=1000 \
# --data.val_dataset.epoch_size=1000 \
# --trainer_model.eval_dir=<previous_eval_dir> # to resume from previous run, thanks to stateful dataloader
```

Time: 21h for encode.
