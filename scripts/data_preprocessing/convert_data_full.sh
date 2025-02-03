set -e
NUM_WORKERS=120 bash scripts/data_preprocessing/241213-encode.sh

# No sequence tokenization, 1x time but 3x space
# NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-encode.sh

# Tokenize sequence, takes 3x more time, but use 1x space
NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-tokenized-encode.sh


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