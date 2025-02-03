declare -a params=(
    "241213-tcga_array"
    "241213-tcga_wgbs"
    "241213-encode_wgbs"
)

# Get parameters for this task
for DATASET in ${params[@]}; do
    echo "DATASET: $DATASET"
    OUTPUT_LOG="data/parquet/$DATASET/check_cpg.log"
    python scripts/tools/read_parquet.py --i data/parquet/$DATASET/me.parquet --columns='Unnamed: 0' > $OUTPUT_LOG 2>&1
    python scripts/tools/read_parquet.py --i data/parquet/$DATASET/cpg_bg.parquet/ --columns='CpG_location' >> $OUTPUT_LOG 2>&1
    python scripts/tools/read_parquet.py --i data/parquet/$DATASET/cpg_island.parquet/ --columns='cpg' >> $OUTPUT_LOG 2>&1
    echo "Done: $DATASET"
done
