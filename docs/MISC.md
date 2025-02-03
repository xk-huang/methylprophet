# MISC


## UCSC VLAA Machines

### Pull logs from other nodes

```shell
# server_id_ls=(03 04 06 08 10)
server_id_ls=(03 04 06 08)
base_dir=$(pwd)

for server_id in ${server_id_ls[@]}; do
    rsync -avP --checksum vlaa-${server_id}:${base_dir}/outputs/ ${base_dir}/outputs/ 
done
```

### Send data to other nodes

```shell
# server_id_ls=( 03 04 08 10 )
server_id_ls=( 03 08 )
base_dir=$(pwd)

for server_id in ${server_id_ls[@]}; do
    rsync -avP ${base_dir}/data/processed/241023-encode_wgbs-* vlaa-${server_id}:${base_dir}/data/processed
done
```

## Tarball results

```shell
output_dir=$(pwd)/outputs
results=$(find outputs \( -name 'eval_results-*' -not -path '*train_shuffle*' \))
# path: outputs/240912-slurm-yf_model/encode_wgbs-1e5x1e2-val_ind_tissue-yf_model_tiny-5xa5000/eval/version_0/eval_results-test-val_ind_cpg_ind_sample.csv

# NOTE xk: remove temp dir in all cases. ref: https://stackoverflow.com/a/22644006
# ref: https://stackoverflow.com/questions/687014/removing-created-temp-files-in-unexpected-bash-exit
MYTMPDIR="$(mktemp -d)"
trap 'rm -rf -- "$MYTMPDIR"' EXIT

for result in $results; do
    exp_dir=$(echo $result | cut -d'/' -f2)
    exp_name=$(echo $result | cut -d'/' -f3)
    full_exp_name="${exp_dir}-${exp_name}"
    mkdir -p $MYTMPDIR/$full_exp_name
    cp $result $MYTMPDIR/$full_exp_name
done
ls $MYTMPDIR/*

output_tar_file=$output_dir/eval_results.$(date +"%y%m%d_%H%M%S").tar.gz
# NOTE xk: -C to temporarily change the directory, in both -c (compress) and -x (decompress).
tar -czvf $output_tar_file -C $MYTMPDIR .
```

## Tar results

```shell
output_dir=$(pwd)/outputs
results=$(find outputs \( -name 'eval_results-*' -not -path '*train_shuffle*' \))
# path: outputs/240912-slurm-yf_model/encode_wgbs-1e5x1e2-val_ind_tissue-yf_model_tiny-5xa5000/eval/version_0/eval_results-test-val_ind_cpg_ind_sample.csv

# NOTE xk: remove temp dir in all cases. ref: https://stackoverflow.com/a/22644006
# ref: https://stackoverflow.com/questions/687014/removing-created-temp-files-in-unexpected-bash-exit
MYTMPDIR="$(mktemp -d)"
trap 'rm -rf -- "$MYTMPDIR"' EXIT

for result in $results; do
    exp_dir=$(echo $result | cut -d'/' -f2)
    exp_name=$(echo $result | cut -d'/' -f3)
    full_exp_name="${exp_dir}-${exp_name}"
    mkdir -p $MYTMPDIR/$full_exp_name
    cp $result $MYTMPDIR/$full_exp_name
done

output_tar_file=$output_dir/eval_results.$(date +"%y%m%d_%H%M%S").tar.gz
tar -czvf $output_tar_file -C $MYTMPDIR .

# Include additional files
additional_files=(
    "data/processed/encode_wgbs-240802-1e4x1e2-val_ind_tissue/cpg_chr_pos_to_idx.csv"
    "data/processed/encode_wgbs-240802-1e4x1e2-val_ind_tissue/sample_name_to_idx.csv"
)

NAME_TO_IDX_DIR=$MYTMPDIR/name_to_idx
mkdir -p $NAME_TO_IDX_DIR

for file in "${additional_files[@]}"; do
    cp $file $NAME_TO_IDX_DIR
done

ls $MYTMPDIR/*
output_tar_file=$output_dir/name_to_idx.tar.gz
tar -czvf $output_tar_file -C $NAME_TO_IDX_DIR .
```

## Time your shell script

```shell
SECONDS=0 && \
sleep 10 && \
duration=$SECONDS && hours=$((duration / 3600)) && minutes=$(( (duration % 3600) / 60 )) && seconds=$((duration % 60)) && printf "Time taken: %02d:%02d:%02d\n" $hours $minutes $seconds
```
