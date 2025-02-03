# AWS

## Common

Run commands for all servers

```bash
# sequential
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "gpustat"
done

# parallel
output_files=()
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    temp_file=$(mktemp)
    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; ssh $target_node "gpustat" >> $temp_file) &
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"
```

## Setup

### Node list
```shell
Host 250105-aws-64-l40s*
    IdentityFile ~/.ssh/pretrain.pem
    User ubuntu
    ForwardAgent yes

Host 250105-aws-64-l40s-0
    HostName ec2-54-82-87-57.compute-1.amazonaws.com
Host 250105-aws-64-l40s-1
    HostName ec2-174-129-159-21.compute-1.amazonaws.com
Host 250105-aws-64-l40s-2
    HostName ec2-54-225-49-221.compute-1.amazonaws.com
Host 250105-aws-64-l40s-3
    HostName ec2-50-19-63-26.compute-1.amazonaws.com
Host 250105-aws-64-l40s-4
    HostName ec2-54-90-227-248.compute-1.amazonaws.com
Host 250105-aws-64-l40s-5
    HostName ec2-18-212-215-191.compute-1.amazonaws.com
Host 250105-aws-64-l40s-6
    HostName ec2-23-22-228-118.compute-1.amazonaws.com
Host 250105-aws-64-l40s-7
    HostName ec2-54-80-160-39.compute-1.amazonaws.com

for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node 'hostname -I | cut -d " " -f1'
done

172.31.21.59
172.31.21.41 slots=8
172.31.27.181 slots=8
172.31.30.84 slots=8
172.31.31.20 slots=8
172.31.21.64 slots=8
172.31.17.80 slots=8
172.31.31.128 slots=8


250105-aws-64-l40s-0 172.31.21.59
250105-aws-64-l40s-1 172.31.21.41
250105-aws-64-l40s-2 172.31.27.181
250105-aws-64-l40s-3 172.31.30.84
250105-aws-64-l40s-4 172.31.31.20
250105-aws-64-l40s-5 172.31.21.64
250105-aws-64-l40s-6 172.31.17.80
250105-aws-64-l40s-7 172.31.31.128


for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "ping -c 3 172.31.21.59 && echo yes $target_node"
done
```

### Storage

```bash
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node 'mkdir -p efs && sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 172.31.21.28:/ efs'
done
```

backup data
```bash
ls /home/ubuntu/efs/xiaoke
mkdir -p /home/ubuntu/efs/xiaoke/codes/methylformer/data/extracted/
mkdir -p /home/ubuntu/efs/xiaoke/codes/methylformer/data/raw/
rsync -ah --info=progress2 /opt/dlami/nvme/xiaoke/codes/methylformer/data/extracted/ /home/ubuntu/efs/xiaoke/codes/methylformer/data/extracted/
rsync -ah --info=progress2 /opt/dlami/nvme/xiaoke/codes/methylformer/data/raw/ /home/ubuntu/efs/xiaoke/codes/methylformer/data/raw/
```


### EFA problem
```bash
# EFA problem on all nodes
# https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-efa-launching.html
fi_info -p efa

for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node '/opt/amazon/efa/bin/fi_info -p efa'
done

# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)
# fi_getinfo: -61 (No data available)

# The docs of nccl test on aws: https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-efa-launching.html
# Choose 172.31.21.59 as lead node, try nccl communication (torch cuda dist backend).
# failed

/opt/amazon/openmpi/bin/mpirun -n 2 -N 1 --hostfile ~/hosts \
-x LD_LIBRARY_PATH=/usr/local/cuda-12.4/efa/lib:/usr/local/cuda-12.4/lib:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4:$LD_LIBRARY_PATH \
--mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
/opt/aws-ofi-nccl/tests/nccl_message_transfer

# A process or daemon was unable to complete a TCP connection
# to another process:
#   Local host:    ip-172-31-27-181
#   Remote host:   ip-172-31-21-59
# This is usually caused by a firewall on the remote host. Please
# check that any firewall (e.g., iptables) has been disabled and
# try again.


# torch DDP jobs
# nccl_net_ofi_rdma_init:7737 NCCL WARN NET/OFI OFI fi_getinfo() call failed: No data available
# nccl_net_ofi_create_plugin:261 NCCL WARN NET/OFI Unable to find a protocol that worked.  Failing initialization.
# nccl_net_ofi_create_plugin:341 NCCL WARN NET/OFI aws-ofi-nccl initialization failed
# nccl_net_ofi_init:134 NCCL WARN NET/OFI Initializing plugin failed
```

Security group problem https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-efa-launching.html
```bash
cd /opt/amazon/efa/test/
./efa_test.sh
# Localhost fi_pingpong test: Attempt 1 (max 3)...
# Starting server on port 57280...
# Starting client on port 64479...
# Server Log:
# libfabric:493076:1736319850::efa:domain:efa_domain_hmem_info_init_cuda():169<warn> Failed to register CUDA buffer with the EFA device, FI_HMEM transfers that require peer to peer support will fail.
# Client Log:
# libfabric:493096:1736319850::efa:domain:efa_domain_hmem_info_init_cuda():169<warn> Failed to register CUDA buffer with the EFA device, FI_HMEM transfers that require peer to peer support will fail.
# Error: fi_pingpong test failed because of timeout.
```


### Shell Env

copy key
```bash
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    rsync -avP ~/.ssh/pretrain.pem $target_node:/home/ubuntu/.ssh --delete &
    rsync -avP ~/.ssh/config $target_node:/home/ubuntu/.ssh --delete &
done


for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node 'echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCqdfpbKYWOAxKF3B2aQ8si3KKXPF+4YXamnjCza6AGODxvuq72mRRdbLQxKFHORtjtOX3OR09mACDJahZMD+6whKSwDcgnNuiTDed6L6OVieAi8fXflF/b13ev9jd+1tjBCrtP/qd9Csnavt4a343MgFmslIiOt+NKT3Zmo1ebimFH731ge1CTkceaqZJLy2YKEW9XLhGfu9SFcv+7oJaKNBjOn3Yz2AXEtbhZ142p2svcR6/hKdZYqhA26zTNZszeows3/zOSFHpChg/YdJag/SDREJH/+YIeYMzoBgSp6owj8zAh81CGH++z6rmn1mYjLODvEYE1MMdKQFZ6K60ttMpqs9vs7HHxzBHlN2QHzDbksfzY6fgSd2ocNXpRsaXoEqzN7l58eeH12EbkCLF6B2CaCmNOdCTyFuaOruB4QExKOZDwCTIxYQU3/YaeVOZDL05wUEzu3mPBZRbKG7VNXsl/FwWsn3nuHD9hwapLXbNnnxG3xBWGKyx2SQEzfd8= ubuntu@ip-172-31-21-59" >> ~/.ssh/authorized_keys'
done

for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node 'hostname -I | cut -d " " -f1'
done
```

Shell env

```shell
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node 'bash -c "$(curl -fsSL https://raw.githubusercontent.com/xk-huang/dotfiles/main/scripts/setup_env.sh)"'
done

# ssh config, for node finding
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    rsync -avP /home/ubuntu/.ssh/config $target_node:/home/ubuntu/.ssh/config
done
```

The storage are not shared between nodes

```bash
ROOT_DIR=/opt/dlami/nvme
mkdir -p $ROOT_DIR/xiaoke/{codes,data,misc}
mkdir -p $ROOT_DIR/xiaoke/codes/methylformer

BASE_DIR=$ROOT_DIR/xiaoke/codes/methylformer
cd $BASE_DIR
```

## Code
Sync code from node 0
```shell
source_dir=/opt/dlami/nvme/xiaoke/codes/methylformer/
target_dir=/opt/dlami/nvme/xiaoke/codes/methylformer/

for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "mkdir -p $target_dir"

    rsync -avP ${source_dir}/.git $target_node:$target_dir --delete
    rsync -avP ${source_dir}/.env $target_node:$target_dir --delete

    ssh $target_node "cd $target_dir && pwd && git reset --hard main && git status"
done
```

Sync code from my desktop

```shell
source_dir=/insomnia001/depts/houlab/users/xh2689/codes/methylformer/
source_node=xh2689@insomnia.rcs.columbia.edu
temp_dir=/tmp/rsync_temp/
rsync -avP $source_node:$source_dir/.git ${temp_dir} --delete
rsync -avP $source_node:$source_dir/.env ${temp_dir} --delete

target_dir=/opt/dlami/nvme/xiaoke/codes/methylformer/

for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "mkdir -p $target_dir"

    rsync -avP ${temp_dir}/.git $target_node:$target_dir --delete
    rsync -avP ${temp_dir}/.env $target_node:$target_dir --delete

    ssh $target_node "cd $target_dir && pwd && git reset --hard main && git status"
done
rm -rf $temp_dir
```

## Env

Install env `scripts/env/aws/env.sh`

Test torch:
```shell
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "source activate base && \
    conda activate xiaoke-methylformer && \
    python -c 'import torch; torch.ones(100).cuda() * 1000' && \
    root_dir=/opt/dlami/nvme && \
    base_dir=$root_dir/xiaoke/codes/methylformer && \
    cd $base_dir && \
    pip install -e ."
    if [[ $? != 0 ]]; then
        echo "failed for $target_node"
        break
    else
        echo "success for $target_node"
    fi
done
```

Install gpustat

```bash
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "pip install gpustat" &
done
```

## Data

```shell
bash scripts/env_setup/aws/download_data-encode.sh
```

After finish on one node, rsync to others.
```shell
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id

    ssh $target_node "mkdir -p /opt/dlami/nvme/xiaoke/codes/methylformer/data/extracted/"
    ssh $target_node "mkdir -p /opt/dlami/nvme/xiaoke/codes/methylformer/data/raw/"

    rsync -ah --info=progress2 /opt/dlami/nvme/xiaoke/codes/methylformer/data/extracted/ $target_node:/opt/dlami/nvme/xiaoke/codes/methylformer/data/extracted/
    rsync -ah --info=progress2 /opt/dlami/nvme/xiaoke/codes/methylformer/data/raw/ $target_node:/opt/dlami/nvme/xiaoke/codes/methylformer/data/raw/
done
```

```shell
bash scripts/env_setup/aws/process_data-encode.sh
```

```shell
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "bash /opt/dlami/nvme/xiaoke/codes/methylformer/scripts/env_setup/aws/process_data-encode.sh" &
done
```



On each node:

```bash
root_dir=/opt/dlami/nvme && \
base_dir=$root_dir/xiaoke/codes/methylformer && \
cd $base_dir

source activate base && \
conda activate xiaoke-methylformer
pip install -e .

du -sh data/processed

NUM_WORKERS=120 bash scripts/data_preprocessing/241221-convert_mds-encode.sh
```

## 250106 Eval from HF models

download model
```bash
# Download Model
# export HF_TOKEN=hf_?
huggingface-cli whoami
REPO_URL=241229-train-encode-base-64xl40s
REPO_URL=xk-huang/${REPO_NAME}
REPO_DIR=outputs/ckpts/${REPO_NAME}
mkdir -p $REPO_DIR
huggingface-cli download --repo-type model --local-dir $REPO_DIR ${REPO_URL}

# Sync model
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "mkdir -p /opt/dlami/nvme/xiaoke/codes/methylformer/outputs/ckpts/"
    rsync -ah --info=progress2 /opt/dlami/nvme/xiaoke/codes/methylformer/outputs/ckpts/${REPO_NAME} $target_node:/opt/dlami/nvme/xiaoke/codes/methylformer/outputs/ckpts/
done


for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "ls /opt/dlami/nvme/xiaoke/codes/methylformer/outputs/ckpts/${REPO_NAME}"
    if [[ $? == 0 ]]; then
        echo "$target_node is ok"
    else
        echo "$target_node is not ok"
    fi
done
```


<!-- Run infer on 8 nodes -->

Node 0
```bash
root_dir=/opt/dlami/nvme && \
base_dir=$root_dir/xiaoke/codes/methylformer && \
cd $base_dir && \
source activate base && \
conda activate xiaoke-methylformer && \
export NCCL_DEBUG=INFO && \
bash scripts/experiments/241229-methylformer_bert/aws/250105-eval.sh
```

<!-- Node 1-7 from node 0
```bash
for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "root_dir=/opt/dlami/nvme && \
    base_dir=\$root_dir/xiaoke/codes/methylformer && \
    cd \$base_dir && \
    source activate base && \
    conda activate xiaoke-methylformer && \
    export NCCL_DEBUG=INFO && \
    bash scripts/experiments/241229-methylformer_bert/aws/250105-eval.sh 8" &
done
echo "All job started"
``` -->




## 250109 Train encode 90% cpg 64 l40s

```bash
root_dir=/opt/dlami/nvme && \
base_dir=$root_dir/xiaoke/codes/methylformer && \
cd $base_dir && \
source activate base && \
conda activate xiaoke-methylformer && \
export NCCL_DEBUG=INFO && \
bash scripts/experiments/241229-methylformer_bert/aws/250109-train-encode-base-64xl40s.sh 8

for node_id in `seq 1 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "root_dir=/opt/dlami/nvme && \
    base_dir=\$root_dir/xiaoke/codes/methylformer && \
    cd \$base_dir && \
    source activate base && \
    conda activate xiaoke-methylformer && \
    export NCCL_DEBUG=INFO && \
    bash scripts/experiments/241229-methylformer_bert/aws/250109-train-encode-base-64xl40s.sh 8" &
done
echo "All job started"


for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    ssh $target_node "pgrep -f 'src.' | xargs kill -9" &
done
echo "All job canceled"


output_files=()
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    temp_file=$(mktemp)
    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; ssh $target_node "gpustat" >> $temp_file) &
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"
```


Find models
```bash
output_files=()
for node_id in `seq 0 1 7`; do
    target_node=250105-aws-64-l40s-$node_id
    temp_file=$(mktemp)
    output_files+=($temp_file)
    (echo "=== $target_node ===" > $temp_file; ssh $target_node "find /opt/dlami/nvme/xiaoke/codes/methylformer/outputs/241229-methylformer_bert-ins -name '*.ckpt'" >> $temp_file) &
done
wait
cat "${output_files[@]}"
rm "${output_files[@]}"
```


## Rsync and Eval ckpt on NVME (separate storage)

```bash
BASE_DIR=/opt/dlami/nvme/xiaoke/codes/methylformer/

cd $BASE_DIR

NODE=250105-aws-64-l40s-7

find . -name 'finished.ckpt' -or -name 'config.yaml' | rsync -avP --files-from=- ${BASE_DIR} ${NODE}:${BASE_DIR} --dry-run
```

Eval

```bash
master_addr=172.31.17.80 \
job_name=250118-train-tcga_chr1-base-beta_0_95-32xl40s \
ckpt_dir=outputs/241229-methylformer_bert-ins/250118-train-tcga_chr1-base-beta_0_95-32xl40s/ckpt/version_9 \
num_nodes=4 \
num_process=8 \
master_port=51244 \
model_config_path="${ckpt_dir}/config.yaml" \
weight_path="${ckpt_dir}/finished.ckpt" \
bash scripts/experiments/241229-methylformer_bert/aws/eval-250116-train-tcga_chr1.sh
```

