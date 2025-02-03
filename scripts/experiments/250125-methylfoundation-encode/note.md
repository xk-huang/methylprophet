# 250125-methylfoundation-encode

## Train

```bash
master_addr="172.31.21.59"
master_port=51344 \
num_nodes=8 \
num_processes=8 \
bash scripts/experiments/250125-methylfoundation-encode/aws/encode_wgbs-bs_512.sh
```

Note: gene dim for ENCODE is 24337, not 25017.
