# 250125-methylfoundation-tcga

## Train

```bash
sbatch --exclude=c0103 scripts/experiments/250125-methylfoundation-tcga/c2b2/tcga_mix_chr1-bs_512.sh
```

Note: gene dim for ENCODE is 24337, not 25017.
