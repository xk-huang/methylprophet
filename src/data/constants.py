CHR_LIST = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrx",
    "chry",
]
CHR_IDX_MAPPING = {chr_: idx for idx, chr_ in enumerate(CHR_LIST)}
CHR_IDX_MAPPING_INV = dict(enumerate(CHR_LIST))

NBASE_MAPPING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}
