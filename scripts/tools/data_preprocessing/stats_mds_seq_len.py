"""
python scripts/tools/data_preprocessing/stats_mds_seq_len.py \
    --num_workers=160 \
    --batch_size=20000 \
    --local=data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.utils
import tqdm
from absl import app, flags, logging
from streaming import StreamingDataset

from transformers import AutoTokenizer


flags.DEFINE_string(
    "local", "data/mds/241213-encode_wgbs-train_0_9_val_0_1-ind_tissue/val_10_shards", "Path to local data"
)
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("num_workers", 0, "Number of workers")
flags.DEFINE_string("dna_tokenizer_name", "zhihan1996/DNABERT-2-117M", "DNA tokenizer name")
flags.DEFINE_integer("num_nbase", 1000, "Number of N bases")


flags.DEFINE_bool("overwrite", False, "Overwrite existing data")
flags.DEFINE_bool("debug", False, "Debug mode")

FLAGS = flags.FLAGS


class ProcessStreamingDataset(StreamingDataset):
    def __init__(
        self,
        dna_tokenizer_name="zhihan1996/DNABERT-2-117M",
        num_nbases=1000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer_name, trust_remote_code=True)
        self.num_nbases = num_nbases

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        dna_seq = data["sequence"].upper()
        mid_pos = len(dna_seq) // 2
        left_pos = max(mid_pos - self.num_nbases // 2, 0)
        right_pos = min(mid_pos + self.num_nbases // 2, len(dna_seq))
        dna_seq = dna_seq[left_pos:right_pos]
        tokens = self.tokenizer.encode(dna_seq, add_special_tokens=False)
        return len(tokens)

    def collate_fn(self, data):
        return np.array(data)


def main(_):
    local = Path(FLAGS.local)
    output_path = local / "dna_seq_stats.json"
    overwrite = FLAGS.overwrite
    if output_path.exists():
        if overwrite is True:
            logging.info(f"Overwrite existing data at {output_path}")
        else:
            logging.info(f"Skip existing data at {output_path}")
            return

    dna_tokenizer_name = FLAGS.dna_tokenizer_name
    num_nbase = FLAGS.num_nbase
    logging.info(f"Local: {local}, dna tokenizer name: {dna_tokenizer_name}, num nbase: {num_nbase}")

    dataset = ProcessStreamingDataset(
        dna_tokenizer_name=dna_tokenizer_name,
        num_nbases=num_nbase,
        local=Path(FLAGS.local),
        batch_size=FLAGS.batch_size,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        num_workers=FLAGS.num_workers,
        collate_fn=dataset.collate_fn,
    )
    logging.info(f"num samples: {len(dataset)}, num batches: {len(data_loader)}")
    logging.info(f"Batch size: {FLAGS.batch_size}, num workers: {FLAGS.num_workers}")

    dna_seq_lens = []
    for data in tqdm.tqdm(data_loader):
        dna_seq_lens.append(data)

    dna_seq = np.concatenate(dna_seq_lens)
    dna_seq_stats = {
        "num_points": len(dna_seq),
        "num_nbase": num_nbase,
        "mean": int(dna_seq.mean()),
        "std": int(dna_seq.std()),
        "min": int(dna_seq.min()),
        "max": int(dna_seq.max()),
    }
    if FLAGS.overwrite or not output_path.exists():
        with open(output_path, "w") as f:
            json.dump(dna_seq_stats, f, indent=4)
        logging.info(f"Save dna seq stats to {output_path}")


if __name__ == "__main__":
    app.run(main)
