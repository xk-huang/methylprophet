import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import tqdm
from absl import app, flags, logging

flags.DEFINE_string(
    "me_parquet_dir",
    "data/processed/tcga_450k-240802/me_cpg_dataset/train.parquet",
    "Directory of the me parquet data",
)
flags.DEFINE_boolean("overwrite", False, "Overwrite the existing parquet files")
flags.DEFINE_integer("num_workers", 20, "Number of workers to use for data loading")
flags.DEFINE_integer("batch_size", 20, "Number of batch size to use for data loading")
flags.DEFINE_integer("num_data_points_per_shard", 10000, "Number of data points per shard")


FLAGS = flags.FLAGS


class MECPGDataset(torch.utils.data.IterableDataset):
    def __init__(self, me_parquet_files):
        self.me_parquet_files = me_parquet_files

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            logging.info("Single-process data loading, set worker_id to 0, num_workers to 1")
            worker_id = 0
            num_workers = 1
        else:  # in a worker process
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            logging.info(f"Worker {worker_id} of {num_workers}")

        tic = time.time()
        for idx, me_parquet_file in enumerate(self.me_parquet_files):
            me_sharded_df = pd.read_parquet(me_parquet_file)
            if idx != 0:
                toc = time.time()
                remaining_time = (toc - tic) / (idx + 1) * (len(self.me_parquet_files) - idx - 1)
            else:
                remaining_time = -1
            if worker_id == 0:
                logging.info(
                    f"[worker {worker_id}] Reading {me_parquet_file}, length: {len(me_sharded_df)}, remaining time: {remaining_time:.2f}s ({remaining_time / 60:.2f}m) ({remaining_time / 3600:.2f}h)"
                )

            for i in range(len(me_sharded_df)):
                if i % num_workers != worker_id:
                    continue
                data = me_sharded_df.iloc[i]
                yield data


def get_columns(me_parquet_file):
    return pd.read_parquet(me_parquet_file).columns


def main(_):
    me_parquet_dir = Path(FLAGS.me_parquet_dir)
    output_me_parquet_dir = me_parquet_dir.parent / f"{me_parquet_dir.name}_reordered"

    if output_me_parquet_dir.exists():
        if FLAGS.overwrite:
            shutil.rmtree(output_me_parquet_dir)
            logging.info(f"Removed existing {output_me_parquet_dir}")
        else:
            raise ValueError(f"{output_me_parquet_dir} already exists, use --overwrite to overwrite")

    output_me_parquet_dir.mkdir(parents=True, exist_ok=True)

    me_parquet_files = sorted(me_parquet_dir.glob("*.parquet"))

    num_data_points_per_shard = FLAGS.num_data_points_per_shard
    if num_data_points_per_shard % FLAGS.batch_size != 0:
        raise ValueError(
            f"num_data_points_per_shard must be divisible by batch, got {num_data_points_per_shard} and {FLAGS.batch_size}"
        )
    num_batches_per_shard = num_data_points_per_shard // FLAGS.batch_size

    dataset = MECPGDataset(me_parquet_files)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, collate_fn=lambda x: x
    )

    columns = get_columns(me_parquet_files[0])
    data_point_ls = []
    reordered_idx = 0
    for batch_idx, batch_data_point in tqdm.tqdm(enumerate(dataloader)):
        if batch_idx % num_batches_per_shard == 0:
            if data_point_ls:
                shard_df = pd.DataFrame(data_point_ls, columns=columns)

                output_me_parquet_file = output_me_parquet_dir / f"{reordered_idx:05d}.parquet"
                shard_df.to_parquet(output_me_parquet_file)
                logging.info(f"Saved {output_me_parquet_file}")

                data_point_ls = []
                reordered_idx += 1

        data_point_ls.extend(batch_data_point)


if __name__ == "__main__":
    app.run(main)
