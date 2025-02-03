import multiprocessing
import shutil
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("me_parquet_dir", "data/parquet/encode_wgbs-240802/me.parquet", "Directory of the me parquet data")
flags.DEFINE_string(
    "save_cpg_bg_parquet_dir",
    "data/parquet/encode_wgbs-240802/cpg_bg.parquet",
    "Directory to save the cpg_bg parquet files",
)
flags.DEFINE_string("hg38_parquet_file", "data/parquet/grch38_hg38/hg38.fa.parquet", "hg38 parquet file")
flags.DEFINE_boolean("overwrite", False, "Overwrite the existing parquet files")
flags.DEFINE_integer("cpg_bg_nbase_length", 1200, "Number of bases to extract from the cpg_bg file")
flags.DEFINE_integer("num_workers", multiprocessing.cpu_count(), "Number of worker processes to use")

FLAGS = flags.FLAGS

# Global variable for hg38_df to be shared among processes
hg38_df = None


def init_process(hg38_parquet_file):
    """
    Initializer function for each worker process.
    Reads the hg38 parquet file and stores it in a global variable.
    """
    global hg38_df
    logging.info(f"Reading {hg38_parquet_file}")
    hg38_df = pd.read_parquet(hg38_parquet_file)
    logging.info(f"Read {hg38_df.shape[0]} rows")

    # Ensure there is no duplicated chr
    if hg38_df["chr"].duplicated().sum() > 0:
        raise ValueError("Duplicated chr in hg38_df")
    hg38_df.set_index("chr", inplace=True)


def add_sequence_to_chr_pos_df(hg38_df, cpg_bg_nbase_length):
    def _add_sequence_to_chr_pos_df(row):
        chr, pos = row["chr"], row["pos"]
        seq, seq_len = hg38_df.loc[chr]

        try:
            pos = int(pos)
        except ValueError:
            pos = int(eval(pos))  # Handle scientific notation

        cg_nbase = seq[pos]
        if cg_nbase.upper() not in ("C", "G"):
            raise ValueError(f"Invalid CpG base {cg_nbase} at {chr}:{pos}")

        left_pos = pos - cpg_bg_nbase_length // 2
        right_pos = pos + cpg_bg_nbase_length // 2

        left_pad_len = max(0, -left_pos)
        right_pad_len = max(0, right_pos - seq_len)

        left_pos = max(0, left_pos)
        right_pos = min(seq_len, right_pos)

        cpg_bg_seq = seq[left_pos:right_pos]
        cpg_bg_seq = "N" * left_pad_len + cpg_bg_seq + "N" * right_pad_len

        if left_pad_len > 0 or right_pad_len > 0:
            logging.warning(f"Padding {chr}:{pos} with {left_pad_len} left and {right_pad_len} right")

        return cpg_bg_seq

    return _add_sequence_to_chr_pos_df


def process_parquet_file(me_parquet_file):
    """
    Function to process a single parquet file.
    Reads the file, processes it, and saves the output.
    """
    global hg38_df
    cpg_bg_nbase_length = FLAGS.cpg_bg_nbase_length

    me_sharded_df = pd.read_parquet(me_parquet_file)

    # Rename columns if necessary
    me_sharded_df.rename(columns={"Unnamed: 0": "CpG_location"}, inplace=True)
    cpg_bg_df = me_sharded_df["CpG_location"].str.split("_", expand=True)
    cpg_bg_df.columns = ["chr", "pos"]

    # Apply the sequence extraction function
    cpg_bg_df["sequence"] = cpg_bg_df.apply(add_sequence_to_chr_pos_df(hg38_df, cpg_bg_nbase_length), axis=1)
    cpg_bg_df.rename(columns={"chr": "cpg_chr", "pos": "cpg_pos"}, inplace=True)

    cpg_bg_df["CpG_location"] = me_sharded_df["CpG_location"]
    cpg_bg_df["CpG_name"] = me_sharded_df.index
    cpg_bg_df = cpg_bg_df[["CpG_name", "CpG_location", "sequence"]]

    # Save the processed dataframe
    save_filepath = Path(FLAGS.save_cpg_bg_parquet_dir) / me_parquet_file.name
    cpg_bg_df.to_parquet(save_filepath)

    return me_parquet_file.name  # Return the filename for logging


def main(_):
    me_parquet_dir = Path(FLAGS.me_parquet_dir)
    save_cpg_bg_parquet_dir = Path(FLAGS.save_cpg_bg_parquet_dir)
    hg38_parquet_file = Path(FLAGS.hg38_parquet_file)

    if save_cpg_bg_parquet_dir.exists():
        if not FLAGS.overwrite:
            print(f"{save_cpg_bg_parquet_dir} already exists. Skipping...")
            return
        else:
            shutil.rmtree(save_cpg_bg_parquet_dir)
    save_cpg_bg_parquet_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of parquet files to process
    me_parquet_files = sorted(me_parquet_dir.glob("*.parquet"))

    # Set up multiprocessing pool
    num_workers = FLAGS.num_workers
    print(f"Using {num_workers} worker processes")
    with multiprocessing.Pool(processes=num_workers, initializer=init_process, initargs=(hg38_parquet_file,)) as pool:
        # Use tqdm to display progress
        with tqdm.tqdm(total=len(me_parquet_files)) as pbar:
            for result in pool.imap_unordered(process_parquet_file, me_parquet_files):
                pbar.set_postfix_str(f"Processed {result}")
                pbar.update()

    logging.info(f"Processing complete. Saved to {save_cpg_bg_parquet_dir}")


if __name__ == "__main__":
    app.run(main)
