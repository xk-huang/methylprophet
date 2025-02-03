"""
Save the data as ThunderDB format.
The mapping is `sample_name` -> all cpg me values whose index is `cpg_id`
The other keys include `keys-sample_name` and `keys-cpg_id`.
See `read_tdb.py` for how to read the data.

NOTE xk: CpG pos_name has duplicated values. We need cpg id to differentiate them.
Specifically, 393309 CpG sites -> 393292 unique CpG sites.
"""

import datetime
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import tqdm
from thunderpack import ThunderDB


if len(sys.argv) != 6:
    raise ValueError("Please provide data_dir, save_dir, num_total_cpgs, num_rows, and num_cols.")
data_dir = sys.argv[1]
save_dir = sys.argv[2]
NUM_CPG = int(sys.argv[3])
NUM_ROWS = int(sys.argv[4])
NUM_COLS = int(sys.argv[5])

data_dir = Path(data_dir)
save_dir = Path(save_dir)

dnam_file_name = "me_rownamesloc.csv"  # 393309 cpg sites X 8578 samples, 57GB
gene_expr_file_name = "ge.csv"  # 58560 genes X 8578 samples, 4.5GB
cancer_type_file_name = "project.csv"  # 8578 samples X 2 {sample name, cancer type} pair, 0.2MB
cpg_bg_file_name = (
    "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB
)

cpg_bg_dna_seq_path = data_dir / cpg_bg_file_name

cpg_bg_df = pd.read_csv(cpg_bg_dna_seq_path)
cpg_bg_df.rename(columns={"CpG_name": "cpg_id", "CpG_location": "cpg_chr_pos"}, inplace=True)


cpg_me_path = data_dir / dnam_file_name

save_dir.mkdir(parents=True, exist_ok=True)

save_cpg_me_path = save_dir / "cpg_me.tdb"
if save_cpg_me_path.exists():
    shutil.rmtree(save_cpg_me_path)


def get_columns_from_csv(path, chunk_size=10):
    with pd.read_csv(path, chunksize=chunk_size) as reader:
        chunk = next(reader)
        return chunk.columns


def load_csv_by_cols(path, cols=None, chunk_size=10000, num_rows=None):
    df = []
    num_loaded_rows = 0
    with pd.read_csv(path, chunksize=chunk_size, usecols=cols) as reader:
        for chunk in tqdm.tqdm(reader, desc="Loading cols", total=NUM_CPG // chunk_size):
            df.append(chunk)

            num_loaded_rows += len(chunk)
            if num_rows is not None and num_loaded_rows >= num_rows:
                break
    df = pd.concat(df)
    if num_rows is not None:
        df = df.iloc[:num_rows]
    return df


def convert_cpg_me_df(cpg_me_df):
    # NOTE xk: column 0 is `gene_id`, but it is named in `Unnamed: 0`.
    cpg_me_df.rename(columns={"Unnamed: 0": "cpg_chr_pos"}, inplace=True)

    if not cpg_me_df["cpg_chr_pos"].isin(cpg_bg_df["cpg_chr_pos"]).all():
        raise ValueError("cpg_chr_pos does not match.")
    cpg_me_df.drop("cpg_chr_pos", axis=1, inplace=True)

    cpg_me_df["cpg_id"] = cpg_bg_df["cpg_id"]
    cpg_me_df.set_index("cpg_id", inplace=True)
    return cpg_me_df


def process_one_col_in_csv(args):
    if args is None:
        queue.put(None)
        return
    col_chunk_start_id, col_chunk_end_id, num_rows = args
    col_idx = [0] + list(range(col_chunk_start_id, col_chunk_end_id))
    cpg_me_col_df = load_csv_by_cols(path=cpg_me_path, cols=col_idx, num_rows=num_rows)
    cpg_me_col_df = convert_cpg_me_df(cpg_me_col_df)
    queue.put(cpg_me_col_df)


def save_col_df_to_tdb(col_df):
    with ThunderDB.open(str(save_cpg_me_path), "c") as db:
        for key in col_df.columns:
            db[key] = col_df[key]


def write_queue(queue):
    print("Start writing queue...")

    cpg_me_columns = get_columns_from_csv(cpg_me_path)
    # NOTE xk: column 0 is `cpg_chr_pos`, but it is named in `Unnamed: 0`.
    cpg_me_columns = cpg_me_columns[1:]
    num_cols = len(cpg_me_columns)
    with ThunderDB.open(str(save_cpg_me_path), "c") as db:
        db["keys-sample_name"] = cpg_me_columns
    print(f"Total number of columns: {num_cols}")

    num_saved_samples = 0
    df = None
    while True:
        item = queue.get()
        if item is None:
            print("Received None, break...")
            break
        else:
            df = item

        # NOTE: sample names
        columns = df.columns
        save_col_df_to_tdb(df)

        num_saved_samples += len(columns)
        print(
            f"Saving column {columns}... Total samples: {num_cols}, Saved samples: {num_saved_samples}, saved rows {len(df)}."
        )

    with ThunderDB.open(str(save_cpg_me_path), "c") as db:
        any_sample_name = columns[0]
        cpg_id = df[any_sample_name].index
        db["keys-cpg_id"] = cpg_id


if __name__ == "__main__":
    queue = multiprocessing.Queue()

    writer_process = multiprocessing.Process(target=write_queue, args=(queue,))
    writer_process.start()

    cpg_me_columns = get_columns_from_csv(cpg_me_path)
    # NOTE xk: column 0 is `cpg_chr_pos`, but it is named in `Unnamed: 0`.
    cpg_me_columns = cpg_me_columns[1:]
    num_all_cols = len(cpg_me_columns)
    # Make sure we can load some data without any bug.
    process_one_col_in_csv((0, 1, 22))

    num_rows = NUM_ROWS
    num_cols = NUM_COLS
    if num_cols > num_all_cols:
        print(f"num_cols: {num_cols} > num_all_cols: {num_all_cols}. Set num_cols to num_all_cols.")

        num_cols = num_all_cols
    col_chunk_size = 100
    # NOTE xk: maximum idx is num_cols - 1 for 0 base. But there is an addtional column at the start,
    # so it should be num_cols. The range is num_cols + 1.
    col_chunk_range = list(range(1, num_cols + 1, col_chunk_size))
    args = [
        (chunk_start_id, min(chunk_start_id + col_chunk_size, num_cols + 1), num_rows)
        for chunk_start_id in col_chunk_range
    ]

    num_cores = 50
    tic = time.time()
    with multiprocessing.Pool(num_cores) as p:
        # NOTE xk: tqdm.tqdm does work with multiprocessing.Pool.imap https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
        r = list(tqdm.tqdm(p.imap(process_one_col_in_csv, args), total=num_cols, desc="Multiprocessing"))

        # NOTE xk: send None to queue to stop writer process. https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
        p.close()
        p.join()
        queue.put(None)

    period = time.time() - tic
    format_period = str(datetime.timedelta(seconds=period))
    print(f"Processing time: {format_period}")
