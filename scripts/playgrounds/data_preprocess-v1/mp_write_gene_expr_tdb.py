"""
Save the data as ThunderDB format.
The mapping is `sample_name` -> all gene expression values whose index is `gene_id`
The other keys include `keys-sample_name` and `keys-gene_id`.
See `read_tdb.py` for how to read the data.
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


if len(sys.argv) != 3:
    raise ValueError("Please provide data_dir, save_dir.")
data_dir = sys.argv[1]
save_dir = sys.argv[2]

data_dir = Path(data_dir)
save_dir = Path(save_dir)


dnam_file_name = "me_rownamesloc.csv"  # 393309 cpg sites X 8578 samples, 57GB
gene_expr_file_name = "ge.csv"  # 58560 genes X 8578 samples, 4.5GB
cancer_type_file_name = "project.csv"  # 8578 samples X 2 {sample name, cancer type} pair, 0.2MB
cpg_bg_file_name = (
    "CpG_name_location__DNAsequence.csv"  # 393309 cpg sites X 3 {cpg name, cpg location, 100bp DNA seq} columns, 47MB
)

NUM_GENE = 58560

gene_expr_path = data_dir / gene_expr_file_name

save_dir.mkdir(parents=True, exist_ok=True)

save_gene_expr_path = save_dir / "gene_expr.tdb"
if save_gene_expr_path.exists():
    shutil.rmtree(save_gene_expr_path)


def get_columns_from_csv(path, chunk_size=10):
    with pd.read_csv(path, chunksize=chunk_size) as reader:
        chunk = next(reader)
        return chunk.columns


def load_csv_by_cols(path, cols=None, chunk_size=1000):
    df = []
    with pd.read_csv(path, chunksize=chunk_size, usecols=cols) as reader:
        for chunk in tqdm.tqdm(reader, desc="Loading cols", total=NUM_GENE // chunk_size):
            df.append(chunk)
    df = pd.concat(df)
    return df


def strip_gene_id(row):
    try:
        out = row["gene_id"].split(";")[-1]
    except Exception as e:
        print(e)
        breakpoint()
    return out


def convert_gene_expr_df(gene_expr_df):
    # NOTE xk: column 0 is `gene_id`, but it is named in `Unnamed: 0`.
    gene_expr_df.rename(columns={"Unnamed: 0": "gene_id"}, inplace=True)

    # NOTE xk: dropna for gene_id
    gene_expr_df = gene_expr_df.dropna()

    # NOTE xk: strip gene_id to get the last part of the gene_id. e.g., "TSPAN6;ENSG00000000003"
    gene_expr_df["gene_id"] = gene_expr_df.apply(strip_gene_id, axis=1)

    gene_expr_df.set_index("gene_id", inplace=True)
    return gene_expr_df


def process_one_col_in_csv(args):
    if args is None:
        queue.put(None)
        return
    col_chunk_start_id, col_chunk_end_id = args
    col_idx = [0] + list(range(col_chunk_start_id, col_chunk_end_id))
    gene_expr_col_df = load_csv_by_cols(gene_expr_path, col_idx)
    gene_expr_col_df = convert_gene_expr_df(gene_expr_col_df)
    queue.put(gene_expr_col_df)


def save_col_df_to_tdb(col_df):
    with ThunderDB.open(str(save_gene_expr_path), "c") as db:
        for key in col_df.columns:
            db[key] = col_df[key]


def write_queue(queue):
    print("Start writing queue...")

    gene_expr_columns = get_columns_from_csv(gene_expr_path)
    # NOTE xk: column 0 is `gene_id`, but it is named in `Unnamed: 0`.
    gene_expr_columns = gene_expr_columns[1:]
    num_cols = len(gene_expr_columns)
    with ThunderDB.open(str(save_gene_expr_path), "c") as db:
        db["keys-sample_name"] = gene_expr_columns
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
        print(f"Saving column {columns}... Total samples: {num_cols}, Saved samples: {num_saved_samples}.")

    with ThunderDB.open(str(save_gene_expr_path), "c") as db:
        any_sample_name = columns[0]
        gene_id = df[any_sample_name].index
        db["keys-gene_id"] = gene_id


if __name__ == "__main__":
    # process_one_col_in_csv((1, 10))  # For debug
    queue = multiprocessing.Queue()

    writer_process = multiprocessing.Process(target=write_queue, args=(queue,))
    writer_process.start()

    gene_expr_columns = get_columns_from_csv(gene_expr_path)
    # NOTE xk: column 0 is `gene_id`, but it is named in `Unnamed: 0`.
    gene_expr_columns = gene_expr_columns[1:]
    num_cols = len(gene_expr_columns)

    # DEBUG xk:
    # num_cols = 10
    col_chunk = 100
    # NOTE xk: maximum idx is num_cols - 1 for 0 base. But there is an addtional column at the start,
    # so it should be num_cols. The range is num_cols + 1.
    col_chunk_range = list(range(1, num_cols + 1, col_chunk))
    args = [(chunk_start_id, min(chunk_start_id + col_chunk, num_cols + 1)) for chunk_start_id in col_chunk_range]

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
