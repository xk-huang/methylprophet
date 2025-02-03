"""
The following script is used to convert the CpG island tuple to parquet format.

The format of the CpG island tuple is as follows:
```
{
    'cpg': 'chr17_48723082',
    'cpg_island_tuple': 'upshore1_22236,upshore1_22237,upshore2_22237,upshore3_22238,shelve_22239,shelve_22240',
    'num_cpg_island_tuple': 6
}

{
    'cpg': 'chr1_9071724',
    'cpg_island_tuple': 'sea_-1',
    'num_cpg_island_tuple': 1
}
```
"""

import shutil
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_cpg_island_parquet", None, "Path to the CpG island parquet file")
flags.DEFINE_string("input_me_parquet", None, "Path to the ME parquet file")
flags.mark_flag_as_required("input_me_parquet")
flags.mark_flag_as_required("input_cpg_island_parquet")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("output_file_name", None, "Output file name")
flags.mark_flag_as_required("output_dir")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_integer("row_chunk_size", 10000, "Row chunk size")

flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")

FLAGS = flags.FLAGS


def main(_):
    output_dir = Path(FLAGS.output_dir)
    output_dir = output_dir / FLAGS.output_file_name
    output_dir = prepare_output_dir(output_dir, FLAGS.overwrite)

    cpg_island_df = pd.read_parquet(FLAGS.input_cpg_island_parquet)
    # fill NaN (cpg sea) with -1
    cpg_island_df["cgiIndex"] = cpg_island_df["cgiIndex"].fillna(-1)
    # convert cpg_island_df["cgiIndex"] to int
    cpg_island_df["cgiIndex"] = cpg_island_df["cgiIndex"].astype(int)

    cpg_island_df["cpg_island_tuple"] = cpg_island_df["location"] + "_" + cpg_island_df["cgiIndex"].astype(str)
    cpg_island_df = (
        cpg_island_df[["cpg", "cpg_island_tuple"]].groupby("cpg")["cpg_island_tuple"].agg(lambda x: ",".join(x))
    )
    cpg_island_df = cpg_island_df.reset_index()

    cpg_island_df["num_cpg_island_tuple"] = cpg_island_df["cpg_island_tuple"].str.count(",") + 1
    cpg_island_df = cpg_island_df.set_index("cpg")

    cpg_index = pd.read_parquet(FLAGS.input_me_parquet, columns=["Unnamed: 0"])
    cpg_index = cpg_index["Unnamed: 0"]

    if cpg_index.sort_values().reset_index(drop=True).equals(pd.Series(cpg_island_df.index.sort_values())) is False:
        logging.warning(
            f"ME CpG index does not match CpG island index for {FLAGS.input_me_parquet} and {FLAGS.input_cpg_island_parquet}"
        )

    cpg_island_df = cpg_island_df.loc[cpg_index]
    cpg_island_df = cpg_island_df.reset_index()

    # write to parquet shards
    row_chunk_size = FLAGS.row_chunk_size
    logging.info(f"Converting cpg island tuple to {output_dir} with row chunk size: {row_chunk_size}")

    for i, start_idx in enumerate(tqdm.trange(0, len(cpg_island_df), row_chunk_size)):
        cpg_island_df.iloc[start_idx : start_idx + row_chunk_size].to_parquet(
            output_dir / f"{i:05d}.parquet", index=True
        )


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if overwrite:
            logging.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            logging.warning(f"Output directory already exists: {output_dir}, skipping...")
            exit()

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    return output_dir


if __name__ == "__main__":
    app.run(main)
