import multiprocessing
import shutil
from pathlib import Path

import pandas as pd
import tqdm
from absl import app, flags, logging


flags.DEFINE_string("input_me_parquet_file", None, "Path to the parquet file")
flags.DEFINE_string("input_cpg_bg_parquet_file", None, "Path to the parquet file")
flags.mark_flag_as_required("input_me_parquet_file")
flags.mark_flag_as_required("input_cpg_bg_parquet_file")

flags.DEFINE_string("input_me_index_name", None, "Index name in the ME parquet file")
flags.DEFINE_string("input_cpg_bg_index_name", None, "Index name in the CpG BG parquet file")
flags.mark_flag_as_required("input_me_index_name")
flags.mark_flag_as_required("input_cpg_bg_index_name")

flags.DEFINE_string("output_dir", None, "Output directory")
flags.mark_flag_as_required("output_dir")
flags.DEFINE_string("output_file_name", None, "Output directory")
flags.mark_flag_as_required("output_file_name")

flags.DEFINE_boolean("overwrite", False, "Overwrite the output directory if it exists")
flags.DEFINE_boolean("debug", False, "Enable debug mode")
flags.DEFINE_integer("num_workers", 20, "Number of worker processes to use")

FLAGS = flags.FLAGS


def init_process(input_me_index_name, input_cpg_bg_index_name, output_dir):
    init_process.input_me_index_name = input_me_index_name
    init_process.input_cpg_bg_index_name = input_cpg_bg_index_name
    init_process.output_dir = output_dir


def check_me_cpg_bg_parquet_same_index(args):
    me_parquet_file, cpg_bg_parquet_file = args

    me_index_name = getattr(init_process, "input_me_index_name", None)
    cpg_bg_index_name = getattr(init_process, "input_cpg_bg_index_name", None)
    if me_index_name is None or cpg_bg_index_name is None:
        logging.error("Index names are not initialized in init_process")
        return

    me_index_df = pd.read_parquet(me_parquet_file, columns=[me_index_name])
    cpg_bg_index_df = pd.read_parquet(cpg_bg_parquet_file, columns=[cpg_bg_index_name])

    if not me_index_df[me_index_name].equals(cpg_bg_index_df[cpg_bg_index_name]):
        logging.warning(f"Index mismatch: {me_parquet_file.stem}")
        output_dir = getattr(init_process, "output_dir", None)
        with open(output_dir / "index_mismatch.txt", "a") as f:
            f.write(f"{me_parquet_file.stem}\n")
        return None

    return me_parquet_file.stem


def main(argv):
    output_dir = Path(FLAGS.output_dir) / FLAGS.output_file_name

    if output_dir.exists():
        if not FLAGS.overwrite:
            logging.info(f"{output_dir} already exists. Skipping...")
            return
        else:
            logging.info(f"Overwriting {output_dir}")
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if FLAGS.input_me_parquet_file is None or FLAGS.input_cpg_bg_parquet_file is None:
        logging.error("Both input files are required")
        return

    input_me_parquet_file = Path(FLAGS.input_me_parquet_file)
    input_cpg_bg_parquet_file = Path(FLAGS.input_cpg_bg_parquet_file)

    me_parquet_files = sorted(input_me_parquet_file.glob("*.parquet"))
    cpg_bg_parquet_files = sorted(input_cpg_bg_parquet_file.glob("*.parquet"))
    if len(me_parquet_files) == 0:
        logging.error(f"No parquet files found in {input_me_parquet_file}")
    if len(cpg_bg_parquet_files) == 0:
        logging.error(f"No parquet files found in {input_cpg_bg_parquet_file}")
    if len(me_parquet_files) != len(cpg_bg_parquet_files):
        logging.error(
            f"Number of parquet files in {input_me_parquet_file} and {input_cpg_bg_parquet_file} do not match: {len(me_parquet_files)} vs {len(cpg_bg_parquet_files)}"
        )
        return

    # check if the file names are the same
    for me_parquet_file, cpg_bg_parquet_file in zip(me_parquet_files, cpg_bg_parquet_files):
        if me_parquet_file.stem != cpg_bg_parquet_file.stem:
            logging.error(f"File names do not match: {me_parquet_file.stem} vs {cpg_bg_parquet_file.stem}")
            return

    logging.info(f"Reading ME parquet file: {input_me_parquet_file}: {len(me_parquet_files)} files")
    logging.info(f"Reading CpG BG parquet file: {input_cpg_bg_parquet_file}: {len(cpg_bg_parquet_files)} files")

    if FLAGS.debug:
        logging.info("Debug mode enabled")
        init_process(FLAGS.input_me_index_name, FLAGS.input_cpg_bg_index_name, output_dir)
        check_me_cpg_bg_parquet_same_index((me_parquet_files[0], cpg_bg_parquet_files[0]))
        return

    found_mismatch = False
    with multiprocessing.Pool(
        processes=FLAGS.num_workers,
        initializer=init_process,
        initargs=(FLAGS.input_me_index_name, FLAGS.input_cpg_bg_index_name, output_dir),
    ) as pool:
        with tqdm.tqdm(total=len(me_parquet_files)) as pbar:
            for result in pool.imap_unordered(
                check_me_cpg_bg_parquet_same_index, zip(me_parquet_files, cpg_bg_parquet_files)
            ):
                if result is None:
                    found_mismatch = True
                pbar.set_postfix_str(f"Processed {result}")
                pbar.update()
    if found_mismatch:
        logging.warning("Found index mismatches")
    else:
        logging.info("All index matches")
        with open(output_dir / "all_index_match.txt", "w") as f:
            f.write("All index matches")


if __name__ == "__main__":
    app.run(main)
