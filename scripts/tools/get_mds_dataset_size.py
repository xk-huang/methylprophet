from pathlib import Path

from absl import app, flags, logging
from streaming import StreamingDataset


flags.DEFINE_string("local", None, "Path to the dataset")
flags.mark_flag_as_required("local")
flags.DEFINE_alias("d", "local")

flags.DEFINE_string("remote", None, "Path to the dataset")

flags.DEFINE_string("output_dir", None, "Path to the output directory")
flags.DEFINE_bool("overwrite", False, "Overwrite the output file")


FLAGS = flags.FLAGS


def main(_):
    logging.info(f"Local: {FLAGS.local}")
    logging.info(f"Remote: {FLAGS.remote}")
    dataset = StreamingDataset(local=FLAGS.local, remote=FLAGS.remote)
    logging.info(f"dataset.size: {dataset.size:,}; len(dataset): {len(dataset):,}")

    if FLAGS.output_dir:
        output_dir = Path(FLAGS.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "mds_dataset_size.tsv"
        logging.info(f"Output path: {output_path}")

        has_header = False
        if output_path.exists():
            if FLAGS.overwrite:
                output_path.unlink()
                logging.info(f"Removed existing file: {output_path}")
            else:
                has_header = True

        with open(output_path, "a") as f:
            if not has_header:
                f.write("local\tremote\tdataset.size\tlen(dataset)\n")
            f.write(f"{FLAGS.local}\t{FLAGS.remote}\t{dataset.size:,}\t{len(dataset):,}\n")


if __name__ == "__main__":
    app.run(main)
