import os
from pathlib import Path
from pprint import pformat

from absl import app, flags, logging


flags.DEFINE_string("input_dir", None, "Path to the directory")
flags.mark_flag_as_required("input_dir")
flags.DEFINE_alias("d", "input_dir")
flags.DEFINE_bool("dry_run", False, "Dry run")


FLAGS = flags.FLAGS


def main(_):
    input_dir = Path(FLAGS.input_dir)

    logging.info(f"Removing empty directories in: {input_dir}")

    # Recursively find empty directories
    empty_dirs = [
        dirpath for dirpath, dirnames, filenames in os.walk(input_dir, topdown=False) if not dirnames and not filenames
    ]

    logging.info(f"Found {len(empty_dirs)} empty directories: {pformat(empty_dirs)}")

    if FLAGS.dry_run:
        logging.info("Dry run, exiting")
        return

    for empty_dir in empty_dirs:
        logging.info(f"Removing empty directory: {empty_dir}")
        os.rmdir(empty_dir)


if __name__ == "__main__":
    app.run(main)
