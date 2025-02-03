from pathlib import Path

import pandas as pd
from absl import app, flags, logging


flags.DEFINE_string("input_csv_file", None, "Path to the CSV file")
flags.mark_flag_as_required("input_csv_file")
flags.DEFINE_alias("i", "input_csv_file")
flags.DEFINE_boolean("ipython", False, "Enable IPython at the end of the script")
flags.DEFINE_boolean("verbose", False, "Enable IPython at the end of the script")

flags.DEFINE_integer("nrows", None, "Number of rows to read")
flags.DEFINE_string("sep", ",", "CSV separator")

FLAGS = flags.FLAGS


def main(argv):
    input_csv_file = Path(FLAGS.input_csv_file)
    logging.info(f"Reading CSV file: {input_csv_file}, nrows={FLAGS.nrows}")
    df = pd.read_csv(input_csv_file, nrows=FLAGS.nrows, sep=FLAGS.sep)
    print(df)

    if FLAGS.ipython:
        from IPython import embed

        embed()


if __name__ == "__main__":
    app.run(main)
