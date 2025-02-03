import json
from collections import OrderedDict

from absl import app, flags, logging


FLAGS = flags.FLAGS
flags.DEFINE_string("input", None, "Input JSON file path")
flags.mark_flag_as_required("input")
flags.DEFINE_alias("i", "input")

flags.DEFINE_string("output", None, "Output markdown file path")
flags.DEFINE_alias("o", "output")

flags.DEFINE_string("length_column_name", None, "Name of the column with the length of the dataset")
flags.DEFINE_float("infer_speed_its", None, "Inference speed in iterations per second")
flags.DEFINE_float("batch_size", None, "Number of samples per batch")


def json_to_markdown(json_data, length_column_name=None, infer_speed_its=None, batch_size=None):
    # Get all unique keys while maintaining order from first occurrence
    headers = []
    for item in json_data.values():
        for key in item.keys():
            if key not in headers:
                headers.append(key)

    log_infer_time = False
    extra_headers = []
    if infer_speed_its is not None and batch_size is not None and length_column_name is not None:
        extra_headers.append("Inference speed (its/s)")
        extra_headers.append("Batch size")
        extra_headers.append("Inference Time (s)")
        extra_headers.append("Inference Time (h)")
        log_infer_time = True
        logging.info("Adding inference speed and batch size columns")
        if length_column_name not in headers:
            raise ValueError(f"Length column name {length_column_name} not found in headers")
    else:
        logging.info("Skipping inference speed and batch size columns")

    # Create the markdown header
    markdown = "| Dataset | " + " | ".join(headers + extra_headers) + " |\n"
    # Add the separator line
    markdown += "|" + "|".join(["-" * 10] * (len(headers + extra_headers) + 1)) + "|\n"

    # Add each row of data
    for dataset, values in json_data.items():
        row = [dataset]
        for header in headers:
            # Format numbers with commas for readability
            value = values.get(header, "")
            if isinstance(value, (int, float)):
                value = "{:,}".format(value)
            row.append(str(value))
        if log_infer_time:
            infer_time = values.get(length_column_name) / infer_speed_its / batch_size
            row.append(f"{infer_speed_its:.2f}")
            row.append(f"{batch_size}")
            row.append(f"{infer_time:.2f}")
            row.append(f"{infer_time / 3600:.2f}")
        markdown += "| " + " | ".join(row) + " |\n"

    return markdown


def main(argv):
    # Read JSON file with OrderedDict to preserve order
    with open(FLAGS.input, "r") as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    # Convert to markdown
    markdown_table = json_to_markdown(
        json_data=json_data,
        length_column_name=FLAGS.length_column_name,
        infer_speed_its=FLAGS.infer_speed_its,
        batch_size=FLAGS.batch_size,
    )

    # Either print to stdout or write to file
    if FLAGS.output:
        with open(FLAGS.output, "w") as f:
            f.write(markdown_table)
        print(f"Markdown table written to {FLAGS.output}")
    else:
        print(markdown_table)


if __name__ == "__main__":
    app.run(main)
