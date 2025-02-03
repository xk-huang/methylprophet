import glob
import json
import os

import pandas as pd
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", None, "Directory containing JSON files", required=True)
flags.DEFINE_alias("d", "input_dir")


def extract_split_name(filename):
    # Extract the last part of the filename (log_dict-{split}.json)
    base = os.path.basename(filename)
    # Remove 'log_dict-' prefix and '.json' suffix
    split = base.replace("log_dict-", "").replace(".json", "")
    return split


def process_json_files(input_dir):
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(input_dir, "**", "log_dict-*.json"), recursive=True)
    json_files = sorted(json_files)

    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    # List to store all data
    data = []

    # Process each JSON file
    for json_file in json_files:
        with open(json_file, "r") as f:
            metrics = json.load(f)

        # Create a row with split name and metrics
        row = {"split": extract_split_name(json_file)}
        row.update(metrics)

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort columns to ensure consistent order
    cols = ["split"] + sorted([col for col in df.columns if col != "split"])
    df = df[cols]

    # Save as CSV
    csv_path = os.path.join(input_dir, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)

    # Save as markdown
    markdown_path = os.path.join(input_dir, "metrics_summary.md")
    with open(markdown_path, "w") as f:
        f.write(df.to_markdown(index=False))

    print(f"Files saved:\n{csv_path}\n{markdown_path}")


def main(argv):
    process_json_files(FLAGS.input_dir)


if __name__ == "__main__":
    app.run(main)
