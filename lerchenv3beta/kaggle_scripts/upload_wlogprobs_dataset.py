import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Upload wlogprobs results to a Kaggle dataset")
    parser.add_argument("--results_dir", default="./", help="Directory containing result CSV files")
    parser.add_argument("--dataset_slug", required=True, help="<username>/<dataset-name>")
    parser.add_argument("--title", required=True, help="Title for the Kaggle dataset")
    parser.add_argument("--message", default="Upload results", help="Version message")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    metadata = {
        "title": args.title,
        "id": args.dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }

    meta_path = os.path.join(args.results_dir, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # First try creating the dataset (will fail if it already exists)
    create_cmd = ["kaggle", "datasets", "create", "-p", args.results_dir, "-u"]
    result = subprocess.run(create_cmd)

    if result.returncode != 0:
        # Dataset likely exists, push a new version instead
        version_cmd = ["kaggle", "datasets", "version", "-p", args.results_dir, "-m", args.message]
        subprocess.check_call(version_cmd)


if __name__ == "__main__":
    main()
