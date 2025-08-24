import argparse
import os
import subprocess
from glob import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge wlogprobs CSVs from multiple Kaggle datasets")
    parser.add_argument("--datasets", nargs='+', required=True, help="Dataset slugs to download and merge")
    parser.add_argument("--output_csv", default="merged_wlogprobs.csv", help="Name for merged CSV")
    args = parser.parse_args()

    temp_dir = "kaggle_downloads"
    os.makedirs(temp_dir, exist_ok=True)

    for slug in args.datasets:
        print(f"Downloading {slug}...")
        subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", temp_dir, "--unzip"])

    csv_files = glob(os.path.join(temp_dir, "*.csv"))
    df_list = [pd.read_csv(f) for f in csv_files]
    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"Merged {len(csv_files)} files into {args.output_csv}")


if __name__ == "__main__":
    main()
