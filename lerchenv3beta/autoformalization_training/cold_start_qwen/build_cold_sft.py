import argparse
import pandas as pd


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv)
    df['prompt'] = df['Question']
    df['response'] = df['Lean']
    df[['prompt', 'response']].to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen formalizations to SFT dataset")
    parser.add_argument("--input_csv", required=True, help="Output from qwen_formalize.py")
    parser.add_argument("--output_csv", default="qwen_sft.csv")
    args = parser.parse_args()
    main(args)
