# import genai in above cell.
import argparse
import pandas as pd


def is_equivalent(model, original: str, generated: str) -> bool:
    prompt = (
        "Determine if the following two mathematical statements are logically and mathematically identical. "
        "Respond with simply 'Yes' or 'No'.\n\n"
        f"STATEMENT 1:\n{original}\n\nSTATEMENT 2:\n{generated}"
    )
    response = model.generate_content(prompt)
    answer = response.text.strip().lower()
    return answer.startswith("yes")


def main(args: argparse.Namespace) -> None:
    genai.configure(api_key=args.google_api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    df = pd.read_csv(args.input_csv)
    sft_rows = []

    for _, row in df.iterrows():
        if is_equivalent(model, row['Original'], row['Gemini_Text']):
            prompt = row['Original']
            response_text = row['Lean']
            sft_rows.append({'prompt': prompt, 'response': response_text})

    pd.DataFrame(sft_rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare natural language statements and build SFT dataset")
    parser.add_argument("--input_csv", required=True, help="CSV from lean_to_natural_language.py")
    parser.add_argument("--output_csv", default="sft_dataset.csv")
    parser.add_argument("--google_api_key", default=None)
    args = parser.parse_args()
    main(args)
