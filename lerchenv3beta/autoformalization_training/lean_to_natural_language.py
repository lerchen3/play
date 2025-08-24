# import genai in above cell.
import argparse
import pandas as pd


def remove_comments(lean_code: str) -> str:
    lines = []
    for line in lean_code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--") or stripped.startswith("/-"):
            continue
        lines.append(line)
    return "\n".join(lines)


def summarize_lean(model, code: str) -> str:
    prompt = (
        "You will be given Lean4 code with comments removed. "
        "Return an equivalent natural language statement of the problem, and if a solution is present, the solution as well."
        "\n\nLean4 Code:\n" + code
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def main(args: argparse.Namespace) -> None:
    genai.configure(api_key=args.google_api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    df = pd.read_csv(args.input_csv)
    results = []

    for _, row in df.iterrows():
        lean = row[args.lean_column]
        clean = remove_comments(lean)
        nl = summarize_lean(model, clean)
        results.append({
            'Original': row[args.original_column],
            'Lean': clean,
            'Gemini_Text': nl,
        })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back translate Lean code using Gemini")
    parser.add_argument("--input_csv", required=True, help="CSV produced by formalize_dataset.py")
    parser.add_argument("--output_csv", default="lean_nl.csv")
    parser.add_argument("--google_api_key", default=None, help="Gemini API key")
    parser.add_argument("--lean_column", default="Lean")
    parser.add_argument("--original_column", default="Question")
    args = parser.parse_args()
    main(args)
