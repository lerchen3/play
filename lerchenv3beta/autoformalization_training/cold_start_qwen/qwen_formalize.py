import argparse
import pandas as pd
import os
from openai import OpenAI

MODEL = "accounts/fireworks/models/qwen3-235b-a22b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def build_prompt(question: str, solution: str | None, include_solution: bool) -> str:
    if include_solution and solution is not None:
        return (
            "Formalize the following math question and its solution into Lean4.\n\n"
            f"Question: {question}\n\nSolution: {solution}\n\nLean4 proof:"
        )
    return (
        "Formalize the following math question in Lean4.\n\n"
        f"Question: {question}\n\nLean4 proof:"
    )


def main(args: argparse.Namespace) -> None:
    client = OpenAI(api_key=os.environ.get("FIREWORKS_API_KEY"), base_url=FIREWORKS_BASE_URL)
    df = pd.read_csv(args.input_csv)
    results = []
    for _, row in df.iterrows():
        prompt = build_prompt(row['Question'], row.get('Solution', ''), args.include_solution)
        chat = [
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(model=MODEL, messages=chat)
        results.append({
            'Question': row['Question'],
            'Solution': row.get('Solution', ''),
            'Lean': response.choices[0].message.content.strip(),
        })
    pd.DataFrame(results).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formalize dataset using Qwen via Fireworks")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", default="qwen_formalized.csv")
    parser.add_argument("--include_solution", action="store_true")
    args = parser.parse_args()
    main(args)
