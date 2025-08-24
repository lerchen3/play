import argparse
import pandas as pd
from vllm import LLM, SamplingParams


def build_prompt(question: str, solution: str | None, include_solution: bool) -> str:
    if include_solution and solution is not None:
        return (
            "Formalize the following math problem and its solution in Lean4.\n\n"
            f"Question: {question}\n\nSolution: {solution}\n\nLean4 proof:"
        )
    return (
        "Formalize the following math problem in Lean4.\n\n"
        f"Question: {question}\n\nLean4 proof:"
    )


def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv)

    llm = LLM(
        args.model_path,
        max_model_len=args.max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True)

    results = []
    for start in range(0, len(df), args.batch_size):
        batch = df.iloc[start:start + args.batch_size]
        list_of_messages = []
        for _, row in batch.iterrows():
            question = row['Question']
            solution = row.get('Solution', '')
            prompt = build_prompt(question, solution, args.include_solution)
            list_of_messages.append([{'role': 'user', 'content': prompt}])

        list_of_texts = [
            tokenizer.apply_chat_template(conversation=m, tokenize=False, add_generation_prompt=True)
            for m in list_of_messages
        ]
        outputs = llm.generate(prompts=list_of_texts, sampling_params=sampling_params)
        for row, out in zip(batch.itertuples(index=False), outputs):
            results.append({
                'Question': row.Question,
                'Solution': getattr(row, 'Solution', ''),
                'Lean': out.outputs[0].text.strip(),
            })

    pd.DataFrame(results).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formalize questions in Lean4 using a local model")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with Question and Solution columns")
    parser.add_argument("--output_csv", default="formalized.csv", help="Where to store Lean4 output")
    parser.add_argument("--model_path", default="/kaggle/input/model", help="Path to local model")
    parser.add_argument("--include_solution", action="store_true", help="Include the solution text in the prompt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()
    main(args)
