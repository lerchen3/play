import os
import pandas as pd
import time
from openai import OpenAI

prompt = """Roleplay as an english teacher. You are an expert at understanding math. I have given you the solution, so no actual math solving should occur. You should not try to solve the question.

Student attempted a math question. My final answer was wrong.
Question: {question}
CORRECT Solution: {solution}
CORRECT Answer: {answer}
My reasoning process: {reasoning}

Your task is the following: identify the specific, initial point of divergence from an optimal or intended logical path; then, provide the single, correct insight that should have occurred at that moment to maintain a correct problem-solving trajectory.

More verbosely, 

1. Identify the Point of Divergence: Scrutinize the provided text to locate the first specific statement or phrase where my reasoning deviates from the required logical sequence. This "cutoff spot" marks the precise end of the user's valid line of thought before a misstep. This point of divergence may occur anywhere within the provided text. The user may have continued their thought process past this point, even arriving at a final, incorrect conclusion with confidence. The task is to pinpoint the initial error in the logical chain, not necessarily the user's final statement. This point of divergence should be copied exactly from the reasoning process, without leaving out text with ellipsis (...). The point of divergence should be the last thing student should have said before saying the correct insight. 

2. Formulate the Correct Insight: Construct a single, concise thought that represents a pivotal insight the user was missing at that exact moment. This thought must serve as a direct and seamless continuation of the user's reasoning up to the cutoff spot. It must not contain any meta-commentary (e.g., "the solution is," "a hint is," "you should have..."). The language must be natural and phrased as if it were the user's own next, spontaneous thought. It should be as small as possible while still being substantial.

Required Output Format:
The response must be structured in the following two parts, without any additional introductory or concluding remarks.
EXACT COPY OF SPOT: [A verbatim, character-for-character reproduction of the identified point of divergence.]
WHAT STUDENT SHOULD HAVE THOUGHT DIRECTLY FOLLOWING THAT: [The formulated correct insight.]"""

MODEL = "o4-mini"
INPUT_CSV = "merged_results.csv"
OUTPUT_CSV = "hints_openai.csv"

if __name__ == "__main__":
    print("Initializing OpenAI client with model:", MODEL)
    client = OpenAI(api_key='')
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    hints = []
    for index, row in df.iterrows():
        question = row["Question"]
        solution = row["Solution"]
        answer = row.get("Answer", "")
        reasoning = row.get("Assistant_Response", row.get("Reasoning", ""))
        formatted_prompt = prompt.format(
            question=question,
            solution=solution,
            answer=answer,
            reasoning=reasoning
        )
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
            )
            hint_text = response.choices[0].message.content.strip()
            hints.append({
                "Question": question,
                "Solution": solution,
                "Answer": answer,
                "Reasoning": reasoning,
                "Hint": hint_text
            })
            out_df = pd.DataFrame(hints)
            out_df.to_csv(OUTPUT_CSV, index=False)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            hints.append({
                "Question": question,
                "Solution": solution,
                "Answer": answer,
                "Reasoning": reasoning,
                "Hint": f"Error: {e}"
            })
        time.sleep(1)
    out_df = pd.DataFrame(hints)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Extracted hints saved to {OUTPUT_CSV}") 