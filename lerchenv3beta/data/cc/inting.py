import os
import re
import time
import pandas as pd

# --- Prompt Template ---
prompt_template = r"""DO NOT attempt to solve the following question! Your task is to:
1. Suggest an addition to the question
2. Suggest an addition to the solution
3. Modify the answer to be a nonnegative integer between 0 and 999.

FOR EXAMPLE (this may not apply to the specific question provided), if the original answer was 1728, add "What is this number modulo 1000?" to the question, add "Hence, the final answer is 1728 mod(1000) = 728" to the solution, and return the answer as 728.
Other common examples include:
1) Sum of numerator and denominator
2) Square of value
3) "If it can be expressed as a+sqrt(b) where b is squarefree, find a+b"

Please ensure the answer is neither too easily guessable nor overly complex. The modification should be a simple answer extraction that maintains the question's integrity.
If the answer is already an integer between 0 and 999, respond with "None" for the question and solution additions, and provide only the nonnegative integer as the new answer.

Original Question:
{Question}
Original Solution:
{Solution}
Original Answer:
{Answer}

Return your output exactly as follows:
Addition to question: ...
Addition to solution: ...
New Answer: (integer between 0 and 999, and only this integer)"""
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Settings
REQUEST_TIMEOUT = 900  # seconds
MAX_RETRIES = 3

# File paths
INPUT_CSV = "/content/results.csv"    # adjust path if needed
OUTPUT_CSV = "output.csv"  # adjust path if needed

# Load input data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Input CSV '{INPUT_CSV}' not found.")
    exit(1)

# Ensure required columns exist
required = {"Question", "Solution", "Answer"}
if not required.issubset(df.columns):
    print(f"Error: Input CSV must contain columns: {required}")
    exit(1)

results = []
print(f"Processing {len(df)} rows with timeout {REQUEST_TIMEOUT}s and up to {MAX_RETRIES} retries each.")

for idx, row in df.iterrows():
    question = str(row["Question"])
    solution = str(row["Solution"])
    answer = str(row["Answer"])
    prompt = prompt_template.format(Question=question, Solution=solution, Answer=answer)

    processed = False
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Row {idx}: Attempt {attempt}/{MAX_RETRIES}")
        try:
            start = time.time()
            gen_cfg = genai.types.GenerationConfig(
                max_output_tokens=32768,
                temperature=0.1,
            )
            response = model.generate_content(
                prompt,
                request_options={"timeout": REQUEST_TIMEOUT},
                generation_config=gen_cfg
            )
            duration = time.time() - start

            if not response.candidates:
                raise RuntimeError("No candidates returned")
            if response.candidates[0].finish_reason != "STOP":
                print(f"  Warning: finish_reason = {response.candidates[0].finish_reason}")

            llm_out = response.text

            # Parse the three fields
            m = re.search(
                r"Addition to question:\s*(.*?)\s*Addition to solution:\s*(.*?)\s*New Answer:\s*(\d+)",
                llm_out,
                re.DOTALL | re.IGNORECASE
            )
            if not m:
                raise ValueError(f"Could not parse output:\n{llm_out}")

            add_q = m.group(1).strip()
            add_s = m.group(2).strip()
            new_ans = m.group(3).strip()
            if(len(add_q) + len(add_s) + len(new_ans) > 4000):
                print("Addition to question and solution is too long.")
                continue
            # Construct new question/solution
            new_question = question + " " + add_q if add_q.lower() != "none" else question
            new_solution = solution + " " + add_s if add_s.lower() != "none" else solution

            results.append({
                "New_Question": new_question,
                "New_Solution": new_solution,
                "New_Answer": new_ans
            })
            print(f"  Row {idx} succeeded.")
            processed = True
            break

        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                backoff = 2 ** attempt
                print(f"  Retrying in {backoff}s...")
                time.sleep(backoff)
    if not processed:
        print(f"Row {idx} failed after {MAX_RETRIES} attempts.")
        

# Save to output CSV
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Results written to '{OUTPUT_CSV}'.")
from google.colab import files

files.download(OUTPUT_CSV)