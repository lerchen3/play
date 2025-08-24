from google.generativeai.types import HarmCategory, HarmBlockThreshold
prompt = """You are an expert mathematics tutor with a deep understanding of logical reasoning. You will be given a math problem, its solution, and a student's flawed reasoning process.

Your task is to analyze the student's reasoning and identify the "point of leverage." This is the spot in their reasoning where a minimal hint would be most effective. This isn't necessarily a mistake, but rather a moment of missed opportunity—a point where the student uncovers a key insight but fails to act on it, or is about to commit to a flawed strategy. You will then provide the minimal, most natural next thought that should have followed to capitalize on that moment.

### Core Principles:

* **Non-Linearity:** Critically, you must recognize that discovering the components of a solution is not always linear. The student may have found key ideas in a "rearranged order" compared to the provided solution, and this is perfectly fine. Your goal is to identify their most potent, under-utilized idea, even if their reasoning up to that point follows an unconventional sequence.
* **Minimal Nudge:** Your hint must be the smallest possible nudge. It should not explicitly state the next major step or introduce a complex new mechanism from the solution. Instead, it should prompt the student to recognize the significance of an "ingredient" they have already found, encouraging them to connect it back to the overall goal of the problem.
* **Natural Flow:** The insight you provide must be phrased as a natural, spontaneous thought or question that flows seamlessly from the student's last stated thought. It should feel like the student's own internal monologue, not an external instruction.
* **Solution-Oriented:** While subtle, the hint must ultimately guide the student towards the same core mathematical concepts used in the provided solution. It is your expertise that is needed to identify what is in the given solution that is not in the student's reasoning, and provide a natural *why* it makes sense to go towards the particular part of the solution - you are the expert.
* **Exact copy:** A verbatim, character-for-character reproduction of the identified point of leverage from the student's reasoning should be produced, so I can ctrl-f it and find it in the reasoning. Double-check that it indeed is a perfect reproduction (and make it rather short - I only need to know where to find the cutoff spot; for example, if the sentence is "Hence, ..., so the shape is indeed a square", simply saying "so the shape is indeed a square" is sufficient)

### Your Task:

1.  **Identify the Point of Leverage:** Scrutinize the student's reasoning to locate the phrase that represents the greatest point of leverage. This "cutoff spot" should be quoted verbatim. It is the thought that contains the most untapped potential before the student gets stuck, makes an error, or moves on prematurely.
2.  **Formulate the Correct Insight:** Construct a single, concise thought or question. This thought should:
    * Be phrased as if it were the student's own natural, spontaneous next thought.
    * Contain no meta-commentary (e.g., "you should have," "a key hint is").
    * Act as a minimal prompt, encouraging the student to connect their last stated thought with the problem's objective.
    * Leverage the "ingredient" the student has just found, pushing them to see its importance rather than giving them something new.

### Required Output Format:

Your entire response must consist *only* of the following two parts, with no additional introduction, explanation, or conclusion. Make sure you put exactly these **Markers:**, as well as copy the cutoff spot verbatim. 

**EXACT COPY OF SPOT:** [A verbatim, character-for-character reproduction of the identified point of leverage from the student's reasoning.]

**WHAT STUDENT SHOULD HAVE THOUGHT DIRECTLY FOLLOWING THAT:** [The single, concise, and seamless insight.]

Question: {question}

Solution: {solution}
Final Answer: {answer}

Student Reasoning: {reasoning}"""

import os
import pandas as pd
import time

BATCH_NUM = 0
REQUEST_TIMEOUT = 900  # 15 minutes per request
MAX_RETRIES = 1

# Configure the Gemini API client
genai.configure(api_key='')
model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05')

# Paths to input and output CSV files
INPUT_CSV = "/content/merged_results.csv"
OUTPUT_CSV = "hints_gemini_{BATCH_NUM}.csv"

# Load the input data
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Prepare for processing
results = []

print(f"Starting processing with a timeout of {REQUEST_TIMEOUT} seconds per request...")

for idx, row in df.iterrows():
    # Process in batches (100 rows per batch)
    if idx < BATCH_NUM * 100 or idx >= (BATCH_NUM + 1) * 100:
        continue

    print(f"Processing row {idx}...")
    question = row["Question"]
    solution = row["Solution"]
    answer = row["Answer"]
    reasoning = row["Assistant_Response"]
    formatted_prompt = prompt.format(
        question=question,
        solution=solution,
        answer=answer,
        reasoning=reasoning
    )

    attempts = 0
    loop_start_time = time.time()

    while attempts < MAX_RETRIES:
        try:
            start_time = time.time()
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=65536,
                temperature=0.6,
            )
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }

            response = model.generate_content(
                formatted_prompt,
                request_options={'timeout': REQUEST_TIMEOUT},
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            end_time = time.time()
            duration = end_time - start_time
            print(f"  Attempt {attempts+1}/{MAX_RETRIES} successful in {duration:.2f} seconds.")

            # Handle generation output
            if not response.candidates:
                text = f"Error: No content generated. Finish reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}"
                finish_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'NO_CANDIDATES'
            elif response.candidates[0].finish_reason != 'STOP':
                text = response.text
                finish_reason = response.candidates[0].finish_reason
            else:
                text = response.text
                finish_reason = response.candidates[0].finish_reason

            # Validate EXACT COPY OF SPOT occurs exactly once in the original Assistant_Response
            marker1 = "**EXACT COPY OF SPOT:**"
            marker2 = "**WHAT STUDENT SHOULD HAVE THOUGHT DIRECTLY FOLLOWING THAT:**"
            if marker1 not in text or marker2 not in text:
                attempts += 1
                wait = 2 ** attempts
                print(f"  Validation failed: missing markers, retrying in {wait} seconds.")
                time.sleep(wait)
                continue

            one = text.index(marker1) + len(marker1)
            two = text.index(marker2)
            three = text.index(marker2) + len(marker2)
            spot = text[one:two].strip()
            hint = text[three:].strip()

            # Use the original row['Assistant_Response'] for validation
            assistant_response = row["Assistant_Response"]
            occurrences = assistant_response.count(spot)
            if occurrences != 1:
                attempts += 1
                wait = 2 ** attempts
                print(f"  Validation failed: spot '{spot}' occurs {occurrences} times in Assistant_Response, retrying in {wait} seconds.")
                time.sleep(wait)
                continue

            # Create new Assistant_Response by cutting off at spot and adding hint
            cutoff_index = assistant_response.find(spot) + len(spot)
            new_assistant_response = assistant_response[:cutoff_index] + " " + hint

            results.append({
                "Question": question,
                "Solution": solution,
                "Answer": answer,
                "Hints": [hint],
                "Assistant_Response": new_assistant_response
            })
            break

        except Exception as e:
            attempts += 1
            print(f"  Attempt {attempts}/{MAX_RETRIES} failed: {e}")
            if attempts < MAX_RETRIES:
                wait = 2 ** attempts
                print(f"  Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                total_duration = time.time() - loop_start_time
                print(f"  Error: All {MAX_RETRIES} attempts failed after {total_duration:.2f} seconds.")
                break

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    from google.colab import files
    files.download(OUTPUT_CSV)
    print(f"Hints saved to {OUTPUT_CSV}")
else:
    print("No hints were generated.") 