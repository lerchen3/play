!pip install datasets
BATCH_NUM = 0
import pandas as pd
import time
# Used to securely store your API key
# from google.colab import userdata # Use this in Colab
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Initialization ---
# Consider specifying the latest stable or desired model version explicitly
# e.g., 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro'
# Check availability and naming conventions.
model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25') # Using flash for potentially faster responses, adjust if needed

from datasets import load_dataset
dataset = load_dataset("GAIR/LIMO")
df = pd.DataFrame(dataset['train'])

# --- Processing ---
results = []
# Define the timeout in seconds (15 minutes)
REQUEST_TIMEOUT = 900 # 15 * 60
MAX_RETRIES = 5 # Maximum number of retries

print(f"Starting processing with a timeout of {REQUEST_TIMEOUT} seconds per request...")

for idx, row in df.iterrows():
    # Limit the number of rows processed (as in the original code)
    if idx < BATCH_NUM * 100 or idx >= (BATCH_NUM + 1) * 100:
        continue

    print(f"Processing row {idx}...")
    question = row['question']
    solution = row['solution']
    answer = row.get('answer', 'N/A') # Use .get for safety if 'Answer' might be missing

    prompt = f"""
Please modify the provided 'reasoning process' text. Consider this task a refinement, not a foundational rewrite. Approach the changes as if applying a 'diff patch': the output text must still clearly originate from the initial 'reasoning process'. It's crucial that MOST of the original text remains literally copied verbatim or is otherwise unchanged, like a diff patch would do. With this in mind, apply modifications to the original text strictly based on the following rules:
Enhance Logical Flow and Justification:
Ground Ideas: Before proposing any significant step or approach, explicitly state the observation about the current problem state that motivates it. (e.g., "The expression resembles..., so let's try...")
Justify Actions: Clearly explain why each proposed approach or step seems promising at that moment. What goal does it aim to achieve?
Articulate Failure/Dead Ends: If an approach is abandoned, explicitly state the specific reason it proved intractable, led to a dead end, became too complex, or failed to yield useful results. Don’t just say “might be overly complex”; say something actually insightful, like actual justification, something like “this would require an integral with four complicated terms, which is almost surely intractable”.
Articulate Success: If an approach seems like it is working out, say so! Something like, “Wait, this looks like it could work, because … . Awesome!” or something.
Ensure Logical Transitions: The choice of the next step must logically follow from the conclusion (success, partial success, or specific failure) of the previous step. Avoid random jumps or unexplained switches in strategy (like "Alternatively...").
Implement Extreme Verbosity and Contextual Explanation:
Explain Everything: Assume the reader has a basic knowledge of theorems/formulas but minimal competition math problem-solving experience. Explain the reasoning behind every micro-decision and manipulation.
Define in Context: When writing variables, equations, geometric constructions, or specific cases, explain exactly what every single little thing represents, where they came from, and why they are being introduced in this specific context. For example, for equations, explain the origin and meaning of each term or side.
Total Motivational Transparency: Constantly articulate the underlying thought process – the "why" behind every action.
Maintain Authenticity, Chronology, and Style:
Strict Chronological Order: The reasoning must unfold step-by-step as if discovering the solution in real-time. Avoid any hindsight, "spoilers," or revealing insights before they are logically derived. Do not use section headers or phrasing that anticipates discoveries (e.g., avoid "Finding the root x=3" or "Constructing the key point P").
Preserve Plausible Failed Attempts: Do not remove lines of reasoning from the original text that represent plausible attempts, even if they ultimately failed, if possible. Instead, integrate them into the enhanced flow, explaining (per rules 1 & 2) why they were tried and precisely why they were abandoned. It’s okay to rearrange the order in which attempts were tried.
Stylistic Consistency: The added explanations and justifications must blend seamlessly with the original author's writing style and tone. Enhance the existing voice, don't replace it.
Output: Provide only the new reasoning process in your output, and nothing else.

Okay, please adhere to everything!
Here is the question that the reasoning process attempts to solve:
===
{question}
===
The answer given was {answer}.
And here is the “reasoning process” text that I want you to modify.
===
{solution}
===
"""

    attempts = 0
    processed_successfully = False
    loop_start_time = time.time() # Track total time for retries if needed

    while attempts < MAX_RETRIES:
        try:
            start_time = time.time() # Time for this specific attempt
            generation_config = genai.types.GenerationConfig(
            max_output_tokens=32768,
            temperature=0.6,
            )
            response = model.generate_content(
                prompt,
                # Set the client-side timeout here
                request_options={'timeout': REQUEST_TIMEOUT},
                generation_config=generation_config
            )
            end_time = time.time() # Time for this specific attempt
            attempt_duration = end_time - start_time
            print(f"  Attempt {attempts + 1}/{MAX_RETRIES} successful for row {idx} in {attempt_duration:.2f} seconds.")


            # It's good practice to check if the response finished for reasons other than success
            # See Safety Ratings: https://ai.google.dev/gemini-api/docs/safety-settings
            if not response.candidates:
                 print(f"  Warning: Row {idx} - No content generated. Finish reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}")
                 text = f"Error: No content generated. Finish reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}"
                 finish_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'NO_CANDIDATES'
            elif response.candidates[0].finish_reason != 'STOP':
                 print(f"  Warning: Row {idx} - Generation finished unexpectedly. Reason: {response.candidates[0].finish_reason}")
                 # You might still want to save partial results or handle specific finish reasons
                 text = response.text # Or handle based on the reason
                 finish_reason = response.candidates[0].finish_reason
            else:
                text = response.text
                finish_reason = response.candidates[0].finish_reason
            print("prefix btw")
            print(text[:100])
            print("(...)")
            print("=========================")
            results.append({
                'text': text,
                'Question': question,
                'Solution': solution,
                'Answer': answer,
                'ProcessingTime': attempt_duration, # Duration of the successful attempt
                'FinishReason': finish_reason
            })
            processed_successfully = True # Mark as successful
            break # Exit retry loop on success

        except Exception as e:
            # Catch potential errors, including timeouts (which might raise DeadlineExceeded or similar)
            # or other API errors.
            attempts += 1
            print(f"  Attempt {attempts}/{MAX_RETRIES} failed for row {idx}: {e}")
            if attempts < MAX_RETRIES:
                wait_time = 2**attempts # Exponential backoff (2, 4, 8, 16 seconds)
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # This block runs only if the last attempt failed
                total_retry_duration = time.time() - loop_start_time
                print(f"  Error: All {MAX_RETRIES} attempts failed for row {idx} after {total_retry_duration:.2f} seconds. Final error: {e}")
                results.append({
                    'text': f"Error after {MAX_RETRIES} attempts: {e}",
                    'Question': question,
                    'Solution': solution,
                    'Answer': answer,
                    'ProcessingTime': total_retry_duration, # Total time spent trying for this row
                    'FinishReason': 'ERROR_MAX_RETRIES'
                })
                # No need to append anything else, loop will terminate

# --- Save Results ---
if results:
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(f'limoveri_{BATCH_NUM}.csv', index=False)
        from google.colab import files
        files.download(f"limoveri_{BATCH_NUM}.csv")
        print(f"Processing complete. Results saved to 'limoveri_{BATCH_NUM}.csv'.")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")
else:
    print("No results were generated.")