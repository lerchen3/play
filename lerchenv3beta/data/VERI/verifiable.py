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


# --- Data Loading ---
try:
    # Adjust the path as per your environment (Kaggle, Colab, local)
    df = pd.read_csv('/content/prepped_aops (4).csv')
    # Or if running locally:
    # df = pd.read_csv('prepped_aops (4).csv')
except FileNotFoundError:
    print("Error: Input CSV file not found. Please check the path.")
    exit()

# --- Processing ---
results = []
# Define the timeout in seconds (15 minutes)
REQUEST_TIMEOUT = 900 # 15 * 60
MAX_RETRIES = 5 # Maximum number of retries

print(f"Starting processing with a timeout of {REQUEST_TIMEOUT} seconds per request...")

# BATCH NUM 0: 135 to 200 and 535 to 595
# BATCH NUM 1: 335 to 400 and 595 to 655
# BATCH NUM 2: 655 to 780.
for idx, row in df.iterrows():
    # Limit the number of rows processed (as in the original code)
    if(BATCH_NUM == 0):
        if idx < 135 or idx >= 200:
            continue
    elif(BATCH_NUM == 1):
        if idx < 335 or idx >= 400:
            continue
    elif(BATCH_NUM == 2):
        if idx < 655 or idx >= 780:
            continue

    print(f"Processing row {idx}...")
    question = row['Question']
    solution = row['Solution']
    answer = row.get('Answer', 'N/A') # Use .get for safety if 'Answer' might be missing

    prompt = f"""
I'm a student trying to learn how one should approach solving mathematics questions, and I am also trying to practice being completely clear and rigorous in proofs. The solution that I was given to read is currently very unrigorous and lacking in motivation. Please help me on this.

Please first understand both the question and solution, then respond in the format given.

Question: {question}

Solution Given: {solution}

Answer Given: {answer}

VERY VERY IMPORTANT, format your final response as exactly (i.e. nothing before or after):
Motivation And Process 1.
Claim 1.
Proof 1.
Motivation And Process 2.
Claim 2.
Proof 2.
etc.

Guidelines:
1) The "Motivation And Process" explains why you choose a particular claim as your next step. Include calculations and exploration of the problem in the Motivation And Process section. EXECUTE the step itself within the Motivation And Process section. Please do not forget to do this. I know that I already have the given solution, but still write it out so that it flows with the actual motivation. Anything to basically imitate how I should actually be solving the problem.

For example, "Because ..., it is natural to ... . <execution> ... <result>" is a good Motivation And Process.
All steps should be shown; avoid phrases like "A short analysis reveals ..." - instead spell out every single detail.
Whenever you use a trick, formula, theorem, or prior knowledge that you had that is not obvious, please state it explicitly and state the full statement and intuition of it. I'm trying to learn.

2) Restate the executed step formally as "Claim X".

3) Provide a completely rigorous proof for each claim with NO details omitted. Again, avoid phrases like "A short analysis reveals" - instead spell out every single detail; be as verbose as possible!
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
        results_df.to_csv(f'results_with_timeout_{BATCH_NUM}.csv', index=False)
        from google.colab import files
        files.download(f"results_with_timeout_{BATCH_NUM}.csv")
        print(f"Processing complete. Results saved to 'results_with_timeout_{BATCH_NUM}.csv'.")
    except IOError as e:
        print(f"Error saving results to CSV: {e}")
else:
    print("No results were generated.")