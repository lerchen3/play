import pandas as pd
import time
import re

# Used to securely store your API key
# from google.colab import userdata # Use this in Colab
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Initialization ---
# Consider specifying the latest stable or desired model version explicitly
# e.g., 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro'
# Check availability and naming conventions.
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # Using flash for potentially faster responses, adjust if needed

# --- Data Loading ---
try:
    # Adjust the path as per your environment (Kaggle, Colab, local)
    df = pd.read_csv('/content/EXTRACTED_PDFS.csv')
    # Or if running locally:
    # df = pd.read_csv('prepped_aops (4).csv')
except FileNotFoundError:
    print("Error: Input CSV file not found. Please check the path.")
    exit()

# --- Processing ---
results = []
# Define the timeout in seconds (15 minutes)
REQUEST_TIMEOUT = 900 # 15 * 60
MAX_RETRIES = 3 # Adjusted for potentially quicker testing, can be increased
# Maximum number of retries

print(f"Starting processing with a timeout of {REQUEST_TIMEOUT} seconds per request and {MAX_RETRIES} retries...")

for idx, row in df.iterrows():
    if(idx%10 == 9 and idx > 0): # Checkpoint every 10 rows, after the 10th row (index 9)
        if results: # Only save if there are results to save
            print(f"Saving checkpoint at row {idx}...")
            results_df_checkpoint = pd.DataFrame(results)
            checkpoint_filename = f'results_checkpoint_row_{idx}.csv'
            results_df_checkpoint.to_csv(checkpoint_filename, index=False)
            try:
                from google.colab import files
                files.download(checkpoint_filename)
                print(f"Checkpoint '{checkpoint_filename}' saved and download initiated (if in Colab).")
            except ImportError:
                print(f"Checkpoint '{checkpoint_filename}' saved locally (google.colab.files not found).")
            except Exception as e:
                print(f"Error during Colab download for checkpoint: {e}")
        else:
            print(f"Checkpoint at row {idx}: No results to save yet.")

    print(f"Processing row {idx}...")
    extracted_content = str(row['latex_content']) # Ensure it's a string
    original_lines = extracted_content.splitlines()
    if not original_lines:
        print(f"  Skipping row {idx} due to empty 'latex_content'.")
        results.append({
            'Original_LaTeX_Content': extracted_content,
            'Extracted_Items': [],
            'LLM_Raw_Response': "Skipped - Empty input",
            'ProcessingTime': 0,
            'FinishReason': "SKIPPED_EMPTY_INPUT"
        })
        continue

    numbered_content = "\\n".join([f"Line {i+1}: {line}" for i, line in enumerate(original_lines)])

    prompt = f"""Original Text with Line Numbers:
{numbered_content}

Instructions:
The text above may contain one or more question, solution, and answer triples.
Your task is to identify the line ranges for EACH such triple.
For each triple, provide the starting and ending line numbers (inclusive) for the question, the solution, and the answer. These line numbers must correspond to the line numbers shown in the "Original Text with Line Numbers" section.

Output ONLY the line numbers in the following strict format for each triple found. Repeat this block for each triple. If no triples are found, output nothing or an empty response.

Question: Line <start_line_number> to <end_line_number>
Solution: Line <start_line_number> to <end_line_number>
Answer: Line <start_line_number>

Example of a single triple:
Question: Line 1 to 5
Solution: Line 6 to 10
Answer: Line 11

- Ensure line numbers are valid and within the range of the provided text (1 to {len(original_lines)}).
- Do NOT output any other text, explanations, apologies, or the content of the lines themselves.
- If multiple triples are present, provide the line ranges for all of them, each in the specified format, one after another.
"""

    processed_row_successfully = False
    row_processing_start_time = time.time() # For total time per row including retries

    for attempt_num in range(MAX_RETRIES): # attempt_num is 0 to MAX_RETRIES-1
        print(f"  Processing row {idx}, Attempt {attempt_num + 1}/{MAX_RETRIES}...")
        try:
            api_call_start_time = time.time()
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=32768, # Output is line numbers, should be small
                temperature=0.1,      # Low temperature for deterministic output
            )
            response = model.generate_content(
                prompt,
                request_options={'timeout': REQUEST_TIMEOUT},
                generation_config=generation_config
            )
            api_call_duration = time.time() - api_call_start_time
            
            llm_response_text = ""
            finish_reason_str = "UNKNOWN_REASON"

            if not response.candidates:
                print(f"  Warning: Row {idx} - No content generated. Finish reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}")
                llm_response_text = f"Error: No content generated. Finish reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}"
                finish_reason_str = response.prompt_feedback.block_reason if response.prompt_feedback else 'NO_CANDIDATES'
            elif response.candidates[0].finish_reason != "STOP":
                print(f"  Warning: Row {idx} - Generation finished unexpectedly. Reason: {response.candidates[0].finish_reason}")
                llm_response_text = response.text
                finish_reason_str = response.candidates[0].finish_reason
            else:
                llm_response_text = response.text
                finish_reason_str = response.candidates[0].finish_reason
                print(f"  Attempt got API response. Finish Reason: {finish_reason_str}")

            # New parsing logic: normalize the response to ignore newlines/extra spaces and group segments into triples
            if isinstance(llm_response_text, str):
                normalized_response = " ".join(llm_response_text.split())
            else:
                normalized_response = llm_response_text

            # Updated pattern to handle Q/S with "to" and Answer without "to"
            pattern = re.compile(r'(Question|Solution): Line (\d+) to (\d+)|(Answer): Line (\d+)', re.IGNORECASE)
            raw_segments = []
            for m in pattern.finditer(normalized_response):
                if m.group(1):  # Matched Question or Solution with a range
                    raw_segments.append((m.group(1), m.group(2), m.group(3))) # (label, start, end)
                elif m.group(4):  # Matched Answer
                    raw_segments.append((m.group(4), m.group(5), "")) # (label, start, end is empty string)
            
            parsed_triples_for_attempt = []
            all_matches_in_response_valid = True # Flag for this attempt's response

            if not llm_response_text.strip() and finish_reason_str == "STOP":
                print(f"  Row {idx}, Attempt {attempt_num + 1}: LLM returned an empty response with STOP. Assuming 0 triples.")
                results.append({
                    'Original_LaTeX_Content': extracted_content,
                    'Extracted_Items': [], # Empty list for no triples
                    'LLM_Raw_Response': llm_response_text,
                    'ProcessingTime_API_Call': api_call_duration,
                    'FinishReason': finish_reason_str
                })
                processed_row_successfully = True
                # break # Break from retry loop for this row - this break needs to be conditional on processed_row_successfully at the end of the try block

            if not raw_segments and llm_response_text.strip(): # If there was text, but nothing was parsed
                print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - Could not parse any segments (Q/S/A) from LLM response after normalization: {normalized_response}")
                all_matches_in_response_valid = False
            elif len(raw_segments) % 3 != 0:
                 print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - Number of parsed segments ({len(raw_segments)}) is not a multiple of 3. LLM response: {normalized_response}")
                 all_matches_in_response_valid = False
            elif not raw_segments and not llm_response_text.strip() and finish_reason_str == "STOP": # Empty response, STOP reason
                # This case is handled above, but kept here for logical flow, effectively a pass.
                # It should have already set processed_row_successfully = True and will break later.
                pass
            elif not raw_segments: # Any other reason for no segments (e.g. empty response but not STOP)
                print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - No segments parsed and response was either non-empty or finish reason was not STOP. LLM response: {normalized_response}")
                all_matches_in_response_valid = False

            if all_matches_in_response_valid and raw_segments: # only proceed if segments were found and count is valid
                for i in range(0, len(raw_segments), 3):
                    group = raw_segments[i:i+3]
                    triple_dict = {}
                    # Ensure the group has one of each label before trying to access them
                    labels_in_group = {label.lower() for (label, _, _) in group}
                    if not {"question", "solution", "answer"}.issubset(labels_in_group):
                        print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - Incomplete triple in group (missing Q/S/A): {group}. Normalized response: {normalized_response}")
                        all_matches_in_response_valid = False
                        break # Stop processing this attempt's segments

                    # Reorder the group to be Q, S, A if needed
                    ordered_group = [None, None, None] # Q, S, A
                    for seg_label, seg_start, seg_end in group:
                        if seg_label.lower() == "question":
                            ordered_group[0] = (seg_label, seg_start, seg_end)
                        elif seg_label.lower() == "solution":
                            ordered_group[1] = (seg_label, seg_start, seg_end)
                        elif seg_label.lower() == "answer":
                            ordered_group[2] = (seg_label, seg_start, seg_end)
                    
                    if not all(ordered_group): # Check if all three were found and placed
                        print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - Failed to reorder group into Q, S, A: {group}. Normalized response: {normalized_response}")
                        all_matches_in_response_valid = False
                        break

                    try:
                        # Now extract from the ordered_group
                        _, q_s_str, q_e_str = ordered_group[0]
                        _, s_s_str, s_e_str = ordered_group[1]
                        _, a_s_str, a_e_str = ordered_group[2]

                        q_s = int(q_s_str)
                        q_e = int(q_e_str) if q_e_str and q_e_str.strip() != "" else q_s
                        s_s = int(s_s_str)
                        s_e = int(s_e_str) if s_e_str and s_e_str.strip() != "" else s_s
                        a_s = int(a_s_str)
                        a_e = int(a_e_str) if a_e_str and a_e_str.strip() != "" else a_s
                        
                        if not (0 < q_s <= q_e <= len(original_lines) and
                                0 < s_s <= s_e <= len(original_lines) and
                                0 < a_s <= a_e <= len(original_lines)):
                            print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - Invalid line numbers: Q({q_s}-{q_e}), S({s_s}-{s_e}), A({a_s}-{a_e}). Max lines: {len(original_lines)}. Normalized: {normalized_response}")
                            all_matches_in_response_valid = False
                            break 
                        
                        question_text = "\\n".join(original_lines[q_s-1:q_e])
                        solution_text = "\\n".join(original_lines[s_s-1:s_e])
                        answer_text   = "\\n".join(original_lines[a_s-1:a_e])
                        parsed_triples_for_attempt.append({
                            'Identified_Question': question_text,
                            'Identified_Solution': solution_text,
                            'Identified_Answer': answer_text,
                            'Line_Numbers': {'Q': (q_s, q_e), 'S': (s_s, s_e), 'A': (a_s, a_e)}
                        })
                    except ValueError as ve:
                        print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - ValueError converting line numbers: {ve}. Group: {ordered_group}. Normalized: {normalized_response}")
                        all_matches_in_response_valid = False
                        break 
                    except TypeError as te: # Handles issues if ordered_group elements are None
                        print(f"  Warning: Row {idx}, Attempt {attempt_num + 1} - TypeError likely due to missing Q/S/A in reordering: {te}. Group: {ordered_group}. Normalized: {normalized_response}")
                        all_matches_in_response_valid = False
                        break

            if all_matches_in_response_valid and parsed_triples_for_attempt: # Successfully parsed some triples
                results.append({
                    'Original_LaTeX_Content': extracted_content,
                    'Extracted_Items': parsed_triples_for_attempt,
                    'LLM_Raw_Response': llm_response_text, # Save original LLM response
                    'ProcessingTime_API_Call': api_call_duration,
                    'FinishReason': finish_reason_str
                })
                print(f"  Row {idx}: Successfully parsed {len(parsed_triples_for_attempt)} item(s) in attempt {attempt_num + 1}.")
                processed_row_successfully = True
            elif all_matches_in_response_valid and not parsed_triples_for_attempt and not llm_response_text.strip() and finish_reason_str == "STOP":
                # This is the case for "LLM returned an empty response with STOP. Assuming 0 triples."
                # This was handled earlier by appending to results and setting processed_row_successfully = True
                # So, no specific action needed here as the flags are already set.
                pass
            else: # Any other situation means this attempt failed.
                  # This includes:
                  # - all_matches_in_response_valid was False at some point (parse error, line number error, etc.)
                  # - all_matches_in_response_valid was True, but parsed_triples_for_attempt is empty AND it wasn't the "empty response means 0 triples" case.
                  #   (e.g., response had segments, but all were invalid, or structure was bad like non-multiple of 3 segments)
                all_matches_in_response_valid = False # Ensure it's marked as failed for the printout below

            if not all_matches_in_response_valid: # If parsing of this attempt's response failed
                 print(f"  Attempt {attempt_num + 1} for row {idx} failed due to parsing issues, problematic LLM output, or invalid data. Normalized Response: '{normalized_response[:200]}...'")
                 # Fall through to retry logic if attempts remain

        except Exception as e: # Covers API call errors, timeouts, etc.
            print(f"  Attempt {attempt_num + 1}/{MAX_RETRIES} for row {idx} failed with exception: {e}")
            # Fall through to retry logic if attempts remain

        if processed_row_successfully:
            break # Exit retry loop (for attempt_num in range(MAX_RETRIES))

        # If not successful and not the last attempt
        if attempt_num < MAX_RETRIES - 1:
            wait_time = 2**(attempt_num + 1) # Exponential backoff: 2s, 4s, 8s...
            print(f"  Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        # If it IS the last attempt and not successful, the loop ends.
    
    if not processed_row_successfully:
        row_total_duration = time.time() - row_processing_start_time
        print(f"  Error: All {MAX_RETRIES} attempts failed for row {idx} after {row_total_duration:.2f} seconds. No result recorded for this row.")
        # Optionally, add a placeholder for failed rows if needed for consistent output row count
        results.append({
            'Original_LaTeX_Content': extracted_content,
            'Extracted_Items': [],
            'LLM_Raw_Response': "All attempts failed",
            'ProcessingTime_API_Call': -1, # Indicate failure
            'FinishReason': "ALL_RETRIES_FAILED"
        })


# --- Save Results ---
if results:
    results_df = pd.DataFrame(results)
    final_csv_filename = 'results_final_line_extraction.csv'
    try:
        results_df.to_csv(final_csv_filename, index=False)
        print(f"Processing complete. Results saved to '{final_csv_filename}'.")
        try:
            from google.colab import files
            files.download(final_csv_filename)
            print(f"'{final_csv_filename}' download initiated (if in Colab).")
        except ImportError:
            print(f"Final results '{final_csv_filename}' saved locally (google.colab.files not found).")
        except Exception as e:
            print(f"Error during Colab download for final results: {e}")

    except IOError as e:
        print(f"Error saving final results to CSV '{final_csv_filename}': {e}")
else:
    print("No results were generated or all rows failed.")