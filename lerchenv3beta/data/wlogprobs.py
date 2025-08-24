import argparse
import pandas as pd
from transformers import set_seed
import gc
import warnings
warnings.simplefilter('ignore')
import polars as pl
import torch
import re
import time  # new import for time checks
from collections import Counter, defaultdict
import os

CONTINUATION = False

def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'  # intentionally omitted 'b'
    matches = re.findall(pattern, text)
    if not matches:
        return -1
    content = matches[0]
    if content.isdigit():
        num = int(content)
    else:
        nums = re.findall(r'\d+', content)
        if not nums:
            return -1
        num = int(nums[-1])
    return num  # Return full number without modulo

def batch_message_generate(list_of_messages) -> tuple[list[list[dict]], list[str], list[list[float]]]:
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=(not CONTINUATION)
        )
        for messages in list_of_messages
    ]
    # If continuing from previous assistant responses, trim the eos token from each prompt
    if CONTINUATION:
        eos_token_text = " " # this is correct lmao dont change it pls
        for i, text in enumerate(list_of_texts):
            if not text.endswith(eos_token_text):
                raise ValueError(f"Text at index {i} does not end with eos token")
            list_of_texts[i] = text[:-len(eos_token_text)]
    
    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    
    assistant_responses = []
    responses_logprobs = [] # New list to store logprobs for each response
    for messages, single_request_output in zip(list_of_messages, request_output):
        output_obj = single_request_output.outputs[0]
        response_text = output_obj.text
        messages.append({'role': 'assistant', 'content': response_text})
        assistant_responses.append(response_text)
        
        # Extract logprobs for the generated tokens
        if REQUEST_LOGPROBS:
            current_token_logprobs = []
            if output_obj.logprobs and output_obj.token_ids:
                if len(output_obj.logprobs) == len(output_obj.token_ids):
                    for i, token_id in enumerate(output_obj.token_ids):
                        logprob_entry = output_obj.logprobs[i]
                        if logprob_entry and token_id in logprob_entry:
                            current_token_logprobs.append(logprob_entry[token_id].logprob)
                        else:
                            raise ValueError(f"Logprob for token {token_id} is missing")
                else:
                    raise ValueError(f"Length mismatch: {len(output_obj.logprobs)} != {len(output_obj.token_ids)}")
            elif output_obj.token_ids:
                raise ValueError("Logprobs not available, but tokens are")
        else:
            current_token_logprobs = []
        responses_logprobs.append(current_token_logprobs)
            
    return list_of_messages, assistant_responses, responses_logprobs

def process_questions_initial(input_csv: str):
    df = pd.read_csv(input_csv)
    print(f"Total number of questions: {len(df)}")
    df['Answer'] = pd.to_numeric(df['Answer'], errors='coerce')
    df = df.dropna(subset=['Answer'])
    print(f"Total number of questions with numeric answer: {len(df)}")

    # Initialize questions_data for multi-attempt tracking
    questions_data = []
    for _, row in df.iterrows():
        questions_data.append({
            'Question': row['Question'],
            'Answer': row['Answer'],
            'Solution': row.get('Solution'),
            'attempts': [],
            'attempt_count': 0,
            'correct_count': 0,
        })
    
    pending_indices = list(range(len(questions_data)))
    finished_indices = set()

    # Loop until all questions meet stopping criteria
    while pending_indices:
        for base_chunk_indices in chunker(pending_indices, sub_batch_size):
            # Build prompts for this chunk
            list_of_messages = [
                [{"role": "user", "content": questions_data[i]['Question'] + "Return your final integer answer in a boxed format."}]
                for i in base_chunk_indices
            ]
            # Pad if needed
            if len(base_chunk_indices) < sub_batch_size:
                for p in range(sub_batch_size - len(base_chunk_indices)):
                    list_of_messages.append(list_of_messages[p % len(base_chunk_indices)])
            
            # Generate responses
            _, assistant_responses, assistant_logprobs_list = batch_message_generate(list_of_messages)

            # Record attempts and update status
            for k, response_text in enumerate(assistant_responses):
                data_idx = base_chunk_indices[k % len(base_chunk_indices)]
                qd = questions_data[data_idx]
                qd['attempt_count'] += 1
                pred_ans = extract_boxed_text(response_text)
                is_correct = (pred_ans == qd['Answer'])
                attempt_record = {
                    'Question': qd['Question'],
                    'Answer': qd['Answer'],
                    'Attempt_Number': qd['attempt_count'],
                    'Is_Correct': is_correct,
                    'Assistant_Response': response_text,
                    'Solution': qd['Solution'],
                }
                if REQUEST_LOGPROBS:
                    attempt_record['logprobs'] = assistant_logprobs_list[k]
                qd['attempts'].append(attempt_record)
                if is_correct:
                    qd['correct_count'] += 1
                # Check stopping criteria
                if qd['correct_count'] >= num_correct_wanted or qd['attempt_count'] >= giveup_after_n_attempts:
                    finished_indices.add(data_idx)
        # Update pending_indices
        pending_indices = [i for i in range(len(questions_data)) if i not in finished_indices]

    # Flatten results and save
    results = []
    for qd in questions_data:
        for attempt in qd['attempts']:
            results.append(attempt)
    results_df = pd.DataFrame(results)
    output_csv = "initial_attempts.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Saved initial attempts to {output_csv}")

def process_questions_continuation(input_csv: str):
    df = pd.read_csv(input_csv)
    print(f"Total number of questions: {len(df)}")
    df['Answer'] = pd.to_numeric(df['Answer'], errors='coerce')
    df = df.dropna(subset=['Answer'])
    print(f"Total number of questions with numeric answer: {len(df)}")
    
    batch_start = START_IDX
    batch_end = END_IDX
    current_batch_df = df.iloc[batch_start:batch_end]
    
    questions_data = []
    for index, row in current_batch_df.iterrows():
        questions_data.append({
            'Question': row['Question'],
            'Answer': row['Answer'],
            'initial_assistant_response': row.get('Assistant_Response'),
            'current_assistant_response': row.get('Assistant_Response'),
            'Solution': row.get('Solution'),
            'Hint': row.get('Hint'),
            'attempts': [],
            'attempt_count': 0,
            'correct_count': 0,
        })
    
    start_time = time.time()
    cutoff_time = start_time + TIME_LIMIT
    
    pending_indices = list(range(len(questions_data)))
    finished_questions_indices = set() # Tracks original indices (0 to len(questions_data)-1) that are finished
    
    time_cutoff_reached_flag = False

    while pending_indices:
        if time_cutoff_reached_flag:
            break # Exit if cutoff was hit in a previous chunker iteration

        # pending_indices contains indices of questions not yet finished from the previous round
        for base_chunk_indices in chunker(pending_indices, sub_batch_size):
            if time.time() >= cutoff_time:
                print("Cutoff time reached, stopping further inference calls.")
                time_cutoff_reached_flag = True
                break # Break from this chunker loop

            actual_questions_in_chunk = [questions_data[i]['Question'] for i in base_chunk_indices]
            num_actual_questions = len(actual_questions_in_chunk)

            if num_actual_questions == 0:
                continue

            llm_input_questions = list(actual_questions_in_chunk)
            if num_actual_questions < sub_batch_size:
                print(f"Padding batch: original size {num_actual_questions}, target sub_batch_size {sub_batch_size}. Repeating questions to fill.")
                for i in range(sub_batch_size - num_actual_questions):
                    llm_input_questions.append(actual_questions_in_chunk[i % num_actual_questions])
            
            print(f"Processing LLM batch. Original questions indices in this chunk: {list(base_chunk_indices)}. Total prompts sent to LLM: {len(llm_input_questions)}")
            
            # Build conversation messages with existing assistant response
            list_of_messages = [
                [
                    {'role': 'user', 'content': questions_data[i]['Question']},
                    {'role': 'assistant', 'content': questions_data[i]['initial_assistant_response']}
                ]
                for i in base_chunk_indices
            ]
            # Pad messages if needed
            if num_actual_questions < sub_batch_size:
                print(f"Padding batch: original size {num_actual_questions}, target sub_batch_size {sub_batch_size}. Repeating messages to fill.")
                for pad_i in range(sub_batch_size - num_actual_questions):
                    list_of_messages.append(list_of_messages[pad_i % num_actual_questions])
            # Generate continuations
            _, assistant_responses, assistant_logprobs_list = batch_message_generate(list_of_messages)
            responses = []
            for resp_text, resp_logprobs in zip(assistant_responses, assistant_logprobs_list):
                pred_ans = extract_boxed_text(resp_text)
                responses.append([[resp_text, pred_ans, resp_logprobs]])

            # Process all responses from the LLM call (including those from padded questions)
            for k in range(len(llm_input_questions)):
                # Determine which original question this response corresponds to
                original_question_in_chunk_idx = k % num_actual_questions
                data_idx = base_chunk_indices[original_question_in_chunk_idx] # Index in questions_data

                # Increment attempt count for this specific original question
                questions_data[data_idx]['attempt_count'] += 1
                current_attempt_num_for_q = questions_data[data_idx]['attempt_count']
                
                response_text, predicted_answer, response_logprobs = responses[k][0]
                is_correct = (predicted_answer == questions_data[data_idx]['Answer'])
                
                # The full response is the original + the new continuation.
                full_response_text = (questions_data[data_idx]['current_assistant_response'] + response_text) if questions_data[data_idx]['current_assistant_response'] else response_text

                # Update the response in questions_data for the next attempt.
                questions_data[data_idx]['current_assistant_response'] = full_response_text

                attempt_record = {
                    'Assistant_Response': full_response_text,
                    'predicted_answer': predicted_answer,
                    'Is_Correct': is_correct,
                    'Question': questions_data[data_idx]['Question'],
                    'Answer': questions_data[data_idx]['Answer'],
                    'Solution': questions_data[data_idx]['Solution'],
                    'Hint': questions_data[data_idx]['Hint'],
                }
                if REQUEST_LOGPROBS:
                    attempt_record['logprobs'] = response_logprobs
                questions_data[data_idx]['attempts'].append(attempt_record)
                
                if is_correct:
                    questions_data[data_idx]['correct_count'] += 1
                
                # Check if this question (data_idx) just became finished due to this attempt
                q_data_after_attempt = questions_data[data_idx]
                is_now_finished = (q_data_after_attempt['attempt_count'] >= giveup_after_n_attempts and q_data_after_attempt['correct_count'] == 0) or \
                                  (q_data_after_attempt['correct_count'] >= num_correct_wanted)
                
                if is_now_finished and data_idx not in finished_questions_indices:
                    print(f"Question index {data_idx} (original) finished with {q_data_after_attempt['attempt_count']} attempts and {q_data_after_attempt['correct_count']} correct responses.")
                    finished_questions_indices.add(data_idx)
            # End of loop processing responses for one LLM call
        # End of chunker loop (iterating over current pending_indices)

        if time_cutoff_reached_flag: # If cutoff was hit during the chunker loop above
            break # Exit the main while loop

        # Rebuild pending_indices for the next iteration of the while loop
        # It should contain all original question indices that are not yet finished.
        next_round_pending_indices = []
        for i in range(len(questions_data)): # Iterate through all questions defined by START_IDX, END_IDX
            if i not in finished_questions_indices:
                next_round_pending_indices.append(i)
        
        if not next_round_pending_indices:
            # All questions are finished, or no questions were pending to begin with and cutoff was hit.
            break 
            
        pending_indices = next_round_pending_indices
    
    # Flatten all attempts into a results list for CSV output.
    results = []
    at_least_one_correct = 0
    for q in questions_data:
        if(q['correct_count'] > 0):
            at_least_one_correct += 1
        for attempt in q['attempts']:
            results.append(attempt)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"wlogprobs_{START_IDX}_{END_IDX}.csv", index=False)
    print(f"At least one correct: {at_least_one_correct}")
    print(f"Total: {len(results)}")

def main():
    # Parse command-line arguments for hyperparameters
    parser = argparse.ArgumentParser(description="Run wlogprobs with configurable hyperparameters")
    parser.add_argument("--dataset_name", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--batch_num", type=int, required=True, help="Batch number to process")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of questions per batch")
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum tokens for generation")
    parser.add_argument("--sub_batch_size", type=int, required=True, help="Number of sequences per sub-batch")
    parser.add_argument("--num_correct_wanted", type=int, required=True, help="Number of correct responses wanted per question")
    parser.add_argument("--giveup_after_n_attempts", type=int, required=True, help="Maximum attempts per question")
    parser.add_argument("--random_seed", type=int, default=20090302, help="Random seed for reproducibility")
    parser.add_argument("--triton_ptxas_path", type=str, default="/usr/local/cuda/bin/ptxas", help="Path to ptxas for Triton")
    parser.add_argument("--cuda_visible_devices", type=str, default="0,1,2,3", help="CUDA_VISIBLE_DEVICES environment variable")
    parser.add_argument("--tokenizers_parallelism", type=str, default="false", help="TOKENIZERS_PARALLELISM environment variable")
    parser.add_argument("--llm_model_pth", type=str, required=True, help="Path to LLM model")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Trust remote code for LLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="GPU memory utilization fraction")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature")
    parser.add_argument("--skip_special_tokens", action="store_true", default=True, help="Skip special tokens in generation")
    parser.add_argument("--request_logprobs", type=lambda x: x.lower() in ('true','1','yes'), default=False, help="Whether to request logprobs (True/False)")
    parser.add_argument("--time_limit",    type=float, default=11.5, help="Time limit in hours (default: 11.5)")
    parser.add_argument("--continuation", action="store_true", default=False, help="Whether to continue from existing assistant responses")
    args = parser.parse_args()

    # Override hyperparameters from parsed arguments
    global BATCH_NUM, BATCH_SIZE, START_IDX, END_IDX, MAX_TOKENS, sub_batch_size, num_correct_wanted, giveup_after_n_attempts, TIME_LIMIT, CONTINUATION
    BATCH_NUM = args.batch_num
    BATCH_SIZE = args.batch_size
    START_IDX = BATCH_NUM * BATCH_SIZE
    END_IDX = (BATCH_NUM + 1) * BATCH_SIZE
    MAX_TOKENS = args.max_tokens
    sub_batch_size = args.sub_batch_size
    num_correct_wanted = args.num_correct_wanted
    giveup_after_n_attempts = args.giveup_after_n_attempts
    # Set time limit (in seconds) from hours
    TIME_LIMIT = args.time_limit * 3600
    CONTINUATION = args.continuation

    # Set random seed
    set_seed(args.random_seed)

    # Configure environment variables
    os.environ["TRITON_PTXAS_PATH"] = args.triton_ptxas_path
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    from vllm import LLM, SamplingParams
    # Assign model and dataset paths
    global llm_model_pth, csv_path, llm, tokenizer, REQUEST_LOGPROBS, sampling_params
    llm_model_pth = args.llm_model_pth
    csv_path = args.dataset_name

    # Initialize the LLM with parsed parameters
    llm = LLM(
        llm_model_pth,
        max_model_len=MAX_TOKENS,
        max_num_seqs=sub_batch_size,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    tokenizer = llm.get_tokenizer()

    REQUEST_LOGPROBS = args.request_logprobs

    # Conditionally set sampling_params based on REQUEST_LOGPROBS
    if REQUEST_LOGPROBS:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            skip_special_tokens=args.skip_special_tokens,
            max_tokens=MAX_TOKENS,
            logprobs=1
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            skip_special_tokens=args.skip_special_tokens,
            max_tokens=MAX_TOKENS
        )

    # Call the appropriate processing function based on continuation flag
    if CONTINUATION:
        process_questions_continuation(csv_path)
    else:
        process_questions_initial(csv_path)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()