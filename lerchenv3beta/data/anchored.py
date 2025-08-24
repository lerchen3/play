import os
import re
import time
import pandas as pd
from openai import OpenAI

MODEL = "o3-mini"
INPUT_CSV = "/kaggle/input/semihard-qs/semihard_qs.csv"
OUTPUT_CSV = "anchored_qs.csv"

def parse_generated_text(text):
    """
    Parse the generated text to extract the Modified Question and Answer using defined delimiters.
    Expected format:
        <<BeginQuestion>>
        (Modified Question)
        <<EndQuestion>>
        <<BeginAnswer>>
        (Answer)
        <<EndAnswer>>
    """
    question_match = re.search(r"<<BeginQuestion>>(.*?)<<EndQuestion>>", text, re.DOTALL)
    answer_match = re.search(r"<<BeginAnswer>>(.*?)<<EndAnswer>>", text, re.DOTALL)
    modified_question = question_match.group(1).strip() if question_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    return modified_question, answer

if __name__ == "__main__":
    print("Initializing OpenAI client with model:", MODEL)
    # Initialize the client with your API key
    client = OpenAI(api_key='')

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    extracted_qa_pairs = []
    
    for index, row in df.iterrows():
        original_question = row["Question"]
        original_solution = row["Solution"]
        
        # Construct the prompt with clear instructions and expected delimiters.
        prompt = (
            f"Original Question: {original_question}\n"
            f"Original Solution: {original_solution}\n\n"
            "Your task is to generate a modified question and a corresponding answer, using the format below:\n\n"
            "<<BeginQuestion>>\n"
            "Modified Question Here\n"
            "<<EndQuestion>>\n"
            "<<BeginAnswer>>\n"
            "Answer Here\n"
            "<<EndAnswer>>\n\n"
            "Guidelines:\n"
            "1) The Modified Question should be of similar or slightly higher difficulty compared to the original.\n"
            "2) It must be fundamentally different from the original question—not just superficial changes like changing numbers or adding a superficial layer to the question. Think, 'What if we change this to this?', or 'Let's think more about this particular property, and generalize or make a variant of it.' Stuff like that.\n"
            "3) The answer should be an integer (as in the original solution), though not necessarily the same integer as the original solution (different is actually better).\n"
            "4) Make sure that you put only the modified question and answer in your response after thinking, nothing else."
        )
        
        # Define the conversation messages for the API call.
        conversation = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that helps transform math problems. "
                    "Given an original question and its solution, you generate a new modified question "
                    "and corresponding answer in the requested format. Ensure that the modified question is "
                    "fundamentally different yet of similar difficulty, and the answer is an integer."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=conversation,
            )
            generated_text = response.choices[0].message.content
            modified_question, answer = parse_generated_text(generated_text)
            if modified_question is None or answer is None:
                print(f"Warning: Could not parse output for row {index}. Raw output:\n{generated_text}")
            # if answer not int, try to convert.
            try:
                answer = int(answer)
            except:
                print(f"Warning: Answer {answer} is not an integer for row {index}. Raw output:\n{generated_text}")
            else:
                extracted_qa_pairs.append({
                    "Modified Question": modified_question,
                    "Answer": answer
                })
        except Exception as e:
            print(f"Error processing row {index}: {e}")
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
    
    # Save the extracted modified question-answer pairs to the output CSV.
    output_df = pd.DataFrame(extracted_qa_pairs)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Extracted question-answer pairs saved to {OUTPUT_CSV}")