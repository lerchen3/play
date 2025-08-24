import pandas as pd
import time
# Used to securely store your API key
# from google.colab import userdata # Use this in Colab

# --- Load Data ---
print("Loading data from input.csv...")
df = pd.read_csv("/content/merged_outputs.csv")
print(f"Loaded {len(df)} rows")

# --- Model Initialization ---
model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05')

# --- Processing ---
results = []
REQUEST_TIMEOUT = 900  # 15 minutes

print("Starting processing...")

for idx, row in df.iterrows():
    if(idx >= 100):
        break
    print(f"Processing row {idx}/{len(df)-1}...")
    
    try:
        question = row['Question']
        solution = row['Solution']
        
        # --- First API Call: Generate Initial Reasoning Process ---
        initial_prompt = f"""Simulate what a 180 IQ mathematical genius would say when encountering this competition problem for the first time. Show their internal monologue as they work through it.

Question: {question}

Solution: {solution}

Show exactly what this genius would say as they think: their immediate thoughts, initial hunches, exploration of 2-6 different approaches they would consider (including the ones they abandon and why), and their step-by-step reasoning. Write what their actual stream of consciousness would sound like - not as an explanation about reasoning, but as the genuine thoughts this brilliant mind would voice.

The reasoning must flow naturally and sequentially like real thinking [avoid artificial structures like "Section 1" or telegraphing what will happen].

Format your response exactly as:

Reasoning Process: Okay, I have this math competition question to solve. I have never seen anything like it before, but I'm confident that using my amazing ability to come up with innovative ideas, I will end up creating a fully correct solution. The question is 
[Continue with what the genius would actually say as their internal monologue, including discarded approaches and why they were abandoned]"""

        print(f"  Making first API call (initial reasoning)...")
        start_time = time.time()
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=32768,
            temperature=0.6,
        )
        
        response1 = model.generate_content(
            initial_prompt,
            request_options={'timeout': REQUEST_TIMEOUT},
            generation_config=generation_config
        )
        
        # Check if first response is valid
        if not response1.candidates or response1.candidates[0].finish_reason != 'STOP':
            print(f"  Skipping row {idx}: First API call failed")
            continue
            
        initial_reasoning = response1.text
        if initial_reasoning.startswith("Reasoning Process: "):
            initial_reasoning = initial_reasoning[len("Reasoning Process: "):]
        first_call_time = time.time() - start_time
        print(f"  First call completed in {first_call_time:.2f} seconds")
        
        # --- Second API Call: Generate Enhanced Reasoning Process ---
        enhanced_prompt = f"""Continue simulating what the same 180 IQ mathematical genius would say. They previously worked through this problem, but now show what their thought process would sound like in much greater detail.

Original Question: {question}

Initial Reasoning Process: {initial_reasoning}

Show what this genius would say as their mind works in extreme detail: every small calculation, every moment of doubt, every "aha!" moment, every dead end they would explore fully. Write what their complete internal monologue would sound like - showing what a genius would actually say while thinking through every step.

Instructions for what the genius would say:
- What they would say during every calculation and algebraic manipulation
- Their frustration and excitement as they would voice while working through ideas  
- When they try failed approaches, what they would say during the full effort of exploring them before realizing they don't work
- How they would voice their mental shortcuts and pattern recognition
- What their authentic and human thought process would sound like, just incredibly sharp

Format your response exactly as:

Enhanced Reasoning Process: Okay, I have this math competition question to solve. I have never seen anything like it before, so I will have to be innovative and come up with ideas.

[Continue with what the genius would actually say as their complete, detailed internal monologue]"""

        print(f"  Making second API call (enhanced reasoning)...")
        start_time = time.time()
        
        response2 = model.generate_content(
            enhanced_prompt,
            request_options={'timeout': REQUEST_TIMEOUT},
            generation_config=generation_config
        )
        
        # Check if second response is valid
        if not response2.candidates or response2.candidates[0].finish_reason != 'STOP':
            print(f"  Skipping row {idx}: Second API call failed")
            continue
            
        enhanced_reasoning = response2.text
        if enhanced_reasoning.startswith("Enhanced Reasoning Process: "):
            enhanced_reasoning = enhanced_reasoning[len("Enhanced Reasoning Process: "):]
        second_call_time = time.time() - start_time
        total_time = first_call_time + second_call_time
        
        print(f"  Second call completed in {second_call_time:.2f} seconds")
        print(f"  Total processing time: {total_time:.2f} seconds")
        
        # Save only the enhanced reasoning process
        results.append({
            'enhanced_reasoning': enhanced_reasoning,
            'question': question,
            'solution': solution,
            'answer': row['Answer'],
            'processing_time': total_time,
            'row_index': idx
        })
        
        print(f"  Successfully processed row {idx}")
        print(f"  Preview: {enhanced_reasoning[:100]}...")
        print("=" * 50)
        
    except Exception as e:
        print(f"  Skipping row {idx} due to error: {e}")
        continue

# --- Save Results ---
if results:
    results_df = pd.DataFrame(results)
    output_filename = 'enhanced_reasoning_results.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"Processing complete. {len(results)} rows processed successfully.")
    print(f"Results saved to '{output_filename}'.")
    
    # Download file if in Colab
    try:
        from google.colab import files
        files.download(output_filename)
    except ImportError:
        print("Not in Colab environment - file saved locally.")
else:
    print("No results were generated.")