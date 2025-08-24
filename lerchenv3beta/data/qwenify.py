import os
import pandas as pd
from transformers import set_seed
import warnings
from vllm import LLM, SamplingParams

# ---------------------
# Hyperparameters
# ---------------------
INPUT_CSV = "/kaggle/input/limoverireal/LIMOVERI_ALL.csv"            # path to input CSV with 'prompt' and 'proposed_response'
OUTPUT_CSV = "qwenifiedlimoveri"          # path where to save rephrased results
MODEL_PATH = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"  # path to the LLM model directory

BATCH_NUM = 0
BATCH_SIZE = 100
MINIBATCH_SIZE = 4                     # number of examples per inference call (<= MAX_NUM_SEQS)
MAX_MODEL_LEN = 32768               # model context length
MAX_NUM_SEQS = 4                   # number of sequences per inference call

TEMPERATURE = 0.0                  # randomness of sampling
MAX_TOKENS = 16384                    # max tokens to generate for rephrasing
SEED = 2024                         # random seed for reproducibility

# Tensor parallelism and memory
TENSOR_PARALLEL_SIZE = 4            # number of GPUs to use for tensor parallelism
GPU_MEMORY_UTILIZATION = 0.95       # ratio of GPU memory to reserve for the model

# ---------------------
# Setup & Initialization
# ---------------------
# reproducible randomness
set_seed(SEED)

# suppress unwanted logs
warnings.simplefilter('ignore')

# safe environment settings (same as other inference scripts)
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# initialize LLM
llm = LLM(
    MODEL_PATH,
    max_model_len=MAX_MODEL_LEN,
    max_num_seqs=MAX_NUM_SEQS,
    trust_remote_code=True,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    seed=SEED,
)
# get the tokenizer for chat template
tokenizer = llm.get_tokenizer()

# sampling parameters
sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    skip_special_tokens=True,
    max_tokens=MAX_TOKENS,
)


def rephrase_responses(input_csv: str, output_csv: str):
    # load input data
    df = pd.read_csv(input_csv)
    prompts = df["Question"].tolist()
    proposals = df["text"].tolist()

    results = []
    # process in batches
    for i in range(0, len(prompts), MINIBATCH_SIZE):
        if(i < BATCH_NUM * BATCH_SIZE or i >= (BATCH_NUM + 1) * BATCH_SIZE):
            continue
        batch_prompts = prompts[i : i + MINIBATCH_SIZE]
        batch_proposals = proposals[i : i + MINIBATCH_SIZE]
        list_of_messages = []
        # build conversation for each example
        for prompt, proposal in zip(batch_prompts, batch_proposals):
            messages = [
                {"role": "user", "content": f"""Please add to the proposed reasoning process to be more understandable, fleshed out, and verbose. Make sure to include all parts, and just elaborate or rephrase on the reasoning process so that you yourself make even more sense of it. Your rephrased reasoning process should be MORE verbose, and simply scaffold the original reasoning process; don't get rid of anything and just make it more understandable to yourself; it should follow the same structure and trajectory. YOUR FINAL REASONING PROCESS SHOULD BE LONGER THAN THE ORIGINAL; you should really just be adding on to the original one, and possibly rephrasing but NOT condensing it. I do NOT want you to summarize it. I do NOT want you to remove parts of the reasoning process that did not end up leading to the answer, because they are NECESSARY for a coherent reasoning process. Please Return the rephrased reasoning process only, without any other text.
                 
                 Question: {prompt}\n\nCurrent reasoning process: {proposal}"""}
            ]
            list_of_messages.append(messages)

        # apply chat template for model input
        list_of_texts = [
            tokenizer.apply_chat_template(conversation=msgs, tokenize=False, add_generation_prompt=True)
            for msgs in list_of_messages
        ]

        # run inference
        outputs = llm.generate(prompts=list_of_texts, sampling_params=sampling_params)

        # collect outputs
        for prompt, proposal, single_output in zip(batch_prompts, batch_proposals, outputs):
            rephrased = single_output.outputs[0].text.strip()
            results.append({
                "Question": prompt,
                "text": proposal,
                "rephrased_response": rephrased,
                "delta_length": len(rephrased) - len(proposal)
            })

    # save results
    out_df = pd.DataFrame(results)
    batch_output_csv = f"output_{BATCH_NUM}.csv"
    out_df.to_csv(batch_output_csv, index=False)


if __name__ == "__main__":
    rephrase_responses(INPUT_CSV, OUTPUT_CSV)
