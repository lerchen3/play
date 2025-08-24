import pandas as pd 
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
print("imports done?")
local_model_path = "solve_model"

problems_df = pd.read_csv('results.csv')
calibration_texts = []
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
for idx, row in problems_df.iterrows():
    text = "<｜begin▁of▁sentence｜>" + "<｜User｜>" + row['Question'] + "<｜Assistant｜>" + row['Correct_Generation']
    calibration_texts.append(text)

quant_path = 'model-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

calibration_texts = list(set(calibration_texts))
# Load model
model = AutoAWQForCausalLM.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_texts,  # Use the original texts
    n_parallel_calib_samples=4,
    max_calib_samples=40, 
    max_calib_seq_len=12291
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')