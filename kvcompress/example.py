from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import KNormCache
from datasets import load_dataset

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

question = """What is the probability of two integers selected at random having a greatest common divisor of 1."""
input_text = f"<|User|>{question}<|Assistant|><think>\n"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

past_key_values = KNormCache(
    window_length=64,
    max_length=128,
)


out = model.generate(
    **inputs,
    do_sample=True, 
    temperature=0.5, 
    max_new_tokens=4096, 
    past_key_values=past_key_values, 
    streamer=streamer
)