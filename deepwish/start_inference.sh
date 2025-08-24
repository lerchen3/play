%%bash

# Copy model and code repo from input to working directory
cp /kaggle/input/deepwish /kaggle/working/ -r

# Ensure Python can import the local package
export PYTHONPATH=/kaggle/working/deepwish/gpt:${PYTHONPATH}
# Add current directory for relative imports
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Run inference
python /kaggle/working/deepwish/gpt/inference/inference.py \
  --model_save_path "/kaggle/input/deepwishmodel/continue_training_here_checkpoint" \
  --tokenizer_path "/kaggle/input/qwen-3/transformers/0.6b/1" \
  --prompt "Find the value of 1+1." \
  --device "cuda" \
  --max_new_tokens 128 \
  --top_k 0 \
  --top_p 1.0 \
  --temperature 1.0