# %%bash # for jupyter environments
cp /kaggle/input/openr1pretrainlol/openr1_math220k_conversations.csv /kaggle/working/
cp /kaggle/input/bpe-bang/vocab.csv /kaggle/working/
cp -r /kaggle/input/deepwish /kaggle/working/

# compile C++ trie tokenizer shared library
OUT_DIR=/kaggle/working/deepwish/gpt/train bash /kaggle/working/deepwish/gpt/train/build_trie_tokenizer.sh

python -m torch.distributed.run --nproc_per_node=4 /kaggle/working/deepwish/gpt/train/train_fa2_cce_offload.py \
  --data_path /kaggle/working/openr1_math220k_conversations.csv \
  --text_column text \
  --conversational \
  --user_column user \
  --assistant_column assistant \
  --seq_len 2048 \
  --batch_size 1 \
  --epochs 1 \
  --lr 2e-5 \
  --grad_accum_steps 4 \
  --d_model 512 \
  --d_head 32 \
  --n_heads 16 \
  --dc_kv 16 \
  --dc_q 16 \
  --num_layers 12 \
  --rmsnorm_eps 1e-6 \
  --n_shared_experts 1 \
  --n_routed_experts 12 \
  --k_routed_experts 2 \
  --d_ff_expert_mult 2 \
  --vocab_csv_path /kaggle/working/vocab.csv \
  --moe_balance_factor 0.01 \
  --bias_update_speed 0.001 \
  --mtp_depth 4 \
  --mtp_weight 0.25 \
  --model_checkpoint None \
  --checkpoint_save_path /kaggle/working/continue_training_here_checkpoint \
  --model_save_path /kaggle/working/model \
  --val_split 0.0005 \
  --eval_interval 100 \
  --time_limit 41400 \
  --weight_decay 0.01 \
  --lr_warmup_steps 1000 \
  --lr_decay_steps 10000 \
  --checkpoint_dir /kaggle/working/checkpoint_and_plot \
  --max_grad_norm 1.0 \
  --ema_decay 0.9 \
  --offload