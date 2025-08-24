%%bash

# Copy dataset and code repo from input to working directory
cp /kaggle/input/openr1pretrainlol/openr1_math220k_conversations.csv /kaggle/working/
cp -r /kaggle/input/deepwish /kaggle/working/

# Ensure Python can import the local package
export PYTHONPATH=/kaggle/working/deepwish/gpt:${PYTHONPATH}

# Launch distributed training with the new fp32 offload trainer using chat template
python -m torch.distributed.run --nproc_per_node=4 /kaggle/working/deepwish/gpt/train/train.py \
  --data_path /kaggle/working/openr1_math220k_conversations.csv \
  --user_column user \
  --assistant_column assistant \
  --seq_len 1024 \
  --batch_size 64 \
  --epochs 1 \
  --lr 2e-5 \
  --grad_accum_steps 4 \
  --d_model 128 \
  --d_head 16 \
  --n_heads 8 \
  --dc_kv 16 \
  --dc_q 16 \
  --num_layers 4 \
  --rmsnorm_eps 1e-6 \
  --n_shared_experts 1 \
  --n_routed_experts 2 \
  --k_routed_experts 1 \
  --d_ff_expert_mult 2 \
  --moe_balance_factor 0.01 \
  --bias_update_speed 0.001 \
  --mtp_depth 3 \
  --mtp_weight 0.33 \
  --model_checkpoint None \
  --checkpoint_save_path /kaggle/working/continue_training_here_checkpoint \
  --model_save_path /kaggle/working/model \
  --val_split 0.005 \
  --eval_interval 100 \
  --time_limit 41400 \
  --weight_decay 0.01 \
  --lr_warmup_steps 1000 \
  --lr_decay_steps 10000 \
  --checkpoint_dir /kaggle/working/checkpoint_and_plot \
  --max_grad_norm 1.0 \
  --ema_decay 0.9 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_eps 1e-8
