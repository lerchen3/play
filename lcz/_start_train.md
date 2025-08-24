# run in kaggle notebook cell.
!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz

# Compile the C++ engine to a writable path
!OUT_DIR=/kaggle/working bash build_cpp.sh

# Train using the library from /kaggle/working
!LCZ_LIB_DIR=/kaggle/working python train.py \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --model_dir models \
  --num_iterations 50 \
  --games_per_iter 5 \
  --epochs_per_iter 2 \
  --replay_buffer_size 10000 \
  --entropy_bonus 0.01 \
  --weight_decay 0.0001 \
  --lr_step_size 5 \
  --lr_gamma 0.9 \
  --max_moves 1000 \
  --in_channels 103 \
  --num_res_blocks 40 \
  --num_filters 256 \
  --num_mcts_sims 1600 \
  --cpuct 1.0 \
  --batch_mcts_size 32 \
  --root_noise_eps 0.25 \
  --dirichlet_alpha 0.03 \
  --virtual_loss 0.1 \
  --use_fen_game \
  --load_checkpoint_path "/kaggle/input/ckptone/pretrain_latest (2).pt" \
  --reset_iterations 1