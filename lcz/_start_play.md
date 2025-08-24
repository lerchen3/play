# run in kaggle notebook cell.
!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz

# Build the C++ engine
!OUT_DIR=/kaggle/working bash build_cpp.sh

# Play against AI with configurable parameters
!LCZ_LIB_DIR=/kaggle/working python play.py \
  --model_path "/kaggle/input/ckptone/pretrain_latest (2).pt" \
  --max_moves 200 \
  --num_mcts_sims 800 \
  --cpuct 1.0 \
  --show_top_moves 5 \
  --in_channels 103 \
  --num_res_blocks 40 \
  --num_filters 256
