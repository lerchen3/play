# run in kaggle notebook cell.
!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz

# Build the active inference C++ module
!OUT_DIR=/kaggle/working bash build_cpp_active.sh

# Train using active inference MCTS
!LCZ_LIB_DIR=/kaggle/working python train_active.py \
  --games 100 \
  --checkpoint-freq 25 \
  --checkpoint-dir models_active \
  --model "/kaggle/input/ckptone/pretrain_latest (2).pt" 