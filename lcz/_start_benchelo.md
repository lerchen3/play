# run in kaggle notebook cell.
!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz
!OUT_DIR=/kaggle/working bash build_cpp.sh
!LCZ_LIB_DIR=/kaggle/working python benchelo.py \
  --num_games 20 \
  --stats_file elo_stats.json \
  --old_stats_pth None \
  --max_moves 400 \
  --in_channels 103 \
  --num_res_blocks 40 \
  --num_filters 256 \
  --num_mcts_sims 1600 \
  --cpuct 1.0 \
  --batch_mcts_size 32 \
  --root_noise_eps 0.25 \
  --dirichlet_alpha 0.03 \
  --virtual_loss 0.1 \
  "/kaggle/input/spckpt1/model_iter40.pt" "/kaggle/input/ckptone/pretrain_latest (2).pt"
