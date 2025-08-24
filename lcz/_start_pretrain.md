# run in kaggle notebook cell.
!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz
!LCZ_LIB_DIR=/kaggle/working accelerate launch pretrain.py \
  --data_csv_path /kaggle/input/chesspretrain2/chesspretrain.csv \
  --input_model "/kaggle/input/number-one/best_eval_model (1).pt" \
  --output_model blah.pt \
  --epochs 1 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --start_row 900000 \ # 1e7 ish total lmfao
  --time_limit 41400 \
  --eval_fraction 0.05 \
  --eval_steps 1000 \
  --logging_steps 100 \
  --use_fen_game \
  --chunksize 100 \
  --in_channels 103 \
  --num_res_blocks 40 \
  --num_filters 256