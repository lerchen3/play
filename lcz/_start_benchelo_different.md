!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz

# Build C++ engine and active inference (if needed)
!OUT_DIR=/kaggle/working bash build_cpp.sh

# Run benchmark with JSON config directly from CLI
!LCZ_LIB_DIR=/kaggle/working python benchelo_different.py --use_fen_game --config_json '[{"path": "/kaggle/input/modelfive/best_eval_model (8).pt", "name": "regular_mcts", "num_mcts_sims": 1600, "cpuct": 1.0, "pure_network": false}, {"path": "/kaggle/input/modelthree/best_eval_model (6).pt", "name": "model3_pure", "pure_network": true}, {"path": "/kaggle/input/modelfive/best_eval_model (8).pt", "name": "model5_pure", "pure_network": true}]' --time_limit 41400 --stats_file elo_results.json --use_fen_game --reset_iterations 5 "/kaggle/input/modelfive/best_eval_model (8).pt" "/kaggle/input/modelthree/best_eval_model (6).pt"