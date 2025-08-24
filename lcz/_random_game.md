# run in kaggle notebook cell.
!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz
!OUT_DIR=/kaggle/working bash build_cpp.sh
!LCZ_LIB_DIR=/kaggle/working python random_game.py
