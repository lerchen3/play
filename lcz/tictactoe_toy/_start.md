!cp -r /kaggle/input/chesse/lcz /kaggle/working/lcz
%cd /kaggle/working/lcz

!g++ -std=c++17 -O3 -fPIC tictactoe.cpp -shared -o libtictactoe.so

# Train the simple TicTacToe agent
!python tictactoe_train.py

# interactive
%run -i tictactoe_play.py