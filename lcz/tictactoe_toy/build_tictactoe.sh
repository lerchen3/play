#!/usr/bin/env bash
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SRC_DIR}"
mkdir -p "$OUT_DIR"

g++ -std=c++17 -O3 -fPIC "$SRC_DIR/tictactoe.cpp" -shared -o "$OUT_DIR/libtictactoe.so"
