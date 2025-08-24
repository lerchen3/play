#!/usr/bin/env bash
# If CRLF line endings exist, normalize this script and re-run under bash
if grep -q $'\r' "$0"; then
    sed -i 's/\r$//' "$0"
    exec bash "$0" "$@"
fi
# Enable strict mode
set -eu

# Determine script directory and switch to it so the build works from any CWD
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC_DIR"

# Output directory for the resulting library. Useful on read-only filesystems.
OUT_DIR="${OUT_DIR:-$SRC_DIR}"
mkdir -p "$OUT_DIR"

# Use a unique temporary directory to avoid race conditions
TMP_DIR=$(mktemp -d -p "$OUT_DIR" build_XXXXXX)
trap "rm -rf $TMP_DIR" EXIT

# Compile the object files in the temporary directory
cd "$TMP_DIR"
g++ -std=c++17 -O3 -DNDEBUG -fPIC -I"$SRC_DIR" \
    -c "$SRC_DIR/position.cpp" "$SRC_DIR/mcts.cpp" "$SRC_DIR/c_api.cpp" "$SRC_DIR/tables.cpp" "$SRC_DIR/config.cpp"

# Link into a shared library
g++ -shared position.o mcts.o c_api.o tables.o config.o -o "$OUT_DIR/liblcz.so"

# Cleanup is handled by the trap
