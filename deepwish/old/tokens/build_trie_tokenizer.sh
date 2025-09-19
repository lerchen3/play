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
    -c "$SRC_DIR/trie_tokenizer.cpp"

# Link into a shared library
g++ -shared trie_tokenizer.o -o "$OUT_DIR/libtrie_tokenizer.so"

# Cleanup is handled by the trap 