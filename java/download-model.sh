#!/usr/bin/env bash
# Downloads the ONNX export + tokenizer of BAAI/bge-m3.
# The ONNX graph keeps its weights in a separate model.onnx_data file, so both
# must land in the same directory for ONNX Runtime to load them.
set -euo pipefail

DIR="$(dirname "$0")/models/bge-m3"
BASE="https://huggingface.co/BAAI/bge-m3/resolve/main"

mkdir -p "$DIR"
curl -L --fail -o "$DIR/model.onnx"      "$BASE/onnx/model.onnx"
curl -L --fail -o "$DIR/model.onnx_data" "$BASE/onnx/model.onnx_data"
curl -L --fail -o "$DIR/tokenizer.json"  "$BASE/tokenizer.json"

echo "Model files ready in $DIR"
