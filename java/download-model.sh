#!/usr/bin/env bash
# Downloads the ONNX export + tokenizer of intfloat/multilingual-e5-small.
set -euo pipefail

DIR="$(dirname "$0")/models/multilingual-e5-small"
BASE="https://huggingface.co/intfloat/multilingual-e5-small/resolve/main"

mkdir -p "$DIR"
curl -L --fail -o "$DIR/model.onnx"     "$BASE/onnx/model.onnx"
curl -L --fail -o "$DIR/tokenizer.json" "$BASE/tokenizer.json"

echo "Model files ready in $DIR"
