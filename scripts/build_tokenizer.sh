#!/usr/bin/env bash
# Build the WQT20 tokenizer and save to wqt20/tokenizer/
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m wqt20.tokenizer
echo "Tokenizer built at wqt20/tokenizer/"
