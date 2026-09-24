#!/usr/bin/env bash
# Run WQT20 test suite
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m pytest wqt20/tests/ -v "$@"
