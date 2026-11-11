#!/usr/bin/env bash
set -euo pipefail
python src/train.py --data "${DATA_DIR:-data}" --models "${MODEL_DIR:-models}"
