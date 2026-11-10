#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${OUTPUT_DIR:-output}"
python src/predict.py --data "${DATA_DIR:-data}" --models "${MODEL_DIR:-models}" --out "${OUTPUT_DIR:-output}/result.csv"
