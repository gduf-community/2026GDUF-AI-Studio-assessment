#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-models}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"
python src/train.py --data "${DATA_DIR}" --models "${MODEL_DIR}"
python src/predict.py --data "${DATA_DIR}" --models "${MODEL_DIR}" --out "${OUTPUT_DIR}/result.csv"
