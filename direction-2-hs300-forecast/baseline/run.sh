#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${MODEL_DIR:-models}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
PREDICT_DATE="${PREDICT_DATE:-}"

mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}"
python src/train.py --data "${DATA_DIR}" --models "${MODEL_DIR}"
if [[ -n "${PREDICT_DATE}" ]]; then
  python src/predict.py --data "${DATA_DIR}" --models "${MODEL_DIR}" --date "${PREDICT_DATE}" --out "${OUTPUT_DIR}/result.csv"
else
  python src/predict.py --data "${DATA_DIR}" --models "${MODEL_DIR}" --out "${OUTPUT_DIR}/result.csv"
fi
