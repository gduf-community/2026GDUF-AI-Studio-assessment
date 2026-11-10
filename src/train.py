"""训练 5 日收益预测模型（LightGBM）。

按周采样训练样本，标签为 T+1 开盘买入、T+5 开盘卖出的收益，训练前做 ±30% 缩尾。
模型文件保存到 models/，供 predict.py 加载。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F

PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
MIN_HISTORY = 120
LABEL_CLIP = 0.30


def training_frame(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    calendar = pd.DatetimeIndex(sorted(dataset["date"].unique()))
    dates = F.weekly_dates(calendar)
    frame = dataset[dataset["date"].isin(dates)]
    frame = frame[(frame["hist_days"] >= MIN_HISTORY) & frame[F.LABEL].notna()].copy()
    return frame, dates


def fit_model(frame: pd.DataFrame) -> lgb.LGBMRegressor:
    y = frame[F.LABEL].clip(-LABEL_CLIP, LABEL_CLIP)
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(frame[F.MODEL_FEATURES], y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    dataset = F.build_dataset(args.data, cache=True)
    frame, _ = training_frame(dataset)
    model = fit_model(frame)

    out = Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "lightgbm.txt"))
    (out / "features.json").write_text(
        json.dumps(F.MODEL_FEATURES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = {
        "label": F.LABEL,
        "label_clip": LABEL_CLIP,
        "min_history": MIN_HISTORY,
        "train_start": str(frame["date"].min().date()),
        "train_end": str(frame["date"].max().date()),
        "training_rows": int(len(frame)),
        "n_features": len(F.MODEL_FEATURES),
        "params": PARAMS,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
