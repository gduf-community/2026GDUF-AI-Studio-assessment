"""年度滚动回测：每年只用当年之前的数据训练，预测当年每周截面。

输出 reports/backtest_daily.csv 与 reports/backtest_metrics.json。
这是样本外参考指标，不代表未来测试窗口的真实成绩。
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as F
import train as T

TOP_N = 30
PURGE_DAYS = 5  # 训练样本的标签不能跨入测试年份

warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def evaluate(frame: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    scored = frame[["date", "fwd_ret_5"]].copy()
    scored["pred"] = pred
    rows = []
    for date, group in scored.groupby("date"):
        group = group.dropna(subset=["pred", "fwd_ret_5"])
        if len(group) < 50:
            continue
        group = group.sort_values("pred")
        rows.append(
            {
                "date": date,
                "n": len(group),
                "ic": spearmanr(group["pred"], group["fwd_ret_5"]).statistic,
                "long_short": group.tail(TOP_N)["fwd_ret_5"].mean()
                - group.head(TOP_N)["fwd_ret_5"].mean(),
                "mae": (group["pred"] - group["fwd_ret_5"]).abs().mean(),
                "direction": (np.sign(group["pred"]) == np.sign(group["fwd_ret_5"])).mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame) -> dict:
    ic = daily["ic"].astype(float)
    return {
        "dates": int(len(daily)),
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else None,
        "ic_positive_rate": float((ic > 0).mean()),
        "long_short_mean": float(daily["long_short"].mean()),
        "long_short_positive_rate": float((daily["long_short"] > 0).mean()),
        "mae_mean": float(daily["mae"].mean()),
        "direction_accuracy": float(daily["direction"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    dataset = F.build_dataset(args.data, cache=True)
    frame, _ = T.training_frame(dataset)
    calendar = pd.DatetimeIndex(sorted(dataset["date"].unique()))
    years = sorted(frame["date"].dt.year.unique())
    all_daily = []

    for year in years:
        test_start = pd.Timestamp(f"{year}-01-01")
        # 只保留 label 的 5 日窗口完全落在测试年之前的样本
        prior_dates = calendar[calendar < test_start]
        if len(prior_dates) <= PURGE_DAYS:
            continue
        train_cutoff = prior_dates[-1 - PURGE_DAYS]
        train_frame = frame[frame["date"] <= train_cutoff]
        test_frame = frame[(frame["date"] >= test_start) & (frame["date"] < f"{year + 1}-01-01")]
        if len(train_frame) < 5000 or len(test_frame) == 0:
            continue
        print(f"{year}: train={len(train_frame)} (<= {train_cutoff.date()}), test={len(test_frame)}")

        lgb_model = T.fit_model(train_frame)
        predictions = {"lightgbm": lgb_model.predict(test_frame[F.MODEL_FEATURES])}

        ridge = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
        ridge.fit(train_frame[F.MODEL_FEATURES], train_frame[F.LABEL].clip(-T.LABEL_CLIP, T.LABEL_CLIP))
        predictions["ridge"] = ridge.predict(test_frame[F.MODEL_FEATURES])
        predictions["reversal_5d"] = -test_frame["ret_5"].fillna(0.0).to_numpy()

        for name, pred in predictions.items():
            daily = evaluate(test_frame, pred)
            daily["model"] = name
            daily["year"] = year
            all_daily.append(daily)

    daily = pd.concat(all_daily, ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "backtest_daily.csv", index=False, encoding="utf-8-sig")

    summary = {
        "period": f"{int(daily['year'].min())}-{int(daily['year'].max())}",
        "top_n": TOP_N,
        "purge_days": PURGE_DAYS,
        "overall": {
            name: summarize(group) for name, group in daily.groupby("model")
        },
        "by_year": {
            f"{name}_{year}": summarize(group)
            for (name, year), group in daily.groupby(["model", "year"])
        },
    }
    (out / "backtest_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
