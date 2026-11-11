"""年度滚动回测：每年只用此前数据训练，预测当年每周的 Top5 组合。

组合收益按 T+1 开盘买入、T+5 开盘卖出计算，与提交评分口径一致。
输出 reports/backtest_weekly.csv 与 reports/backtest_metrics.json。
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import train as T

TOP_N = 5
PURGE_DAYS = 5  # 训练样本的标签不能跨入测试年份

warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def evaluate_week(frame: pd.DataFrame, scores: np.ndarray, top_n: int = TOP_N) -> dict | None:
    scored = frame[["date", F.LABEL]].copy()
    scored["score"] = scores
    scored = scored.dropna(subset=["score", F.LABEL])
    if len(scored) < max(50, top_n):
        return None
    top = scored.nlargest(top_n, "score")
    portfolio_return = float(top[F.LABEL].mean())
    universe_return = float(scored[F.LABEL].mean())
    return {
        "date": scored["date"].iloc[0],
        "n": int(len(scored)),
        "portfolio_return": portfolio_return,
        "universe_return": universe_return,
        "excess": portfolio_return - universe_return,
        "positive": portfolio_return > 0,
        "beat_universe": portfolio_return > universe_return,
    }


def summarize(weekly: pd.DataFrame) -> dict:
    return {
        "weeks": int(len(weekly)),
        "avg_portfolio_return": float(weekly["portfolio_return"].mean()),
        "avg_universe_return": float(weekly["universe_return"].mean()),
        "avg_excess": float(weekly["excess"].mean()),
        "positive_rate": float(weekly["positive"].mean()),
        "beat_universe_rate": float(weekly["beat_universe"].mean()),
        "best_week": float(weekly["portfolio_return"].max()),
        "worst_week": float(weekly["portfolio_return"].min()),
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
    rng = np.random.default_rng(42)
    all_weekly = []

    for year in years:
        test_start = pd.Timestamp(f"{year}-01-01")
        prior_dates = calendar[calendar < test_start]
        if len(prior_dates) <= PURGE_DAYS:
            continue
        train_cutoff = prior_dates[-1 - PURGE_DAYS]
        train_frame = frame[frame["date"] <= train_cutoff]
        test_frame = frame[(frame["date"] >= test_start) & (frame["date"] < f"{year + 1}-01-01")]
        if len(train_frame) < 5000 or len(test_frame) == 0:
            continue
        print(f"{year}: train={len(train_frame)} (<= {train_cutoff.date()}), test={len(test_frame)}")

        model = T.fit_model(train_frame)
        predictions = {
            "lightgbm": model.predict(test_frame[F.MODEL_FEATURES]),
            "reversal_5d": -test_frame["ret_5"].fillna(0.0).to_numpy(),
        }
        random_scores = np.empty(len(test_frame))
        for date, idx in test_frame.groupby("date").groups.items():
            random_scores[test_frame.index.get_indexer(idx)] = rng.random(len(idx))
        predictions["random"] = random_scores

        for name, pred in predictions.items():
            for date, group in test_frame.groupby("date"):
                row = evaluate_week(group, pred[test_frame.index.isin(group.index)])
                if row is not None:
                    row["model"] = name
                    row["year"] = year
                    all_weekly.append(row)

    weekly = pd.DataFrame(all_weekly)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out / "backtest_weekly.csv", index=False, encoding="utf-8-sig")

    summary = {
        "period": f"{int(weekly['year'].min())}-{int(weekly['year'].max())}",
        "top_n": TOP_N,
        "purge_days": PURGE_DAYS,
        "overall": {name: summarize(group) for name, group in weekly.groupby("model")},
        "by_year": {
            f"{name}_{year}": summarize(group)
            for (name, year), group in weekly.groupby(["model", "year"])
        },
    }
    (out / "backtest_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
