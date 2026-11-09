"""对提交文件打分：Rank IC、多空收益、MAE、方向准确率。

真值文件列为 stock_code,actual_return，由出题方在测试窗口结束后生成。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TOP_N = 30


def compute_metrics(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    pred = pred.rename(columns={pred.columns[0]: "stock_code"})
    truth = truth.rename(columns={truth.columns[0]: "stock_code"})
    pred["stock_code"] = pred["stock_code"].astype(str).str.zfill(6)
    truth["stock_code"] = truth["stock_code"].astype(str).str.zfill(6)
    if pred["stock_code"].duplicated().any():
        raise SystemExit("预测文件存在重复股票代码")
    if truth["stock_code"].duplicated().any():
        raise SystemExit("真值文件存在重复股票代码")
    merged = truth.merge(pred, on="stock_code", how="left", validate="one_to_one")
    if merged["pred_return"].isna().any():
        missing = merged.loc[merged["pred_return"].isna(), "stock_code"].tolist()
        raise SystemExit(f"预测缺少 {len(missing)} 只股票，例如 {missing[:5]}")
    merged = merged.dropna(subset=["actual_return", "pred_return"])
    if len(merged) < 100:
        raise SystemExit(f"有效股票数仅 {len(merged)}，不足以评估")

    ic = spearmanr(merged["pred_return"], merged["actual_return"]).statistic
    ranked = merged.sort_values("pred_return")
    long_short = ranked.tail(TOP_N)["actual_return"].mean() - ranked.head(TOP_N)["actual_return"].mean()
    mae = (merged["pred_return"] - merged["actual_return"]).abs().mean()
    direction = (np.sign(merged["pred_return"]) == np.sign(merged["actual_return"])).mean()
    return {
        "stocks": int(len(merged)),
        "rank_ic": float(ic),
        "long_short": float(long_short),
        "mae": float(mae),
        "direction_accuracy": float(direction),
        "pred_mean": float(merged["pred_return"].mean()),
        "actual_mean": float(merged["actual_return"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--detail", default=None)
    args = parser.parse_args()

    pred = pd.read_csv(args.pred, dtype={"stock_code": str})
    truth = pd.read_csv(args.truth, dtype={"stock_code": str})
    if "pred_return" not in pred.columns:
        raise SystemExit("预测文件需包含 pred_return 列")
    if "actual_return" not in truth.columns:
        raise SystemExit("真值文件需包含 actual_return 列")
    if len(pred) != len(truth):
        raise SystemExit(f"预测 {len(pred)} 行、真值 {len(truth)} 行；提交文件必须与股票池一一对应")
    metrics = compute_metrics(pred, truth)
    if args.detail:
        merged = truth.merge(pred, on="stock_code", how="left")
        merged["abs_error"] = (merged["pred_return"] - merged["actual_return"]).abs()
        Path(args.detail).parent.mkdir(parents=True, exist_ok=True)
        merged.sort_values("abs_error", ascending=False).to_csv(
            args.detail, index=False, encoding="utf-8-sig"
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        f"Rank IC = {metrics['rank_ic']:.4f}   LS = {metrics['long_short']:.4f}   "
        f"MAE = {metrics['mae']:.4f}   Dir = {metrics['direction_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
