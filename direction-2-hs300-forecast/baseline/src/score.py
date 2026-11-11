"""对 Top5 组合提交打分：按权重汇总测试窗口的个股收益。

真值文件列为 stock_code,actual_return，由出题方在测试窗口结束后生成。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MAX_STOCKS = 5


def load_prediction(path: str) -> pd.DataFrame:
    pred = pd.read_csv(path, dtype={"stock_id": str, "stock_code": str}, encoding="utf-8-sig")
    id_col = "stock_id" if "stock_id" in pred.columns else "stock_code" if "stock_code" in pred.columns else None
    if id_col is None or "weight" not in pred.columns:
        raise SystemExit("预测文件需包含 stock_id（或 stock_code）与 weight 两列")
    pred = pred.rename(columns={id_col: "stock_id"})
    pred["stock_id"] = pred["stock_id"].astype(str).str.zfill(6)
    if pred["stock_id"].duplicated().any():
        raise SystemExit("预测文件存在重复股票代码")
    if len(pred) == 0 or len(pred) > MAX_STOCKS:
        raise SystemExit(f"预测文件需包含 1–{MAX_STOCKS} 只股票，实际 {len(pred)} 只")
    if pred["weight"].isna().any() or (pred["weight"] < 0).any():
        raise SystemExit("权重不能为空或负数")
    weight_sum = float(pred["weight"].sum())
    if abs(weight_sum - 1.0) > 1e-4:
        raise SystemExit(f"权重合计应为 1，实际 {weight_sum}")
    return pred


def compute_metrics(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    truth = truth.rename(columns={"stock_code": "stock_id", "股票代码": "stock_id"})
    if "stock_id" not in truth.columns:
        raise SystemExit("真值文件需包含 stock_code 列")
    truth["stock_id"] = truth["stock_id"].astype(str).str.zfill(6)
    if truth["stock_id"].duplicated().any():
        raise SystemExit("真值文件存在重复股票代码")
    if "actual_return" not in truth.columns:
        raise SystemExit("真值文件需包含 actual_return 列")

    merged = pred.merge(truth[["stock_id", "actual_return"]], on="stock_id", how="left")
    missing = merged.loc[merged["actual_return"].isna(), "stock_id"].tolist()
    if missing:
        raise SystemExit(f"真值缺少 {len(missing)} 只股票，例如 {missing[:5]}")
    merged["contribution"] = merged["weight"] * merged["actual_return"]

    universe_mean = float(truth["actual_return"].mean())
    score = float(merged["contribution"].sum())
    return {
        "stocks": int(len(merged)),
        "weight_sum": float(merged["weight"].sum()),
        "portfolio_return": score,
        "universe_mean": universe_mean,
        "excess_vs_universe": score - universe_mean,
        "selected": merged.sort_values("weight", ascending=False).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--detail", default=None)
    args = parser.parse_args()

    pred = load_prediction(args.pred)
    truth = pd.read_csv(args.truth, dtype={"stock_code": str, "stock_id": str}, encoding="utf-8-sig")
    metrics = compute_metrics(pred, truth)

    if args.detail:
        detail = pd.DataFrame(metrics["selected"])
        Path(args.detail).parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.detail, index=False, encoding="utf-8-sig")

    printable = {key: value for key, value in metrics.items() if key != "selected"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(
        f"Portfolio Return = {metrics['portfolio_return']:.4%}   "
        f"Universe Mean = {metrics['universe_mean']:.4%}   "
        f"Excess = {metrics['excess_vs_universe']:.4%}"
    )


if __name__ == "__main__":
    main()
