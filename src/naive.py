"""朴素基准：过去 5 日反转（预测值 = - 过去 5 日收益）。

A 股短周期存在反转效应，这是最直接的零参数对照。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import features as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--date", default=None)
    parser.add_argument("--out", default="submissions/naive_result.csv")
    args = parser.parse_args()

    dataset = F.build_dataset(args.data, cache=True)
    target = args.date or str(dataset["date"].max().date())
    frame = dataset[(dataset["date"] == pd.Timestamp(target)) & (dataset["hist_days"] >= 120)].copy()
    if frame.empty:
        raise SystemExit(f"{target} 无可用数据")
    frame["pred_return"] = -frame["ret_5"].fillna(0.0)
    result = (
        frame[["code", "pred_return"]]
        .rename(columns={"code": "stock_code"})
        .sort_values("stock_code")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {len(result)} rows to {out} (date={target})")


if __name__ == "__main__":
    main()
