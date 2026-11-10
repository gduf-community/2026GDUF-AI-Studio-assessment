"""朴素基准：过去 5 日反转，选出跌幅最大的 5 只股票等权持有。

A 股短周期存在反转效应，这是最直接的零参数对照。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import features as F

TOP_N = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--date", default=None)
    parser.add_argument("--out", default="output/naive_result.csv")
    args = parser.parse_args()

    dataset = F.build_dataset(args.data, cache=True)
    target = args.date or str(dataset["date"].max().date())
    frame = dataset[(dataset["date"] == pd.Timestamp(target)) & (dataset["hist_days"] >= 120)].copy()
    frame = frame.dropna(subset=["ret_5"])
    if len(frame) < TOP_N:
        raise SystemExit(f"{target} 可评分股票仅 {len(frame)} 只，不足 {TOP_N} 只")

    top = frame.nsmallest(TOP_N, "ret_5")
    result = pd.DataFrame(
        {
            "stock_id": top["code"].astype(str).str.zfill(6).to_numpy(),
            "weight": np.full(TOP_N, 1.0 / TOP_N),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8")
    print(f"saved {len(result)} rows to {out} (date={target})")


if __name__ == "__main__":
    main()
