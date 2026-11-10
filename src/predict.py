"""加载模型，在指定交易日选出 Top5 股票并等权配置。

输出 output/result.csv，列为 stock_id,weight，共 5 行、权重合计为 1。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F

TOP_N = 5


def predict_for_date(
    dataset: pd.DataFrame,
    model: lgb.Booster,
    features: list[str],
    date: str,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    target = pd.Timestamp(date)
    frame = dataset[(dataset["date"] == target) & (dataset["hist_days"] >= 120)].copy()
    if frame.empty:
        available = dataset["date"].max().date()
        raise SystemExit(f"{date} 无可用特征；数据快照最新日期为 {available}")
    for col in features:
        frame[col] = frame[col].fillna(frame[col].median()).fillna(0.0)
    frame["score"] = model.predict(frame[features])
    frame = frame.dropna(subset=["score"])
    if len(frame) < top_n:
        raise SystemExit(f"{date} 可评分股票仅 {len(frame)} 只，不足 {top_n} 只")

    top = frame.sort_values(["score", "code"], ascending=[False, True]).head(top_n)
    return pd.DataFrame(
        {
            "stock_id": top["code"].astype(str).str.zfill(6).to_numpy(),
            "weight": np.full(len(top), 1.0 / top_n),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    parser.add_argument("--date", default=None, help="预测日，默认数据快照最新交易日")
    parser.add_argument("--out", default="output/result.csv")
    args = parser.parse_args()

    dataset = F.build_dataset(args.data, cache=True)
    target = args.date or str(dataset["date"].max().date())
    model = lgb.Booster(model_file=str(Path(args.models) / "lightgbm.txt"))
    features = json.loads((Path(args.models) / "features.json").read_text(encoding="utf-8"))

    meta_path = Path(args.models) / "meta.json"
    if meta_path.exists():
        train_end = pd.Timestamp(json.loads(meta_path.read_text(encoding="utf-8"))["train_end"])
        if train_end >= pd.Timestamp(target):
            raise SystemExit(
                f"模型训练截止 {train_end.date()} 不早于预测日 {target}：训练标签已用到预测日之后的价格。"
                "请用冻结到 T 的数据重新运行 train.py"
            )
    else:
        print("warning: models/meta.json 不存在，无法校验模型训练截止日")

    result = predict_for_date(dataset, model, features, target)
    if len(result) != TOP_N:
        raise SystemExit(f"预测结果应为 {TOP_N} 行，实际 {len(result)} 行")
    if result["stock_id"].duplicated().any():
        raise SystemExit("预测结果存在重复股票代码")
    weight_sum = float(result["weight"].sum())
    if abs(weight_sum - 1.0) > 1e-8:
        raise SystemExit(f"权重合计应为 1，实际为 {weight_sum}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8")
    print(
        f"saved {len(result)} rows to {out} (date={target}, "
        f"weight_sum={weight_sum:.4f}, top={','.join(result['stock_id'])}"
    )


if __name__ == "__main__":
    main()
