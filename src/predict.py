"""加载模型，生成指定交易日的 5 日收益预测。

默认使用数据快照中的最新交易日。输出 submissions/result.csv，
列为 stock_code,pred_return，共 300 行。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F


def predict_for_date(dataset: pd.DataFrame, model: lgb.Booster, features: list[str], date: str) -> pd.DataFrame:
    target = pd.Timestamp(date)
    frame = dataset[(dataset["date"] == target) & (dataset["hist_days"] >= 120)].copy()
    if frame.empty:
        available = dataset["date"].max().date()
        raise SystemExit(f"{date} 无可用特征；数据快照最新日期为 {available}")
    for col in features:
        frame[col] = frame[col].fillna(frame[col].median()).fillna(0.0)
    frame["pred_return"] = np.clip(model.predict(frame[features]), -0.5, 0.5)
    return frame[["code", "pred_return"]].rename(columns={"code": "stock_code"}).sort_values("stock_code")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--models", default="models")
    parser.add_argument("--date", default=None, help="预测日，默认数据快照最新交易日")
    parser.add_argument("--out", default="submissions/result.csv")
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
    if len(result) != 300:
        missing = 300 - len(result)
        raise SystemExit(f"预测行数 {len(result)}，缺少 {missing} 只；请检查数据快照和股票池")
    if result["stock_code"].duplicated().any() or result["pred_return"].isna().any():
        raise SystemExit("预测结果存在重复代码或空值")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved {len(result)} rows to {out} (date={target}, "
          f"mean={result['pred_return'].mean():.5f}, std={result['pred_return'].std():.5f})")


if __name__ == "__main__":
    main()
