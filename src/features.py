"""特征工程：把日线面板转换为 (date, code) 的横截面样本。

官方快照支持 train.csv.gz（stock_code/date/OHLCV）或旧版 prices.csv.gz。
所有特征只使用当日及之前的数据；标签为 T+1 开盘到 T+5 开盘的收益。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "ret_120",
    "vol_20", "vol_60",
    "ma_gap_5", "ma_gap_10", "ma_gap_20", "ma_gap_60",
    "rsi_14", "range_pos_20", "dd_60",
    "high_52w_gap", "low_52w_gap",
    "volume_ratio", "amount_ratio", "turnover_20", "log_amount_20",
    "beta_60", "corr_60", "skew_20", "kurt_20",
    "idx_ret_5", "idx_ret_20", "idx_ret_60", "idx_vol_20",
    "breadth", "dispersion",
]
LABEL = "fwd_ret_5"
MARKET_FEATURES = ["idx_ret_5", "idx_ret_20", "idx_ret_60", "idx_vol_20", "breadth", "dispersion"]
CS_BASE = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14", "log_amount_20", "turnover_20", "range_pos_20", "dd_60"]
MODEL_FEATURES = [f for f in FEATURES if f not in MARKET_FEATURES] + [f"cs_{name}" for name in CS_BASE]

ALIASES = {
    "stock_code": "code", "股票代码": "code",
    "date": "date", "日期": "date",
    "open": "open", "开盘": "open",
    "close": "close", "收盘": "close",
    "high": "high", "最高": "high",
    "low": "low", "最低": "low",
    "volume": "volume", "成交量": "volume",
    "amount": "amount", "成交额": "amount",
    "turnover": "turnover", "换手率": "turnover",
    "stock_name": "name", "名称": "name",
}


def _first_existing(data_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{data_dir} 下缺少数据文件，尝试过 {names}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={col: ALIASES.get(col, col) for col in df.columns})
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_data(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    universe = _normalize(pd.read_csv(_first_existing(data_dir, ["universe.csv"])))
    if "name" not in universe.columns:
        universe["name"] = universe["code"]
    prices = _normalize(
        pd.read_csv(
            _first_existing(data_dir, ["train.csv.gz", "train.csv", "prices.csv.gz", "prices.csv"]),
            dtype={"code": str, "stock_code": str},
        )
    )
    prices = prices.sort_values(["code", "date"]).reset_index(drop=True)

    try:
        index = _normalize(pd.read_csv(_first_existing(data_dir, ["index.csv"])))
    except FileNotFoundError:
        # 没有指数文件时，用全池等权收益合成一条市场基准，仅用于 Beta/相关性
        ret = prices.groupby("code", sort=False)["close"].pct_change(fill_method=None)
        synthetic = ret.groupby(prices["date"]).mean().fillna(0.0).add(1.0).cumprod() * 1000.0
        index = pd.DataFrame({"date": synthetic.index, "close": synthetic.values})

    try:
        calendar = pd.read_csv(data_dir / "calendar.csv")
        calendar["trade_date"] = pd.to_datetime(calendar["trade_date"])
    except FileNotFoundError:
        calendar = pd.DataFrame({"trade_date": sorted(index["date"].unique())})
    return {"universe": universe, "prices": prices, "index": index, "calendar": calendar}


def to_wide(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    out = {}
    for col in ["open", "close", "high", "low", "volume", "amount", "turnover"]:
        wide = (
            prices.pivot(index="date", columns="code", values=col)
            .reindex(calendar)
            .sort_index()
        )
        out[col] = wide
    raw_close = out["close"].copy()
    out["close"] = out["close"].ffill()
    listed = raw_close.notna().cummax()
    for col in ["volume", "amount", "turnover"]:
        out[col] = out[col].fillna(0.0).where(listed)
    out["listed"] = listed
    out["hist_days"] = raw_close.notna().cumsum()
    return out


def build_features(wide: dict[str, pd.DataFrame], index: pd.DataFrame) -> pd.DataFrame:
    close, high, low = wide["close"], wide["high"], wide["low"]
    open_px = wide["open"].where(wide["open"] > 0)
    volume, amount, turnover = wide["volume"], wide["amount"], wide["turnover"]
    ret = close.pct_change(fill_method=None)
    feats: dict[str, pd.DataFrame] = {}

    for n in [1, 5, 10, 20, 60, 120]:
        feats[f"ret_{n}"] = close.pct_change(n, fill_method=None)
    for n in [20, 60]:
        feats[f"vol_{n}"] = ret.rolling(n).std()
    for n in [5, 10, 20, 60]:
        feats[f"ma_gap_{n}"] = close / close.rolling(n).mean() - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    low20, high20 = low.rolling(20).min(), high.rolling(20).max()
    feats["range_pos_20"] = (close - low20) / (high20 - low20).replace(0, np.nan)
    feats["dd_60"] = close / close.rolling(60).max() - 1.0
    feats["high_52w_gap"] = close / close.rolling(252).max() - 1.0
    feats["low_52w_gap"] = close / close.rolling(252).min() - 1.0

    feats["volume_ratio"] = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
    feats["amount_ratio"] = amount.rolling(5).mean() / amount.rolling(20).mean().replace(0, np.nan)
    feats["turnover_20"] = turnover.rolling(20).mean()
    feats["log_amount_20"] = np.log1p(amount.rolling(20).mean())

    idx_close = index.set_index("date")["close"].reindex(close.index).astype(float)
    idx_ret = idx_close.pct_change(fill_method=None)
    win = 60
    cov = ret.mul(idx_ret, axis=0).rolling(win).mean() - ret.rolling(win).mean().mul(
        idx_ret.rolling(win).mean(), axis=0
    )
    feats["beta_60"] = cov.div(idx_ret.rolling(win).var(), axis=0)
    feats["corr_60"] = ret.rolling(60).corr(idx_ret)
    feats["skew_20"] = ret.rolling(20).skew()
    feats["kurt_20"] = ret.rolling(20).kurt()

    stack = pd.concat(feats, axis=1)
    stack.columns = pd.MultiIndex.from_product([list(feats), close.columns], names=["feature", "code"])
    long = (
        stack.stack(level="code", future_stack=True)
        .rename_axis(index=["date", "code"])
        .reset_index()
    )

    market = pd.DataFrame({
        "idx_ret_5": idx_close.pct_change(5, fill_method=None),
        "idx_ret_20": idx_close.pct_change(20, fill_method=None),
        "idx_ret_60": idx_close.pct_change(60, fill_method=None),
        "idx_vol_20": idx_ret.rolling(20).std(),
        "breadth": (close > close.rolling(20).mean()).mean(axis=1),
        "dispersion": feats["ret_5"].std(axis=1),
    }, index=close.index)
    market.index.name = "date"
    long = long.merge(market.reset_index(), on="date", how="left")

    fwd = open_px.shift(-5) / open_px.shift(-1) - 1.0
    label = fwd.stack(future_stack=True).rename(LABEL).reset_index()
    label.columns = ["date", "code", LABEL]
    long = long.merge(label, on=["date", "code"], how="left")

    hist = wide["hist_days"].stack(future_stack=True).rename("hist_days").reset_index()
    hist.columns = ["date", "code", "hist_days"]
    long = long.merge(hist, on=["date", "code"], how="left")
    for name in CS_BASE:
        long[f"cs_{name}"] = long.groupby("date")[name].rank(pct=True)
    for col in FEATURES + [f"cs_{name}" for name in CS_BASE] + [LABEL]:
        long[col] = pd.to_numeric(long[col], errors="coerce").astype("float32")
    return long


DATA_FILES = [
    "universe.csv",
    "index.csv",
    "calendar.csv",
    "train.csv.gz",
    "train.csv",
    "prices.csv.gz",
    "prices.csv",
]


def _cache_is_fresh(data_dir: Path, cache_path: Path) -> bool:
    """缓存早于任一原始文件时失效，避免换了快照却读到旧特征。"""
    sources = [data_dir / name for name in DATA_FILES if (data_dir / name).exists()]
    if not sources:
        return False
    newest = max(src.stat().st_mtime for src in sources)
    return cache_path.stat().st_mtime >= newest


def build_dataset(data_dir: str | Path = "data", cache: bool = True) -> pd.DataFrame:
    data_dir = Path(data_dir)
    cache_path = data_dir / "dataset.pkl"
    if cache and cache_path.exists() and _cache_is_fresh(data_dir, cache_path):
        return pd.read_pickle(cache_path)
    if cache and cache_path.exists():
        print("dataset cache is older than raw data, rebuilding")
    raw = load_data(data_dir)
    calendar = pd.DatetimeIndex(raw["calendar"]["trade_date"].sort_values().unique())
    wide = to_wide(raw["prices"], calendar)
    long = build_features(wide, raw["index"])
    long = long.sort_values(["date", "code"]).reset_index(drop=True)
    if cache:
        try:
            long.to_pickle(cache_path)
        except OSError:
            print(f"warning: {cache_path} 不可写，跳过特征缓存（不影响本次运行）")
    return long


def weekly_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(pd.DatetimeIndex(calendar))
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("W")).max().sort_values().values)


def make_model_frame(dataset: pd.DataFrame, dates: pd.DatetimeIndex, min_history: int = 120) -> pd.DataFrame:
    df = dataset[dataset["date"].isin(dates)].copy()
    df = df[df["hist_days"] >= min_history]
    return df
