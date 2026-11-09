"""下载沪深300成分股与后复权日线，生成本地数据快照。

数据源：东方财富公开行情接口，无需 token。首次运行会抓取 300 只股票，
大约 1 分钟；再次运行默认复用缓存，传 --force 可强制刷新。
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
HEADERS = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
KLINE_COLUMNS = [
    "date", "open", "close", "high", "low", "volume",
    "amount", "amplitude", "pct_chg", "change", "turnover",
]
REQUIRED = ["universe.csv", "index.csv", "calendar.csv", "prices.csv.gz"]


def secid(code: str) -> str:
    return ("1." if code[0] in "69" else "0.") + code


def request_json(url: str, params: dict, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url} {params}: {last}")


def fetch_universe() -> pd.DataFrame:
    rows = []
    for page in range(1, 5):
        payload = request_json(
            LIST_URL,
            {
                "pn": page,
                "pz": 100,
                "po": 1,
                "np": 1,
                "fltt": 2,
                "invt": 2,
                "fid": "f3",
                "fs": "b:BK0500",
                "fields": "f12,f14",
            },
        )
        data = payload.get("data") or {}
        diff = data.get("diff") or []
        if not diff:
            break
        rows.extend(diff)
        if len(rows) >= data.get("total", 0):
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows).rename(columns={"f12": "code", "f14": "name"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["secid"] = df["code"].map(secid)
    return df.drop_duplicates("code").sort_values("code").reset_index(drop=True)


def fetch_kline(secid_value: str, start: str, end: str, fqt: int = 2) -> pd.DataFrame:
    payload = request_json(
        KLINE_URL,
        {
            "secid": secid_value,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": 101,
            "fqt": fqt,
            "beg": start,
            "end": end,
        },
    )
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    return pd.DataFrame((line.split(",") for line in klines), columns=KLINE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--start", default="20180101")
    parser.add_argument("--end", default="20500101")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.force and all((out / name).exists() for name in REQUIRED):
        meta_path = out / "meta.json"
        cached = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if cached.get("start") == args.start and cached.get("end") == args.end:
            print(f"cache found in {out}, skip download (use --force to refresh)")
            return
        print(f"cache found in {out} but date range changed, redownloading")

    universe = fetch_universe()
    if len(universe) != 300:
        raise RuntimeError(f"expected 300 constituents, got {len(universe)}")
    print(f"universe: {len(universe)} stocks")

    index = fetch_kline("1.000300", args.start, args.end, fqt=0)
    index.insert(0, "code", "000300")
    index.insert(1, "name", "沪深300")
    calendar = pd.DataFrame({"trade_date": sorted(index["date"].unique())})
    print(f"index rows: {len(index)}, calendar days: {len(calendar)}")

    frames, failed = [], []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_kline, row["secid"], args.start, args.end, 2): row
            for row in universe.to_dict("records")
        }
        for n, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                df = fut.result()
                if df.empty:
                    failed.append(row["code"])
                else:
                    df.insert(0, "code", row["code"])
                    df.insert(1, "name", row["name"])
                    frames.append(df)
            except Exception as exc:  # noqa: BLE001
                failed.append(row["code"])
                print(f"fail {row['code']}: {exc}")
            if n % 50 == 0:
                print(f"fetched {n}/{len(futures)} in {time.time() - started:.1f}s")

    if failed:
        preview = ", ".join(sorted(failed)[:10])
        raise RuntimeError(
            f"{len(failed)}/{len(universe)} stocks failed ({preview}); "
            "no files written, retry later or use --force"
        )

    prices = pd.concat(frames, ignore_index=True)
    for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    prices = prices.sort_values(["code", "date"]).reset_index(drop=True)
    universe.to_csv(out / "universe.csv", index=False, encoding="utf-8-sig")
    index.to_csv(out / "index.csv", index=False, encoding="utf-8-sig")
    calendar.to_csv(out / "calendar.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(out / "prices.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

    meta = {
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "source": "eastmoney",
        "adjust": "hfq",
        "start": args.start,
        "end": args.end,
        "universe_size": int(len(universe)),
        "price_rows": int(len(prices)),
        "failed_codes": failed,
        "date_min": str(prices["date"].min()),
        "date_max": str(prices["date"].max()),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
