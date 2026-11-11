"""主办方工具：加载提交镜像、运行容器、调用评分脚本。

示例：
    python tools/run_container.py --tar submissions/team01.tar \
        --data data --output output/team01 --truth .private/truth.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], timeout: int | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", default=None, help="docker save 导出的 tar；不传则直接使用已有镜像")
    parser.add_argument("--image", default="hs300-top5:latest")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="output")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--timeout", type=int, default=8 * 3600)
    args = parser.parse_args()

    if args.tar:
        run(["docker", "load", "-i", str(Path(args.tar).resolve())])

    data_dir = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.csv"
    if result_path.exists():
        result_path.unlink()

    run(
        [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{data_dir}:/app/data:ro",
            "-v", f"{output_dir}:/app/output",
            "-e", "DATA_DIR=/app/data",
            "-e", "MODEL_DIR=/app/models",
            "-e", "OUTPUT_DIR=/app/output",
            args.image,
        ],
        timeout=args.timeout,
    )

    if not result_path.exists():
        raise SystemExit(f"容器未生成 {result_path}")
    run(
        [
            sys.executable, "src/score.py",
            "--pred", str(result_path),
            "--truth", str(Path(args.truth).resolve()),
        ]
    )


if __name__ == "__main__":
    main()
