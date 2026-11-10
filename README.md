# 沪深300 五日收益预测 Baseline（Top5 组合）

赛题形式对标大数据挑战赛：主办方提供截至 T 日的沪深300行情快照，参赛者在 T 日收盘后从 300 只成分股中最多选出 5 只、分配合计 100% 的权重，持有到 T+5。评分按组合收益计算，收益越高越好。

本仓库是一个可直接运行的参考实现：LightGBM 预测未来 5 日收益，取预测最高的 5 只等权持有，并提供 5 日反转与随机选股两组对照。代码仅用于教学与对照，不构成投资建议。

## 任务口径

- **决策时点**：T 日收盘后，只能使用截至 T 日收盘的数据。
- **收益口径**：T+1 开盘买入，T+5 开盘卖出，个股收益为 `open(T+5) / open(T+1) - 1`。
- **提交内容**：最多 5 只股票，每只权重非负，权重合计为 1。
- **评分**：`组合收益 = Σ 权重 × 个股收益`，同时给出组合相对全池等权平均的超额收益。单期结果波动很大，评定以多期滚动回测为主。

## 数据快照

主办方提供 `data/`：

| 文件 | 内容 |
| --- | --- |
| `train.csv.gz` | 日线行情，`stock_code,date,open,high,low,close,volume,amount,turnover`，后复权 |
| `universe.csv` | 股票池，`stock_code,stock_name`，300 只 |
| `index.csv` | 沪深300指数日线，`date,open,high,low,close,volume,amount` |
| `calendar.csv` | 交易日列表，`trade_date` |
| `meta.json` | 数据来源、复权方式、截止日、股票数与行数 |

快照已冻结在 T 日收盘，T 之后的价格不在文件内。`open` 为复权后价格，停牌日不产生记录，特征中的价格按前值填充，成交量、成交额、换手率记 0。

## 环境

Python 3.12（本地与镜像一致）：

```bash
pip install -r requirements.txt
```

## 本地运行

以下命令均在仓库根目录执行，`T=2026-10-30` 为示例：

```bash
# 1. 训练：按周采样，标签缩尾 ±30%
python src/train.py --data data --models models

# 2. 预测：输出恰好 5 行、权重合计为 1
python src/predict.py --data data --models models --date 2026-10-30 --out output/result.csv

# 3. 对照：过去 5 日跌幅最大的 5 只，等权
python src/naive.py --data data --date 2026-10-30 --out output/naive_result.csv

# 4. 自评：需要测试窗口结束后生成的真值
python src/score.py --pred output/result.csv --truth /path/to/truth.csv

# 5. 年度滚动回测
python src/backtest.py --data data --out reports
```

首次运行会构建特征缓存 `data/dataset.pkl`（约 64 万行 × 45 列）。若数据目录只读，缓存写入失败会自动跳过，不影响运行。预测前会校验模型 `meta.json` 的训练截止日必须早于预测日，防止误用已看到未来价格的模型。

真值文件由主办方在测试窗口结束后生成，列为 `stock_code,actual_return`，覆盖全部 300 只股票。

## Docker 提交

提交物是一个镜像 tar。镜像内只包含代码与依赖，`run.sh` 在评测时先训练再预测：

```bash
# 构建（国内网络可追加镜像与 trusted-host）
docker build -t hs300-top5:latest .
docker build -t hs300-top5:latest . \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg "PIP_EXTRA=--trusted-host pypi.tuna.tsinghua.edu.cn"

# 本地验证：数据只读挂载，输出写到 output/
docker run --rm --network none \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/output:/app/output" \
  -e DATA_DIR=/app/data -e MODEL_DIR=/app/models -e OUTPUT_DIR=/app/output \
  hs300-top5:latest

# 导出提交文件
docker save hs300-top5:latest -o team01.tar
```

容器约定：

- 数据挂载在 `/app/data`（只读），结果写到 `/app/output/result.csv`；
- 模型文件写在容器内，无需持久化；
- 评测时无网络，训练与预测不得依赖外部接口；
- 参考时限 8 小时；本基线在普通笔记本上训练约 20 秒、预测约 5 秒。

主办方批量评测：

```bash
python tools/run_container.py --tar team01.tar \
  --data data --output output/team01 --truth .private/truth.csv
```

脚本会加载镜像、以无网络方式运行容器、检查 `result.csv` 并调用 `src/score.py`。

## 提交格式

`result.csv`，UTF-8，两列，1–5 行：

```csv
stock_id,weight
300433,0.2
002558,0.2
301236,0.2
002625,0.2
600460,0.2
```

约束：

- `stock_id` 为 6 位字符串，保留前导零，不得重复；
- `weight` 非负，合计为 1。

## 方法

- **标签**：`open(T+5) / open(T+1) - 1`，与评分口径一致；训练时缩尾到 ±30%。
- **样本**：每个自然周最后一个交易日取一个横截面，避免 5 日标签重叠带来的自相关；只保留上市满 120 个交易日、标签可得的样本。
- **特征**：35 维。动量（1/5/10/20/60/120 日）、波动率、均线偏离、RSI、区间位置、回撤、52 周高低点距离、量比、成交额比、换手率、60 日 Beta 与相关性、偏度峰度，以及 10 个当日横截面百分位排名（`cs_*`）。市场级特征经测试会降低效果，未入模。
- **模型**：LightGBM 回归，500 棵树、学习率 0.03、31 叶、`min_child_samples=50`，固定随机种子 42 并开启确定性训练。
- **组合**：预测分数最高的 5 只等权，权重各 0.2；并列时按代码升序打破。
- **缺失值**：预测时用当日横截面中位数填充，再以 0 兜底。

## 回测

年度扩展窗口：每年只用此前数据训练，预测当年每个周末横截面，样本外滚动前进。为避免标签跨年泄漏，训练集在年末剔除最后 5 个交易日。2019–2026 共 400 个横截面，组合收益按 T+1 开盘到 T+5 开盘计算：

| 模型 | 平均组合收益 | 全池平均 | 平均超额 | 正收益周占比 | 跑赢全池占比 |
|---|---|---|---|---|---|
| LightGBM Top5 | 0.82% | 0.19% | 0.64% | 57.5% | 55.5% |
| 5 日反转 Top5 | 0.48% | 0.19% | 0.30% | 53.8% | 49.3% |
| 随机 Top5 | 0.34% | 0.19% | 0.15% | 52.3% | 49.8% |

分年度超额（LightGBM）：2019 +0.35%、2020 +0.96%、2021 +0.66%、2022 +1.42%、2023 +0.91%、2024 +0.08%、2025 +0.29%、2026 +0.39%，每年均为正，但 2024 年接近零。

需要注意：

- **单期噪声极大**。5 只股票的等权组合，一周结果主要由个股事件驱动，不能据此判断模型优劣。
- **幸存者偏差**。股票池为当前成分股，历史上被调出的股票不在样本内，回测偏乐观。
- **未计交易成本**。回测为毛收益，未扣佣金、印花税与冲击成本。
- **数据快照固定**。快照截止日之后的市场变化不在数据内；换快照后需要重新训练与回测。

## 目录

```
src/
  features.py     数据加载、特征工程、标签、数据集构建
  train.py        LightGBM 训练
  predict.py      指定交易日选 Top5，输出 result.csv
  naive.py        5 日反转对照
  backtest.py     年度滚动回测（LightGBM / 反转 / 随机）
  score.py        组合收益评分与提交格式校验
tools/
  run_container.py  主办方批量运行与评分
data/             行情快照与特征缓存（不入库）
models/           模型文件（不入库）
output/           预测结果（不入库）
reports/          回测指标与明细（明细不入库）
Dockerfile        提交镜像
run.sh            容器入口：训练 + 预测
```

## 复现性

固定数据快照与依赖版本后，训练与预测结果逐字节一致（模型文件 MD5 稳定）。随机种子为 42，LightGBM 开启 `deterministic`。回测的随机对照使用固定种子的 `numpy.random.default_rng(42)`。
