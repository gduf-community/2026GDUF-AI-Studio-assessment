# 沪深300 五日收益预测 Baseline

对沪深300全部 300 只成分股，基于截至 T 日收盘的公开行情，预测未来 5 个交易日的收益率，并按预测值排序。评估指标为 Rank IC 与 Top30 / Bottom30 多空收益。

这是一个可直接运行的参考实现，用于对照与起步，不构成投资建议。

## 数据

- 来源：东方财富公开行情接口，无需 token、无需付费账号。
- 范围：当前沪深300成分股 300 只（`data/universe.csv`），指数 000300 的交易日作为统一日历。
- 价格：**后复权**日线（`fqt=2`），2018-01-01 起。后复权可避免前复权在长期高分红股票上出现负价格。
- 仓库不附带数据文件，首次运行自行抓取（约 1–3 分钟，约 17 MB）。数据接口随上游调整可能变化；任一股票抓取失败时脚本会报错且不写文件，重试即可。

## 环境

Python 3.10+（在 3.12 上验证），依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 快速开始

以下命令均在仓库根目录执行。首次运行会下载数据并构建特征缓存 `data/dataset.pkl`。

```bash
# 1. 抓取数据；预测日之前必须冻结数据，避免用到未来信息
python src/fetch_data.py --out data --end 20261030

# 2. 训练（按周采样，使用训练期内全部可得样本）
python src/train.py --data data --models models

# 3. 生成预测，恰好 300 行
python src/predict.py --data data --models models --date 2026-10-30 --out submissions/result.csv

# 4. 测试窗口结束后，用出题方提供的真值打分
python src/score.py --pred submissions/result.csv --truth /path/to/truth.csv
```

真值文件由出题方在测试窗口结束后生成，列为 `stock_code,actual_return`，覆盖 300 只股票。`--end` 只影响抓取区间；若先抓了完整数据再指定 `--date`，特征计算仍可能用到 `--date` 之后的价格，因此**必须在 T 日收盘后抓取并冻结，最迟在 T+1 开盘前完成**。若等到 T+1 收盘后再抓，就会把 T+1 的收益带进特征。

其他脚本：

```bash
# 5 日反转对照（pred_return = -ret_5），用于检验模型是否只是反转因子
python src/naive.py --data data --date 2026-10-30 --out submissions/naive_result.csv

# 年度滚动回测，输出 reports/backtest_daily.csv 与 reports/backtest_metrics.json
python src/backtest.py --data data --out reports
```

## 全流程检查清单

从空目录到提交文件，按顺序执行。每一步都给出通过标准，便于定位问题（时间以普通笔记本为参考，首次抓取约 1–3 分钟，特征构建、训练、预测各几秒，回测需训练 8 个年度模型、通常几分钟）。

1. **环境**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   通过标准：`python -c "import pandas, lightgbm, sklearn, scipy"` 无报错。

2. **冻结并抓取数据**

   ```bash
   python src/fetch_data.py --out data --start 20180101 --end <T>
   ```

   通过标准：`data/meta.json` 中 `failed_codes` 为空，`date_max` 为 T 或 T 之前最近交易日；`universe.csv` 恰好 300 行。任一只股票抓取失败时脚本会直接报错且不写文件，重试即可。**必须在 T 日收盘后抓取，最迟在 T+1 开盘前完成**；用截止日之后的数据抓取会把未来价格带进特征。

3. **构建特征**（训练和预测会自动触发，也可单独检查）

   ```bash
   python -c "import sys; sys.path.insert(0,'src'); import features as F; print(F.build_dataset('data').shape)"
   ```

   通过标准：输出约 64 万行 × 45 列，并生成 `data/dataset.pkl`。原始文件更新后缓存会自动失效重建。

4. **训练**

   ```bash
   python src/train.py --data data --models models
   ```

   通过标准：`models/meta.json` 的 `train_end` 比 T 早至少 5 个交易日（末尾 5 日标签未知，会被自动剔除）；`training_rows` 约 10 万以上；`n_features` 为 35。

5. **预测**

   ```bash
   python src/predict.py --data data --models models --date <T> --out submissions/result.csv
   ```

   通过标准：恰好 300 行、代码无重复、无空值；`pred_return` 均值接近 0，标准差约 1%–3%。行数不足通常意味着数据快照缺股票，或 `<T>` 不是交易日。

6. **自检**

   ```bash
   python src/naive.py --data data --date <T> --out submissions/naive_result.csv
   python src/backtest.py --data data --out reports
   ```

   通过标准：naive 生成 300 行；回测打印 LightGBM / 反转 / Ridge 三组指标。单一年份为负是正常噪声，判断模型应看多年均值与 IC>0 占比。

7. **打分**

   ```bash
   python src/score.py --pred submissions/result.csv --truth <truth.csv>
   ```

   通过标准：`stocks=300`，输出 Rank IC、多空收益、MAE 与方向准确率。

## 目录

```
src/
  fetch_data.py   抓取成分股、指数、日线，生成 data/ 快照
  features.py     特征工程、标签、数据集构建（25 个技术特征 + 10 个横截面排名特征，共 35 维；另计算 6 个市场级特征但未入模）
  train.py        LightGBM 训练
  predict.py      指定交易日预测，输出 300 行提交文件
  naive.py        5 日反转对照
  backtest.py     年度扩展窗口滚动回测
  score.py        Rank IC / 多空 / MAE / 方向准确率
data/             数据快照与特征缓存（不入库）
models/           模型文件（不入库）
submissions/      提交文件（不入库）
reports/          回测指标与明细（明细不入库）
```

## 方法

- **标签**：`fwd_ret_5 = close(t+5) / close(t) - 1`；训练时缩尾到 ±30%，评估时使用原始收益。
- **样本**：每个自然周最后一个交易日取一个横截面，降低 5 日标签重叠带来的自相关；只保留上市满 120 个交易日的样本。
- **特征**：动量（1/5/20/60/120 日）、波动率、均线偏离、RSI、区间位置、回撤、52 周高低点距离、量比、成交额比、换手率、60 日 Beta 与相关性、偏度峰度，以及上述部分特征的**当日横截面百分位排名**（`cs_*`）。市场级特征（指数收益、宽度、离散度）经测试会降低 IC，未纳入最终模型。
- **模型**：LightGBM 回归，500 棵树、学习率 0.03、31 叶、`min_child_samples=50`，固定随机种子并开启确定性训练。
- **缺失值**：预测时用当日横截面中位数填充，再以 0 兜底。
- **对照**：Ridge 回归（标准化 + 中位数填充）与 5 日反转，用于判断收益是否仅来自反转效应。

## 回测

年度扩展窗口：每年用此前全部数据训练，预测当年每个周末横截面，样本外滚动前进。为避免 5 日标签跨年泄漏，训练集在年末剔除最后 5 个交易日（`PURGE_DAYS=5`）。回测覆盖 2019–2026 共 400 个横截面。下表报告 2022–2026 的成熟窗口（245 个横截面），也是设定考核线的参考区间：

| 模型 | Rank IC | IC 标准差 | ICIR | IC>0 占比 | Top30-Bottom30（5日） | 多空>0 占比 | MAE | 方向准确率 |
|---|---|---|---|---|---|---|---|---|
| LightGBM | 0.0259 | 0.1618 | 0.160 | 51.4% | 0.77% | 59.6% | 3.61% | 49.6% |
| 5 日反转 | 0.0074 | 0.1945 | 0.038 | 51.0% | -0.10% | 48.2% | 5.11% | 49.2% |
| Ridge | -0.0030 | 0.2162 | -0.014 | 46.9% | 0.63% | 55.1% | 3.61% | 48.8% |

全区间 2019–2026（400 个横截面）上，LightGBM 的平均 Rank IC 为 0.0165、平均多空收益 0.57%；早期年份（2019）IC 为负（-0.046），模型表现随训练样本增加而改善，因此不建议用单一年份或单个窗口评价。

回测指标的含义与局限：

- **单窗口噪声很大**。300 只股票的 Rank IC 在单周的标准误约为 0.058，周度 IC 从 -0.23 到 +0.32（5%–95% 分位）都属常见波动，一周的结果不足以证明模型有效。
- **MAE 与方向准确率不适合作为主指标**。5 日收益本身接近零均值、高噪声，回归绝对值的误差和涨跌方向几乎不可预测；两者主要用于排查提交文件是否异常。
- **排序指标更稳健**。Rank IC 衡量预测排序与真实排序的一致性，多空收益衡量 Top30 相对 Bottom30 的实际差距，二者都只依赖横截面排序。
- **幸存者偏差**。股票池为当前成分股，历史上被调出的股票不在样本内，回测结果偏乐观。
- **交易成本未计**。多空收益为毛收益，未扣除手续费、印花税、冲击成本与融券成本。

## 提交格式

`submissions/result.csv`，UTF-8（带 BOM 或不带均可），两列：

```csv
stock_code,pred_return
000001,0.002865
000002,-0.000839
```

要求：

- 恰好 300 行，覆盖股票池全部成分股，不重复、不缺失；
- `stock_code` 为 6 位字符串，保留前导零；
- `pred_return` 为浮点数，单位与 `close(t+5)/close(t)-1` 一致；
- 只对排序敏感，数值尺度不影响 Rank IC，但会影响 MAE，建议输出真实收益量级而非任意分数。

## 复现性

在相同数据快照与依赖版本下，训练与预测结果逐字节一致（模型文件 MD5 稳定）。数据快照会随抓取日期变化，复现实验时请固定 `--end` 与 `--date`。若接口返回的复权因子随分红调整而变化，历史价格可能出现细微差异，这是数据源特性，不是代码随机性。
