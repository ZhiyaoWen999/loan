# 贷前风控

本仓库用于模拟贷前（审批前）风控流程：生成模拟数据、定义标签、做 vintage 校验、构建特征、EDA/筛选/分箱、训练 WOE 评分卡。

## 覆盖内容
- 模拟表：users、applications、bureau、devices、behavior_agg、loans、loan_outcomes、loan_dpd_snapshots。
- 标签定义：`y_dpd30_ever`（放款后 90 天内是否出现 DPD30+），并做成熟期过滤。
- Vintage：90D 点位 cohort 以及 cohort×MOB 的经典 vintage（基于 DPD 快照）。
- 标签观察窗对比：30/60/90/180 天坏率（按 cohort 与 score decile）。
- 特征构建：按时间切分 train/valid，生成时间特征并做泄露规避。
- EDA & 筛选：缺失、PSI、IV/AUC、低方差、偏态、相关性。
- 分箱与评分卡：WOE 分箱、逐步回归、打分与分位表现。

## 目录结构
- `data/`：模拟数据与输出结果。
- `data/eda/`：EDA 汇总与筛选标记。
- `data/binning/`：单变量分箱结果。
- `data/scorecard/`：WOE、points、打分与分位表。
- `data/figures/`：seaborn 图（vintage、观察窗对比）。
- `scripts/`：流程脚本。

## 最小流程（最快跑通）
```bash
python scripts/generate_mock_data.py --outdir data
python scripts/build_model_table.py --datadir data --maturity-days 90
python scripts/build_features_90d.py --infile data/model_train_90d.csv --outdir data
python scripts/scorecard_pipeline.py --train data/features_train_90d.csv --valid data/features_valid_90d.csv --outdir data/scorecard --bins 10 --iv-threshold 0.02 --stepwise --p-enter 0.05 --p-remove 0.1
```

## 运行批次与 manifest
- 任意脚本可加 `--run-dir auto`（或指定目录），输出会写入 `data/run_YYYYMMDD_HHMMSS/`，并在该目录下生成 `manifest.csv` 记录产出路径与步骤。
- 示例（先设批次目录，再跑最小流程并记录产出）：
```bash
RUN_DIR=data/run_$(date +%Y%m%d_%H%M%S)
python scripts/generate_mock_data.py --run-dir "$RUN_DIR"
python scripts/build_model_table.py --run-dir "$RUN_DIR" --datadir "$RUN_DIR"
python scripts/build_features_90d.py --run-dir "$RUN_DIR" --infile "$RUN_DIR/model_train_90d.csv"
python scripts/scorecard_pipeline.py --run-dir "$RUN_DIR" --train "$RUN_DIR/features_train_90d.csv" --valid "$RUN_DIR/features_valid_90d.csv" --bins 10 --iv-threshold 0.02 --stepwise
python scripts/scorecard_business_view.py --run-dir "$RUN_DIR"
```
> 将同一批次的 `--run-dir` 设为同一目录即可把产出集中放置，`manifest.csv` 会累计记录输出。

## 全流程（含校验与图表）
1. 生成模拟数据
```bash
python scripts/generate_mock_data.py --outdir data
```

2. 构建带标签的建模表（90D 成熟期）
```bash
python scripts/build_model_table.py --datadir data --maturity-days 90
```

3. Vintage 校验（90D point-in-time + MOB）
```bash
python scripts/vintage_90d.py --infile data/model_train_90d.csv --outdir data
python scripts/vintage_mob.py --snapshots data/loan_dpd_snapshots.csv --applications data/applications.csv --out data/vintage_mob_dpd30.csv
python scripts/plot_vintage_seaborn.py --infile data/vintage_mob_dpd30.csv --outdir data/figures
```

4. 标签观察窗对比（30/60/90/180 天）
```bash
python scripts/compare_label_windows.py --infile data/loan_outcomes.csv --outdir data
python scripts/plot_label_window_comparison.py --outdir data/figures
```

5. 构建特征（时间切分）
```bash
python scripts/build_features_90d.py --infile data/model_train_90d.csv --outdir data
```

6. EDA、筛选与分箱
```bash
python scripts/eda_report.py --train data/features_train_90d.csv --valid data/features_valid_90d.csv --outdir data/eda
python scripts/feature_screening.py --train data/features_train_90d.csv --outdir data/eda
python scripts/binning_univariate.py --train data/features_train_90d.csv --outdir data/binning --bins 10
```

7. 评分卡与业务视图
```bash
python scripts/scorecard_pipeline.py --train data/features_train_90d.csv --valid data/features_valid_90d.csv --outdir data/scorecard --bins 10 --iv-threshold 0.02 --stepwise --p-enter 0.05 --p-remove 0.1
python scripts/scorecard_business_view.py --out data/scorecard/scorecard_business_view.csv
```

## 关键产出
- `data/model_dataset.csv`, `data/model_train_90d.csv`：建模表与成熟期样本。
- `data/vintage_90d_m*.csv`, `data/vintage_mob_dpd30*.csv`：vintage 表。
- `data/label_window_bad_rates_*.csv`：观察窗对比表。
- `data/features_train_90d.csv`, `data/features_valid_90d.csv`：特征表。
- `data/scorecard/scorecard_points.csv`：评分卡分数与系数。
- `data/scorecard/scorecard_business_view.csv`：业务视角的 WOE/分数表。

## 备注
- 各脚本默认使用 `data/` 路径，可用参数覆盖。
- 需要集中管理产出时，使用 `--run-dir`，manifest 会写到该目录。
- 图表使用 seaborn/matplotlib，输出到 `data/figures/`。
