# ✈️ NGAFID Maintenance Event Detection (2-Days Binary Benchmark)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Sktime](https://img.shields.io/badge/sktime-latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

面向技术读者的复现与工程化说明：在 NGAFID 2days 子集上进行多变量时间序列二分类，目标是识别 **维护前（潜在异常）** 与 **维护后（健康）** 状态。

## TL;DR
- 任务：4096 时间步、23 通道的飞机传感器序列二分类。
- 关键改进：严格避免归一化泄露（fold 内拟合统计量）+ 时间轴线性插值。
- 模型：MiniRocketMultivariate + RidgeClassifierCV，CPU 即可完成训练。
- 输出：自动生成混淆矩阵、ROC 曲线、各折指标对比图到 `results/`。

## 目录
- [1. 背景与问题定义](#1-背景与问题定义)
- [2. 方法设计与防泄露策略](#2-方法设计与防泄露策略)
- [3. 端到端流程](#3-端到端流程)
- [4. 快速开始](#4-快速开始)
- [5. 可复现性说明](#5-可复现性说明)
- [6. 结果产物与解读建议](#6-结果产物与解读建议)
- [7. 常见问题与排障](#7-常见问题与排障)
- [8. 项目结构](#8-项目结构)
- [9. 实测设备与输出效果总结](#9-实测设备与输出效果总结)

## 1. 背景与问题定义
本项目复现并优化论文 *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID* 的二分类任务。

分类目标：
- `1` = Before Maintenance（维护前，潜在故障/异常）
- `0` = After Maintenance（维护后，健康）

工程上最容易踩坑的是：
- 预处理泄露：用全量数据统计量做标准化会抬高离线指标。
- 缺失值处理不当：直接填 0 会破坏时序形态。

## 2. 方法设计与防泄露策略
核心策略如下：

1. Fold 级隔离
- 每一折单独实例化预处理器。
- 均值/方差只在训练子集拟合，再用于验证子集变换。

2. 时间轴插值
- 对每条样本、每个特征按时间轴做线性插值。
- 首尾连续缺失使用 `limit_direction="both"`，极端全缺失再用 0 兜底。

3. 轻量高效特征提取
- 特征提取器：`MiniRocketMultivariate`
- 分类器：`RidgeClassifierCV`

## 3. 端到端流程
```text
[Raw Aviation Data] (Parquet/Dask)
       |
       v
[Data Pipeline] -> Auto-download & Subsetting (2-Days Benchmark)
       |
       v
[5-Fold CV Engine] (Strict Isolation)
       |
       +-- Fold k --> [FoldPreprocessor]
       |               |- Linear interpolation along time axis
       |               `- Z-score fit on train only, transform on val
       v
[MiniRocketMultivariate] -> [RidgeClassifierCV]
       v
[Metrics + Plots] -> Accuracy / Weighted F1 / ROC-AUC
```

## 4. 快速开始

### 4.1 克隆与环境准备
```bash
git clone https://github.com/RUIKER/TEST2.git
cd TEST2
python -m venv .venv
```

Windows PowerShell:
```powershell
# 若激活脚本被策略拦截，可先执行：
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 一条命令运行
```bash
python main.py
```

运行时会自动：
- 检查并下载 `data/subset_data/2days`
- 执行交叉验证训练与评估
- 输出图表到 `results/`

## 5. 可复现性说明
- 随机种子：`42`
- 默认序列长度：`4096`（可用环境变量 `PM_MAX_LENGTH` 覆盖）
- 交叉验证：优先使用头表中的 `fold` 列；缺失时回退到 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

示例（缩短长度做快速实验）：
```bash
# PowerShell
$env:PM_MAX_LENGTH=2048
python main.py
```

```bash
# bash
PM_MAX_LENGTH=2048 python main.py
```

## 6. 结果产物与解读建议
程序默认会在 `results/` 生成：
- `confusion_matrix.png`
- `roc_curve.png`
- `fold_metrics_comparison.png`

## 7. 常见问题与排障
1. 下载失败（Zenodo/Drive 超时或不可达）
- 可从以下来源手动下载并放到 `data/subset_data/`：
  - https://doi.org/10.5281/zenodo.6624956
  - https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid

2. PowerShell 无法激活虚拟环境
- 先执行：`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

3. 数据路径报错
- 确认目录下存在：`flight_data.pkl`、`flight_header.csv`、`stats.csv`

4. 首折训练明显更慢
- 属于正常现象：MiniRocket/Numba 首次会触发编译缓存。

## 8. 项目结构
```text
.
├── data/
│   ├── subset_data/
│   │   └── 2days/
│   └── ngafiddataset/
├── results/
├── src/
│   ├── data_downloader.py
│   ├── data_preprocessor.py
│   └── train_evaluate.py
├── main.py
└── requirements.txt
```

## 9. 实测设备与输出效果总结

### 9.1 实验设备与运行环境
- 操作系统：Windows 10
- CPU：Intel Core i5-12450H
- 内存：16 GB
- Python：3.11
- 运行命令：`python main.py`

### 9.2 数据规模与任务设置
- 数据集：NGAFID `2days`
- 输入张量：`X=(11446, 4096, 23)`
- 标签张量：`y=(11446,)`
- 标签分布：`[5844, 5602]`（类别相对均衡）
- 评估策略：5 折交叉验证

### 9.3 关键性能结果

| 指标 | 结果（mean ± std） |
|---|---:|
| Accuracy | 0.7215 ± 0.0123 |
| Weighted F1 | 0.7215 ± 0.0123 |
| ROC-AUC | 0.7860 ± 0.0109 |

### 9.4 各折耗时与指标

| Fold | 预处理耗时(s) | 特征提取耗时(s) | 单折总耗时(s) | Accuracy | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 52.72 | 720.08 | 792.17 | 0.7183 | 0.7184 | 0.7883 |
| 2 | 66.41 | 744.43 | 815.71 | 0.7077 | 0.7078 | 0.7688 |
| 3 | 46.98 | 685.60 | 758.10 | 0.7117 | 0.7117 | 0.7791 |
| 4 | 49.60 | 678.95 | 749.64 | 0.7274 | 0.7272 | 0.7946 |
| 5 | 57.37 | 700.58 | 771.74 | 0.7422 | 0.7423 | 0.7990 |

总耗时：`3887.38s`（约 `64.79` 分钟，约 `1.08` 小时）。

### 9.5 输出效果总结
1. 在 i5-12450H + 16GB 的消费级笔记本环境下，完整 5 折流程可以稳定跑通，且无需 GPU。
2. 训练时长主要集中在 MiniRocket 特征提取阶段（每折约 679-744 秒），预处理开销相对较小（每折约 47-66 秒）。
3. 模型在该数据设置上取得稳定的中高区间区分能力（ROC-AUC 约 0.786），说明其对维护前后状态具有可用的排序判别能力。
4. 五折结果波动较小（Accuracy/F1 标准差约 0.012），表明流程在当前数据规模与硬件环境下具备较好的重复性。