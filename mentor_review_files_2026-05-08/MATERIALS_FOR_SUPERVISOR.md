# 导师查看材料说明

本文件夹整理的是毕业设计项目中除论文正文、答辩 PPT、学校模板和中期表格之外，建议提交给导师查看的工程与实验材料。

## 建议导师优先查看

1. `README.md`
   - 项目功能、环境、数据准备、训练、评估和复现命令。

2. `src/`
   - 核心源码。
   - `src/models/agstt.py`：AG-STT 主模型。
   - `src/models/adaptive_graph.py`：自适应图模块。
   - `src/models/baselines.py`：对比基线模型。
   - `src/data/`：数据集、图结构、归一化处理。
   - `src/utils/`：指标、随机种子、IO 工具。

3. `train.py` 和 `evaluate.py`
   - 训练入口和评估入口。

4. `configs/`
   - 主实验、基线、消融、跨划分、随机种子稳健性和公平预算实验配置。

5. `scripts/`
   - 数据预处理、批量实验、显著性检验、结果汇总和可视化分析脚本。
   - 已剔除论文/PPT 生成类脚本。

6. `data/`
   - `data/*.npz`、`data/*_adj.npy`：已处理数据和邻接矩阵。
   - `data/raw/`：原始数据文件，便于导师核查数据来源与处理流程。

7. `outputs/`
   - 实验指标、预测结果、显著性检验、稳健性实验、消融实验、跨划分实验和图片结果。
   - 已剔除 `outputs/ppt_validation/`，因为它只服务于答辩 PPT 校验。

8. `checkpoints/`
   - 已训练模型权重，可直接用于 `evaluate.py` 复核结果。

9. `docs/experiment_summary_2026-04-28/`
   - 面向实验复核的汇总文档、表格、图和 JSON 证据。

## 快速复核命令

安装依赖：

```bash
pip install -r requirements.txt
```

使用已有检查点评估：

```bash
python evaluate.py --config configs/metr_la_agstt.yaml --checkpoint checkpoints/metr_la_metr_la_agstt_full_best.pt
```

查看主实验汇总：

```bash
type outputs\current_results_summary.md
type outputs\pemsd7\benchmark_summary.md
type outputs\metr_la\benchmark_summary.md
```

## 本次未放入的材料

- `paper/`：论文正文、论文图片和论文版式资产。
- `finalReportTemplateLaTeX/`、`finalReportTemplateWord(1).docx`：学校论文模板。
- 根目录和 `docs/` 下的答辩 PPT 文件。
- 中期检查、教师表格等过程性 Word 文档。
- 旧的 `thesis_required_files_2026-04-27/`、`thesis_required_files_2026-04-28/` 及其 zip，避免重复和混入论文/PPT。
- `__pycache__/`、`.pyc` 等 Python 缓存文件。

