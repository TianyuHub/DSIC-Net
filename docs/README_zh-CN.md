# DSIC-Net 中文使用说明

本仓库依据随项目提供的 Methods 整理为论文配套代码。模型为两阶段级联结构：

1. Image Space Net 从 3D T1 加权 sMRI 估计初始脑龄；
2. Bounded Graph-Guided Residual Correction Net 先计算图像条件残差，再使用
   R2SN 图分支给出有界校正。

最终预测严格对应 Methods 公式：

final_age = stage1_age + image_residual + κ tanh(graph_response)。

因此图分支校正始终满足 |graph_correction| < κ。

## 快速安装

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
~~~

如使用 NVIDIA GPU，请先按照 PyTorch 官网选择与本机 CUDA 驱动匹配的安装命令。

## 跑通示例

~~~bash
python scripts/make_demo_data.py --overwrite
python scripts/validate_data.py --manifest data/demo/manifest.csv
python scripts/run_demo.py --manifest data/demo/manifest.csv --device cpu
~~~

运行结果写入 outputs/demo_predictions.csv。示例采用随机初始化权重，仅用于验证
数据读取、Image Space Net、R2SN 图编码、有界校正和结果导出的完整链路，不代表
任何实验性能。

## 两阶段训练

~~~bash
python scripts/train_stage1.py \
  --manifest data/demo/manifest.csv \
  --epochs 1 \
  --device cpu \
  --output outputs/stage1.pt

python scripts/train_stage2.py \
  --manifest data/demo/manifest.csv \
  --stage1-checkpoint outputs/stage1.pt \
  --epochs 1 \
  --device cpu \
  --output outputs/dsic_net.pt
~~~

预测：

~~~bash
python scripts/predict.py \
  --manifest data/demo/manifest.csv \
  --checkpoint outputs/dsic_net.pt \
  --split test \
  --device cpu \
  --output outputs/predictions.csv
~~~

## 数据清单

CSV 必须包含以下列：

| 列名 | 含义 |
| --- | --- |
| subject_id | 唯一受试者 ID |
| image_path | 相对 CSV 所在目录的 3D NIfTI 路径 |
| graph_path | 相对 CSV 所在目录的 R2SN NPZ 路径 |
| age | 年龄（年） |
| sex | 0 或 1，仅用于队列审计，当前模型不使用 |
| split | train、val 或 test |

NPZ 必须包含 node_features（N×d）和 adjacency（N×N）。所有受试者的 N 与 d
必须一致，adjacency 必须对称。模型内部会自动添加自环。

真实数据训练前请执行：

~~~bash
python scripts/validate_data.py --manifest /path/to/manifest.csv
~~~

## 与原压缩包相比的主要修正

- 删除与 DSIC-Net 无关的 GAN 工程、缓存、重复脚本和机器绝对路径；
- 模型文件及类名与 Methods 中的方法名称逐项对应；
- LSAFB 使用局部卷积、频域门控和 D/H/W 三轴注意力后残差融合；
- Image Space Net 使用两组 LSAFB，每组重复两次；
- R2SN 编码使用两层 GCN、图级最大池化、LayerNorm 和标量投影；
- 图校正由单一标量 κ tanh(·) 给出，修复旧实现的双输出和错误尺度；
- 旧包中 6 个 NIfTI 文件均为截断文件，改为可重复生成的合法合成示例；
- 旧 XLS/XLSX 与随包数据不匹配，示例改为路径、标签、划分都明确的 CSV；
- 增加 pytest、GitHub Actions、数据校验、训练、推理和项目打包元数据。

## 科研使用注意

随附 Methods 未包含 Section 2.2.2 的 R2SN 构建细节，也未给出最终训练超参数。
因此本仓库读取预先计算好的 R2SN，训练损失和优化器仅为可运行默认值。论文发表前
应补充最终 R2SN 构建流程、队列信息、预处理、超参数、随机种子、引用和许可证。
