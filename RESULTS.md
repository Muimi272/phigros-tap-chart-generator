# Phigros 踩音到谱面：训练与评估结果

**English summary:** The local RTX 5060 experiment completed its engineering
pipeline, but failed chart-quality gates. Four-fold x-MAE is `0.14860` versus
`0.14188` for the equidistant baseline; side macro-F1 is `0.35055` versus the
weighted majority baseline `0.47846`. No quality claim is made. The full
English report is [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md).

> 状态：**工程闭环通过，生成质量未达标**。
> 本文档记录 2026-11-20 在本机 RTX 5060 Laptop 8 GB 上的真实训练结果。
> 不得把以下 checkpoint 或样例描述为“已训练成功”或“可用于正式谱面制作”。

## 1. 结论摘要

- 数据：24 份谱面、12 首唯一歌曲、20,280 个音符、15,625 个同押组。
- 模型：`PhiTapGroupMLP`，646,148 参数，组级 center/spread 回归 + 逐音符 side 分类。
- 四折分组交叉验证（按歌曲分组，冻结 splits）：
  - 横坐标 `x_MAE = 0.14860`；
  - 最强非学习基线 `equidistant = 0.14188`；
  - 模型比基线**差 4.73%**，未达到“至少改善 10%”的验收门槛。
  - side `macro-F1 = 0.35055`，四折加权多数类基线 `0.47846`，未达到“至少高 10%”的门槛。
- 格式与保真：编译器 dummy 验证 24/24 合法；4 份留出样例 semantic/parser 均通过；输入时间/类型/Hold 保真核验 4/4 通过；越界率 0，非预期同 tick 重复率 0。
- 样例存在明显分布退化：预测横坐标集中在屏幕中部，4 份样例的 below 音符均为 0，组内平均宽度偏大。未做真人上机游玩验证。

## 2. 环境与数据

- Windows 10.0.26200，Python 3.11.9；
- NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，`sm_120`；
- PyTorch 2.7.1+cu128，CUDA 12.8；
- 训练只使用 `.venv\Scripts\python.exe`，未使用系统 Python 安装或训练；
- 24 份谱面全部保留，未丢弃音频或谱面；
- 类型分布：tap 10,262、drag 6,898、hold 1,828、flick 1,292；
- 方向分布：above 18,621、below 1,659（8.2%）；
- `x_abs` 范围 `[0.0551, 0.9635]`，越界 0；
- v1 5 份、v3 19 份；v1 move 事件按 `x=(v//1000)/880`、`y=(v%1000)/520` 解码；
- 冻结切分 `data/processed/splits.json` 的 SHA256：
  `772fd2eac22a32581191f5da366bce178f5319354c05fb674cc0b0c3db2e7aa0`。

### 数据规模估算（规划值，不是保证）

本次只有 12 首独立歌曲，无法从一次实验估计真实学习曲线。建议先新增至少
48 首、达到 60 首做 5x 先导；若先导有稳定改善，再新增至少 108 首、达到
120 首（约 240 张谱、20 万音符）做有约 30 首留出歌曲/折的四折评估。更可信的
原型目标是 240 首 / 480 张谱 / 40 万音符。若 5x 先导仍不超过基线，应先修改
输入条件或模型，不要只堆相同数据。任何数量都不能保证当前踩音 + 单声道局部
音频能够确定横坐标或谱师风格。

## 3. 四折交叉验证结果

每折使用 60 epochs、AdamW、lr=1e-3、bf16 autocast、组级 MLP 结构；
checkpoint 选择依据 inner-val loss，outer test 不参与早停或调参。

| fold | train charts | test charts | test notes | best epoch | train center MAE | test x MAE | equidistant baseline | test side macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 5 | 3,694 | 60 | 0.14835 | 0.14082 | 0.14083 | 0.48238 |
| 1 | 19 | 4 | 1,551 | 27 | 0.14367 | 0.17280 | 0.17150 | 0.47466 |
| 2 | 14 | 9 | 7,274 | 2 | 0.14283 | 0.15732 | 0.14943 | 0.11914 |
| 3 | 17 | 6 | 7,761 | 33 | 0.15716 | 0.13929 | 0.12940 | 0.47988 |
| **加权汇总** |  | **24** | **20,280** |  |  | **0.14860** | **0.14188** | **0.35055** |

关键观察：

- 表中“equidistant baseline”列只用于跨折可比；fold1 的最强基线实际是
  `global_median = 0.17114`（等距为 `0.17150`），模型对两者都更差。
- 训练集 center MAE 始终在 0.143–0.157，模型连训练组的中心都没有拟合；
  这不是“只差更多训练轮数”的问题。
- fold2 的 best epoch 只有 2，side macro-F1 降到 0.119，说明 checkpoint
  选择对 side 头没有约束，且 side 头不稳定。
- fold0 的 test x-MAE 与 equidistant 几乎完全相同，说明模型实际学到的
  主要是全局组模板，而不是从音频/踩音条件中学到可泛化的横向位置。
- 四折中只有 fold0 略优于等距基线（约 0.006%），其余三折均更差。

以下机器报告和 checkpoint 路径只对应原训练工作区，不包含在 GitHub 仓库；
公开汇总以英文优先的 `docs/EXPERIMENT_REPORT.md` 为准。路径用于本机复现审计：

- `artifacts/reports/eval_report.json`（四折）
- `artifacts/reports/eval_audio_4fold.json`
- `artifacts/reports/cv_summary.json`

规范本机 checkpoint 路径（权重压缩包作为独立 release 资产发布）：

- 默认评估与交付使用 `artifacts/checkpoints/model_fold{0..3}_seed20261120.pt`；
- 原始训练输出在 `artifacts/v3_foldN/checkpoints/`；
- 独立评估的暂存副本在 `artifacts/v3_cv_audio/checkpoints/`；
- 上述 fold0 文件彼此 SHA256 相同
  （`38BE5F18...536B9A8C`），样例清单中的 fold0 路径只是生成本地副本，
  不改变权重内容。

## 4. 无音频消融

在 fold0 上使用相同的 60 epochs 和相同结构，`--no-audio` 训练：

| 配置 | test x MAE | test side macro-F1 | 最强基线 | 相对基线 |
|---|---:|---:|---:|---:|
| 有音频 | 0.14082 | 0.48238 | 0.14083 | +0.006% |
| 无音频 | 0.14228 | 0.47656 | 0.14083 | -1.03% |

无音频比有音频的加权 x-MAE 只差 0.00146；按 5 份谱面做逐曲配对 bootstrap，
`no_audio - audio` 的 95% 区间为 `[-0.00071, 0.00338]`，跨零。
在当前数据和特征下，局部 log-mel 音频条件没有带来可验证的泛化增益。

本机报告：`artifacts/reports/eval_noaudio_fold0.json`（未纳入公开仓库）。

## 5. 留出样例

样例只使用 fold0 checkpoint，曲目来自 fold0 留出歌曲
`Anomaly`、`Rubbish_Sorting_SP`、`Winter_Cube_SP`。

| 文件 | chart_id | notes | below | x 范围 | x 标准差 | 组内平均宽度 | 格式 | 保真 |
|---|---|---:|---:|---|---:|---:|---|---|
| `artifacts/samples/fold0_Anomaly_EZ.json` | `Anomaly_Anomaly_EZ` | 873 | 0 | 0.340–0.668 | 0.070 | 0.325 | 通过 | 通过 |
| `artifacts/samples/fold0_Anomaly_IN.json` | `Anomaly_Anomaly_IN` | 1,212 | 0 | 0.199–0.793 | 0.136 | 0.342 | 通过 | 通过 |
| `artifacts/samples/fold0_Rubbish_Sorting_SP.json` | `Rubbish_Sorting_SP_Rubbish_Sorting_SP` | 416 | 0 | 0.301–0.660 | 0.077 | 0.334 | 通过 | 通过 |
| `artifacts/samples/fold0_Winter_Cube_SP.json` | `Winter_Cube_SP_Winter_Cube_SP` | 924 | 0 | 0.301–0.668 | 0.119 | 0.333 | 通过 | 通过 |

本机样例清单：`artifacts/samples/manifest.json`（样例谱面及逐音符预测未纳入公开仓库）。

这些样例只用于证明“踩音 JSON → 模型 → 编译器 → 合法 formatVersion 3 JSON”
的工程链路可运行。它们不是质量验收样例，也不代表可玩性已通过。

以 `fold0_Anomaly_IN` 为例：真实多音符组内宽度均值约 `0.253`，模型输出约
`0.342`；真实 `x` 标准差约 `0.178`，模型输出约 `0.136`。这解释了为什么
x-MAE 看似接近基线，但生成谱面会明显向屏幕中部收缩、同押组偏宽。

## 6. 验收门槛

| 门槛 | 结果 | 判定 |
|---|---|---|
| 编译器 dummy 校验与选定的 fold0 样例 | 24/24 dummy + 4/4 样例通过；未逐折生成全部 test chart | 通过（有限范围） |
| 输入时间/类型/Hold 保真率 100% | 4/4 样例逐 tick 核验 | 通过 |
| 越界率 < 1% | 0% | 通过 |
| 非预期同 tick 完全重叠率 < 1% | 0% | 通过 |
| x-MAE 相对最强基线改善 ≥ 10% | -4.73% | **失败** |
| side macro-F1 相对多数类高 ≥ 10% | -12.79 个百分点 | **失败** |
| 至少 3 首留出曲目样例 | 3 首歌曲 / 4 份谱面 | 通过（仅工程演示） |
| 真人上机可玩性检查 | 未执行 | **未完成** |

## 7. 复现命令

```powershell
# 测试
.\.venv\Scripts\python.exe -m pytest -q

# 四折训练（分别运行；本次每折约 17–27 分钟）
.\.venv\Scripts\python.exe scripts\run_cv.py --output artifacts\v3_fold0 --fold 0 --device cuda --num-workers 0 --model-kind group_mlp --epochs 60
.\.venv\Scripts\python.exe scripts\run_cv.py --output artifacts\v3_fold1 --fold 1 --device cuda --num-workers 0 --model-kind group_mlp --epochs 60
.\.venv\Scripts\python.exe scripts\run_cv.py --output artifacts\v3_fold2 --fold 2 --device cuda --num-workers 0 --model-kind group_mlp --epochs 60
.\.venv\Scripts\python.exe scripts\run_cv.py --output artifacts\v3_fold3 --fold 3 --device cuda --num-workers 0 --model-kind group_mlp --epochs 60

# 四折独立评估（默认 artifacts/checkpoints 已放置 v3 四折权重）
.\.venv\Scripts\python.exe scripts\evaluate.py --out artifacts\reports\eval_report.json --device cuda

# fold0 无音频消融
.\.venv\Scripts\python.exe scripts\run_cv.py --output artifacts\v3_fold0_noaudio --fold 0 --device cuda --num-workers 0 --model-kind group_mlp --no-audio --epochs 60
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint-dir artifacts\v3_fold0_noaudio\checkpoints --out artifacts\reports\eval_noaudio_fold0.json --device cuda

# 生成一份留出样例
.\.venv\Scripts\python.exe scripts\infer_chart.py --tap data\processed\taps\Anomaly_Anomaly_IN.json --checkpoint artifacts\v3_fold0\checkpoints\model_fold0_seed20261120.pt --out artifacts\samples\fold0_Anomaly_IN.json --features-dir data\processed\features --device cpu
```

## 8. 失败原因与后续方向

1. **数据量不足**：24 份谱、12 首歌，无法支撑“音频 → 谱面横坐标”的泛化。
2. **条件信息不足**：踩音 JSON 只给时间/类型/Hold，横向位置本身高度多解；
   局部音频特征与横坐标的相关性弱。
3. **side 标签极不平衡**：below 仅 8.2%，且与单声道音频、类型、时间几乎不可分；
   当前 head 很容易退化为多数类或过预测 below。
4. **组级模板是弱条件**：模型能学到“同押组等距铺开”，但学不到每个组的中心位置，
   于是 x-MAE 与 equidistant 基线持平；样例横坐标集中在屏幕中部。
5. **Checkpoint 选择只看 val loss**：side 头没有独立的早停/选择标准，fold2 出现
   best epoch=2 且 side F1=0.119 的退化。

若要继续，优先顺序应是：

- 扩充曲目数量，并按曲目分组做真正的留出评估；
- 引入更强的谱面条件（风格、难度、段落、谱师/曲目 embedding），或让模型输出
  可解释的组中心/宽度潜变量；
- 把 side 作为可选头，或改用音符类型/位置规则而不是硬学不平衡标签；
- 在数据量足够后，再进入多判定线与动画事件阶段；
- 在正式发布前补真人上机可玩性检查、许可证审查和样例审计。

## 9. 许可证与再分发

`yuameshi/phigros-charts-repo` 未声明明确许可证。本次开源发布不包含原始音频、原始谱面和生成谱面。五个实验权重单独作为 GitHub release 资产发布，并附带 `MODEL_LICENSE.md` 与数据权利风险提示；权重可能编码训练内容，许可不授予语料权利。

- 不发布原始音频、原始谱面、生成谱面；
- `data/raw/`、`data/interim/`、`data/processed/`、本机 reports 和生成样例均由 `.gitignore` 排除。
