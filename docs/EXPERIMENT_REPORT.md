# Experimental Results and Failure Analysis

**Release:** `v0.1.0-research`  
**Date:** 2026-11-20  
**Status:** Engineering pipeline validated; generative quality gates failed.

## Executive summary

This experiment converts Phigros charts to a tap-event JSON, extracts local
audio features, trains a small conditional model, and compiles predictions to
formatVersion 3 candidate charts. It was run on one local RTX 5060 Laptop GPU
(8 GB, `sm_120`). The resulting model did not outperform simple baselines on
held-out songs and is not suitable for production chart authoring.

| Metric | Model | Baseline | Result |
|---|---:|---:|---|
| Weighted x MAE | 0.14860 | 0.14188, equidistant | 4.73% worse |
| Weighted side macro-F1 | 0.35055 | 0.47846, majority | 12.79 percentage points worse |
| Out-of-bounds prediction rate | 0.0% | - | Pass |
| Unexpected exact same-tick duplicate rate | 0.0% | - | Pass |

The x-MAE paired-bootstrap interval against the strongest per-fold baseline
crosses zero; the point estimate is worse. The model does not satisfy either
primary quality threshold. A valid JSON file is not evidence of a playable or
well-designed chart.

## Dataset and split

- 24 chart files, 12 unique songs, 20,280 notes, 15,625 same-tick groups.
- Note types: 10,262 tap, 6,898 drag, 1,828 hold, 1,292 flick.
- Side labels: 18,621 above and 1,659 below (8.18% below).
- Five charts use formatVersion 1 and 19 use formatVersion 3.
- All 13 local audio files decoded; source audio and charts are excluded from
  this release.
- Four outer folds are grouped by unique song ID, so alternate difficulties
  and legacy versions stay together. The split file SHA256 is
  `772fd2eac22a32581191f5da366bce178f5319354c05fb674cc0b0c3db2e7aa0`.

| Fold | Train charts | Test charts | Test notes | Best epoch | Train center MAE | Test x MAE | Equidistant MAE | Side macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 5 | 3,694 | 60 | 0.14835 | 0.14082 | 0.14083 | 0.48238 |
| 1 | 19 | 4 | 1,551 | 27 | 0.14367 | 0.17280 | 0.17150 | 0.47466 |
| 2 | 14 | 9 | 7,274 | 2 | 0.14283 | 0.15732 | 0.14943 | 0.11914 |
| 3 | 17 | 6 | 7,761 | 33 | 0.15716 | 0.13929 | 0.12940 | 0.47988 |
| **Weighted total** |  | **24** | **20,280** |  |  | **0.14860** | **0.14188** | **0.35055** |

The strongest baseline is fold-specific: equidistant in folds 0, 2, and 3;
global median in fold 1 (`0.17114`, versus equidistant `0.17150`). The table's
equidistant column is shown for a consistent cross-fold comparison.

## Model and training procedure

The evaluated model is `PhiTapGroupMLP` with 646,148 parameters. Group-level
features predict a center and spread; a per-note head predicts above/below.
Features include time, BPM, difficulty/source IDs, group size/type mix, local
note density, previous/next group intervals, and a 9 x 64-bin log-mel window.
The x objective combines 128-bin Gaussian classification, Hungarian matching,
center/spread auxiliary loss, and matched-position L1. Side uses weighted
cross-entropy.

All four folds used the fixed seed `20261120`, 60 maximum epochs, AdamW at
`1e-3`, no weight decay, bf16 autocast, early stopping on the inner validation
chart, and the same frozen split. The fold checkpoints were independently
evaluated on their outer test songs. No final all-song checkpoint was trained.

The compiler emits one static centered judge line, sets note speed to `1.0`,
and copies input time, type, hold duration, and same-tick grouping. It predicts
only x and side; it does not infer beat times from audio.

## Ablation

Fold 0 was trained again without audio features using the same architecture,
seed, split, and 60-epoch maximum:

| Fold-0 variant | x MAE | Side macro-F1 | Equidistant baseline |
|---|---:|---:|---:|
| With audio | 0.14082 | 0.48238 | 0.14083 |
| Without audio | 0.14228 | 0.47656 | 0.14083 |

The no-audio x-MAE is 0.00146 higher. Paired bootstrap over five held-out
charts for `no_audio - audio` gives `[-0.00071, 0.00338]`, crossing zero. This
small sample does not establish an audio benefit.

## What failed and why

1. **Insufficient independent songs.** There are only 12 unique songs. Notes
   from one song are highly correlated, so 20,280 notes do not behave like
   20,280 independent examples.
2. **The target is underdetermined.** Tap time, type, hold duration, BPM, and
   local mono audio do not uniquely determine horizontal placement, side, or
   chart style. Different valid charts can map the same song events to
   different positions.
3. **The model underfits even training groups.** Training center MAE remains
   0.143-0.157. The group MLP learns a generic template but not reliable
   song-conditioned group centers.
4. **The audio signal is weak for this target.** The fold-0 ablation shows no
   significant x-MAE difference. Mono log-mel lacks panning cues, and local
   spectral texture is not a stable proxy for a chart author's x choices.
5. **The side label is imbalanced and unstable.** Below is only 8.18% of notes.
   Side macro-F1 is near the majority baseline in folds 0, 1, and 3, then
   collapses in fold 2.
6. **Generated distributions collapse toward center.** Four fold-0 samples
   compile and preserve note timing/type/hold exactly, but predict no below
   notes and have narrower x distributions than the references. They are
   engineering demonstrations only; no human playtest was performed.
7. **No quality claim follows from compilation.** Format validation verifies
   structure and internal consistency, not readability, balance, or playability.

## How much more data?

This experiment has no learning curve, so no exact minimum can be inferred.
The bottleneck includes missing/ambiguous conditioning, not just sample count.
The following are planning milestones, not guarantees:

| Milestone | Unique songs total | Charts (approx.) | Notes (approx.) | Purpose |
|---|---:|---:|---:|---|
| Current | 12 | 24 | 20k | Baseline feasibility run |
| 5x pilot | 60 (add 48) | 120 | 100k | Check whether a learning curve separates from baselines |
| Minimum credible 4-fold study | 120 (add 108) | 240 | 200k | At least about 30 held-out songs per fold |
| Prototype target | 240 (add 228) | 480 | 400k | More useful test of style/song generalization |
| Broader model range | 600 (add 588) | 1,200 | 1M | Plausible scale for a general-model attempt |

The **minimum recommended additional collection is 48 independently authored
songs** for a 5x pilot. For a statistically more credible grouped four-fold
evaluation, collect **at least 108 additional songs** (120 total), ideally
with multiple chart difficulties and cleared audio rights. If the 5x pilot
remains flat against baselines, change the input contract/model or stop rather
than collecting more of the same data. Even 600 songs cannot guarantee success
if the desired x/side/style information is not present in the inputs.

## Candidate outputs and quality checks

Four fold-0-held-out files were generated locally for three test songs:
`Anomaly`, `Rubbish_Sorting_SP`, and `Winter_Cube_SP`. All four passed the
project semantic validator and parser; flattened note counts and per-tick
type/hold multisets matched their tap inputs. All predicted below counts were
zero. The sample JSON and prediction files are **not included** because the
training data license is unclear and the outputs may be derivative material.

Compiler-only validation using deterministic dummy predictions passed for
24/24 processed charts. This is separate from model evaluation and must not be
read as 24/24 model-generated charts.

## Reproduction

The source corpus is not included. Obtain it only if its terms permit your use.
Commands below are PowerShell commands run from the repository root on Windows.

```powershell
# Create an isolated environment and install dependencies.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Acquire source data only after checking its current license and terms.
New-Item -ItemType Directory -Force data\raw | Out-Null
git clone https://github.com/yuameshi/phigros-charts-repo data\raw\phigros-charts-repo

# Build tap JSON, manifest, audit and grouped splits; then extract log-mel.
.\.venv\Scripts\python.exe scripts\build_dataset.py
.\.venv\Scripts\python.exe scripts\build_features.py

# Reproduce the four outer folds (one process at a time).
foreach ($fold in 0..3) {
  .\.venv\Scripts\python.exe scripts\run_cv.py --output "artifacts\v3_fold$fold" --fold $fold --device cuda --num-workers 0 --model-kind group_mlp --epochs 60
}

# Evaluate from a directory containing the four canonical model_foldN checkpoints.
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint-dir artifacts\checkpoints --out artifacts\reports\eval_report.json --device cuda --seed 20261120

# Run tests (works without the corpus; the real-data split test skips cleanly).
.\.venv\Scripts\python.exe -m pytest -q
```

The release's checkpoint archive includes the original five weight artifacts
and a `SHA256SUMS.txt`; the release assets are intended to reproduce the
reported checkpoint experiment, not to certify chart quality. PyTorch checkpoint
files should only be loaded from trusted sources.

## License and release scope

MIT in `LICENSE` covers this project's original source code only. The upstream
chart repository has no declared license in GitHub metadata. No raw audio,
source chart JSON, processed tap data, generated sample charts, or local reports
with machine-specific paths are included in this repository. The weights are
attached as a separate research-only release asset under `MODEL_LICENSE.md`;
they do not convey rights to the data and may encode training information.
Review `DATA_AND_LICENSES.md` before using or redistributing any artifacts.

## 中文摘要

本实验在本机 RTX 5060 Laptop GPU 上完成踩音 JSON、音频特征、分组训练、
推理和 formatVersion 3 编译闭环，但生成质量未达标：四折 `x_MAE=0.14860`，
差于等距基线 `0.14188`；side macro-F1 为 `0.35055`，低于多数类 `0.47846`。
训练数据只有 12 首独立歌曲，且输入中的踩音和单声道局部音频无法唯一决定
横坐标、上下侧或谱师风格。建议至少再收集 48 首有权使用的独立歌曲做 5x
先导实验；更可信的四折研究至少需再增加 108 首（总计 120 首）。这只是
实验规划估算，不保证成功；若先导实验仍不优于基线，应改变条件信息或模型。
完整中文说明见仓库根目录 `RESULTS.md`。
