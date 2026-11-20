# Phigros Tap-to-Chart Generator

Research prototype for generating **format-valid candidate** `formatVersion: 3`
Phigros charts from a tap JSON plus local audio features. The repository
contains the schema, chart parser, audio feature extraction, grouped dataset
builder, model, training, inference, compiler, and independent evaluation code.

**Current status (2026-11-20): engineering loop complete, chart quality not
accepted.** See [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md) for the
English-first detailed report and [`RESULTS.md`](RESULTS.md) for the Chinese
report. The four-fold
model does not beat the non-learning `equidistant` baseline on x-MAE, and its
side head does not beat the majority-class macro-F1. Do not present the current
checkpoints or samples as production-quality chart generation.

## Environment

Verified locally on 2026-11-20:

- Windows 10.0.26200, Python 3.11.9
- NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, capability `(12, 0)` / `sm_120`
- `torch 2.7.1+cu128`, CUDA 12.8
- `numpy 2.4.6`, `scipy 1.17.1`, `soundfile 0.14.0` (libsndfile 1.2.2),
  `jsonschema 4.26.0`, `tqdm 4.70.1`, `pytest 9.2.0`

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

All commands in this repository assume `.venv\Scripts\python.exe`.

## Reproduce the experiment

First review the upstream repository's current data terms and obtain any needed
permissions. Then, from the repository root:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
git clone https://github.com/yuameshi/phigros-charts-repo data\raw\phigros-charts-repo
.\.venv\Scripts\python.exe scripts\build_dataset.py
.\.venv\Scripts\python.exe scripts\build_features.py

foreach ($fold in 0..3) {
  .\.venv\Scripts\python.exe scripts\run_cv.py --output "artifacts\v3_fold$fold" --fold $fold --device cuda --num-workers 0 --model-kind group_mlp --epochs 60
}

.\.venv\Scripts\python.exe -m pytest -q
```

To evaluate, place the four `model_foldN_seed20261120.pt` checkpoints in one
directory and run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --checkpoint-dir artifacts\checkpoints --out artifacts\reports\eval_report.json --device cuda --seed 20261120
```

For the no-audio fold-0 ablation, append `--no-audio` to `run_cv.py`. Complete
commands and fold-level details are in `docs/EXPERIMENT_REPORT.md`.

## Data

- `data/raw/phigros-charts-repo/`: 24 chart JSON files containing
  `judgeLineList`, across 13 song folders and 12 unique song IDs.
- 20,280 notes, 15,625 same-tick groups; all 13 audio files decode with
  `soundfile`.
- `data/processed/dataset.jsonl`: frozen training rows.
- `data/processed/taps/{chart_id}.json`: `phigros-tap/v1` documents.
- `data/processed/features/{chart_id}.npz`: cached log-mel features.
- `data/processed/splits.json`: frozen song-grouped 4-fold split,
  SHA256 `772fd2eac22a32581191f5da366bce178f5319354c05fb674cc0b0c3db2e7aa0`.
- Raw audio/charts are gitignored and not redistributed. The source repository
  has no explicit license; check current terms and obtain permission before use.

Review [`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md) before obtaining
data or using/re-distributing model weights. Five checkpoints are attached only
as a separate research artifact to the GitHub release; weights may encode
unlicensed training content.

## Model and training

The current experiment uses `PhiTapGroupMLP`:

- one static, centered judge line;
- group-level features (time, BPM, density, type ratio, local audio windows,
  neighbor spacing) predict a group center and spread;
- a per-note MLP predicts `above` / `below`;
- x is trained with a 128-bin Gaussian likelihood plus Hungarian group matching,
  center/spread auxiliary L1, and matched position L1;
- 646,148 parameters, 60 epochs, AdamW, lr `1e-3`, bf16 autocast.

Training entry point:

```powershell
.\.venv\Scripts\python.exe scripts\run_cv.py --output artifacts\v3_fold0 --fold 0 --device cuda --num-workers 0 --model-kind group_mlp --epochs 60
```

Use `--no-audio` for the ablation. `scripts\evaluate.py` runs an independent
outer-test evaluation and validates the frozen split SHA256.

## Results

Four-fold song-grouped cross-validation (24 test charts, 20,280 test notes):

| metric | model | strongest non-learning baseline | verdict |
|---|---:|---:|---|
| x MAE | 0.14860 | 0.14188 (`equidistant`) | worse by 4.73% |
| side macro-F1 | 0.35055 | 0.47846 (majority) | worse by 12.79 pp |
| out-of-bounds rate | 0.0 | - | pass |
| unexpected same-tick duplicate rate | 0.0 | - | pass |

The compiler produces valid `formatVersion: 3` JSON for 24/24 dummy validation
charts and 4/4 fold-0 held-out samples; input time/type/Hold fidelity is 4/4. However,
the learned x distribution collapses toward the screen center, the side head
predicts no `below` notes in the samples, and no human playtest has been done.
The local-only samples under `artifacts/samples/` are engineering demonstrations
and are not included in this repository or release.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite covers schema/tap validation, parsing, audio features, dataset
grouping/padding, model forward/masking, compiler semantic/parser validation,
inference round-trip, split immutability, and evaluation metrics.

## Layout

```text
docs/EXPERIMENT_REPORT.md detailed results, methodology, data estimate
RESULTS.md              Chinese results and failure analysis
MODEL_CARD.md           intended use, metrics, limits, data estimate
docs/DATA_AND_LICENSES.md data and release boundaries
src/phigros_gen/        parser, features, dataset, model, train, infer, compiler
scripts/                dataset build, CV training, evaluation, inference CLIs
tests/                  pytest suite
artifacts/              local outputs (gitignored); checkpoints are release assets
```

## Limitations

- 24 charts / 12 songs are insufficient for general chart generation.
- The tap JSON does not contain x, line motion, or rotation; x is inherently
  multi-valued from tap + local audio.
- `below` is only 8.2% of notes and is not learnable from the current mono
  log-mel condition.
- Only a single static judge line is generated; complex multi-line animation is
  out of scope for this phase.
- No final all-song model was trained after the quality gates failed; the
  four fold checkpoints are the authoritative experimental artifacts.

## License and redistribution

The source chart repository has no explicit license. Raw audio, charts, tap
data, generated samples, and machine-specific reports are not included. The
experimental model weights are attached separately to the research release
with a prominent data-rights notice; they may encode training information.
See `MODEL_LICENSE.md` and `docs/DATA_AND_LICENSES.md`.

## 中文摘要

这是一个实验性研究原型，可将踩音 JSON 和本地音频特征转换为格式合法的
Phigros 候选谱面，但不负责从音频检测踩音。本次四折评估没有超过等距横坐标
基线，方向分类也低于多数类基线；生成质量未通过验收。详细中文结果见
[`RESULTS.md`](RESULTS.md)，数据许可和权重边界见
[`docs/DATA_AND_LICENSES.md`](docs/DATA_AND_LICENSES.md)。
