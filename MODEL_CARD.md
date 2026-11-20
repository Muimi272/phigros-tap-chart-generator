# Model Card: Phigros Tap-to-Chart Generator (Experimental v0.1.0)

## Summary

This is an experimental research model that maps a custom Phigros tap-event
document plus local audio features to note x positions and above/below sides.
It does **not** detect note times from audio. The time, note type, hold duration,
and simultaneous-note groups are supplied by the input document.

This model did not pass the project's quality gates. It is released as a
reproducibility artifact, not as a chart-authoring tool or a claim of usable
generative quality.

## Intended use

- Reproduce the reported local experiment and inspect failure modes.
- Test the input schema, model loading, and formatVersion 3 compiler pipeline.
- Research better models on datasets that are licensed for the intended use.

## Out-of-scope use

- Automatically publishing, submitting, or selling generated Phigros charts.
- Claiming that generated charts meet official chart quality or playability.
- Releasing generated charts without checking the source corpus rights.
- Treating predictions as faithful reconstructions of the source charts.

## Architecture and interface

- Architecture: `PhiTapGroupMLP`, a group-level MLP predicting center and
  spread, with a per-note side head.
- Parameters: 646,148.
- Inputs: tap groups, note types, hold durations, timing, BPM, difficulty/source
  metadata, density features, and optional 9-frame x 64-bin local log-mel audio.
- Outputs: normalized horizontal x in `[0, 1]` and `above` / `below` side.
- Compiler: one static centered judge line; generated note speed is fixed at
  1.0. It copies input timing, type, hold length, and grouping.
- Weights: five checkpoints are attached to the GitHub release: four audio
  cross-validation fold checkpoints plus one fold-0 no-audio ablation checkpoint.

## Training data

The local experiment used 24 chart JSON files covering 12 unique song IDs,
20,280 notes, and 15,625 simultaneous-note groups from
`yuameshi/phigros-charts-repo`. Audio was decoded locally and converted to
mono 16 kHz log-mel features. The source repository declares no license in its
GitHub metadata. The dataset is not included in this release; users must
independently verify they have rights to any data they use.

## Evaluation

Evaluation used a frozen four-fold split grouped by unique song ID, keeping
versions and difficulties of the same song together. Every chart appears in
one outer test fold.

- Weighted outer-test x MAE: `0.14860`.
- Strongest aggregate baseline: equidistant within-group placement, `0.14188`.
- Relative x-MAE gain: `-4.73%` (the model is worse).
- Weighted side macro-F1: `0.35055` vs majority baseline `0.47846`.
- Out-of-bounds rate: `0`; unexpected exact same-tick duplicate rate: `0`.
- Fold-0 no-audio x MAE: `0.14228`; audio x MAE: `0.14082`; the paired
  per-chart bootstrap interval for no-audio minus audio crosses zero.
- Four fold-0 held-out sample charts passed compiler/parser validation and
  exact time/type/hold fidelity checks, but all predict zero below notes and
  are visibly concentrated toward the screen center.
- No human playtest was performed.

See `RESULTS.md` for fold-level metrics, execution commands, statistical
details, and the Chinese translation in `RESULTS.zh-CN.md`.

## Limitations and risks

- Only 12 independent songs are available, too few to support a reliable claim
  of generalization to unseen music or chart styles.
- Horizontal placement is underdetermined by tap timing/type and local mono
  audio. The current model learns little beyond a generic group template.
- The below-side class is only 8.2% of the source notes and is poorly
  predictable from current features.
- The outputs have not been assessed by human players and must not be assumed
  playable merely because JSON validation succeeds.
- Checkpoints may encode information from the training corpus. The source
  dataset has no declared license, so public weight redistribution has legal
  uncertainty. The weights are supplied only for transparent research
  reproducibility; this project grants no rights to the training data, source
  charts, or audio.

## Data recommendation

There is no measured learning curve from this experiment, so an exact minimum
cannot be inferred. A 5x pilot should target at least **60 independent songs
total** (48 more than this experiment) to test whether a learning curve emerges.
For a more credible four-fold study with about 30 held-out songs per fold, use
at least **120 songs total** (108 more), approximately 240 charts and 200,000
notes. A more useful prototype target is 240 songs / 480 charts / 400,000 notes;
600 songs / 1,200 charts / about 1 million notes is a plausible broader-model
scale. These are planning estimates, not guarantees; more charts of the same
songs do not replace independent songs.

## Licensing

The repository's software source code is licensed under MIT in `LICENSE`.
The MIT license does not grant rights to third-party charts, audio, the source
dataset, or any material encoded by the weights. The model artifact is
provided for research reproduction with no warranty; see `MODEL_LICENSE.md`.

## 中文摘要

此模型用于踩音 JSON + 局部音频特征到候选谱面横坐标/上下侧的实验；不从音频
检测音符时间。四折 `x_MAE=0.14860`，差于等距基线 `0.14188`；side
macro-F1 `0.35055`，低于多数类 `0.47846`。当前权重仅供研究复现，不能据此
宣称谱面质量或可玩性。训练集来源未声明许可证，权重再分发存在法律不确定性。
建议至少新增 48 首独立歌曲作先导，120 首总量作较可信的分组四折评估；这些
是规划估算，不是成功保证。
