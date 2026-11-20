# Data, Licensing, and Release Policy

## English

### Source data

The experiment used charts and corresponding local audio from
[`yuameshi/phigros-charts-repo`](https://github.com/yuameshi/phigros-charts-repo).
GitHub reports no license for that repository. Public accessibility is not a
license grant. This project does not redistribute the source charts, audio,
processed tap JSON, cached features, or generated chart files.

Anyone reproducing the experiment must independently verify the current source
terms and obtain any required permissions. A license for this repository's
software does not cover the source corpus or Phigros assets.

### Model artifacts

Five small checkpoints (four audio CV folds and the fold-0 no-audio ablation)
are attached to the `v0.1.0-research` GitHub release as a separate archive.
The weights are released for transparent research reproduction under the
limited terms in `MODEL_LICENSE.md`. They may encode information from the
unlicensed training corpus; redistribution therefore has legal uncertainty.
The release makes no claim that weights are free of memorized or derivative
content and does not grant rights to any training data.

Do not use the model to publish or commercialize generated charts without
reviewing the relevant rights. Do not treat structural validation as proof of
human playability.

### Excluded local files

The following are intentionally excluded from GitHub:

- `data/raw/`, `data/interim/`, and `data/processed/`;
- `artifacts/samples/`, including model-generated charts and per-note outputs;
- `artifacts/reports/`, because local reports contain machine-specific absolute
  paths; aggregate metrics are transcribed into `docs/EXPERIMENT_REPORT.md`;
- `artifacts/checkpoints/`; checkpoints are distributed only as the explicit
  release asset and accompanied by SHA-256 hashes.

### Project code license

`LICENSE` grants MIT rights to original project software source code only.
It does not grant rights to third-party data, charts, audio, or content that
may be encoded in a model artifact. See `MODEL_LICENSE.md` for the checkpoint
notice.

## 中文说明

本实验使用了 `yuameshi/phigros-charts-repo` 中的谱面和本地音频。GitHub
仓库元数据没有声明许可证；公开可访问不等于允许复制或再分发。本项目不会
在仓库中发布原谱、音频、踩音数据、特征缓存或生成谱面。

五个微型 checkpoint 作为独立研究复现实验资产附在 `v0.1.0-research`
release。由于训练语料没有明确许可证，权重可能编码训练内容，公开再分发
存在法律不确定性；权重许可不授予任何训练数据权利。请勿在未检查版权和
许可的情况下发布或商业化生成谱面。

原始/处理中间数据、本机 reports、生成样例不会进入 Git。`LICENSE` 中的 MIT
仅覆盖本项目原创代码，不覆盖第三方语料、游戏素材或模型权重中可能包含的
第三方内容。具体边界见 `MODEL_LICENSE.md`。
