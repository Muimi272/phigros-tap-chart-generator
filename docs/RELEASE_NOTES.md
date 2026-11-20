# v0.1.0-research

## English

This is the first public research release of the Phigros tap-to-chart
prototype. It includes source code, tests, an English-first experiment report,
a model card, data/license notices, and five experimental model checkpoints as
a separate GitHub release asset.

### Result

- Four-fold song-grouped x-MAE: `0.14860`; equidistant baseline: `0.14188`.
- Four-fold side macro-F1: `0.35055`; weighted majority baseline: `0.47846`.
- Both primary quality gates failed. This is not a production chart generator.
- The four fold-0 held-out examples passed format and input-fidelity checks,
  but are intentionally not redistributed.

### Assets

- `checkpoints-v0.1.0-research.zip`: four audio fold checkpoints and one
  fold-0 no-audio ablation checkpoint; 11,849,380 bytes.
- SHA-256: `BD972F3D7B90456582058BCF5F3C98A55E3DD3D1F5DD044ECB85BD21A8F53250`.
- `SHA256SUMS.txt`: archive and individual checkpoint hashes.

Checkpoint files are PyTorch serialized objects. Load only assets from sources
you trust. The training corpus has no declared license in its source repository;
weights may encode corpus information. See `MODEL_LICENSE.md` and
`docs/DATA_AND_LICENSES.md`. The release does not include source audio, charts,
tap JSON, generated chart samples, or local reports.

### 中文摘要

这是 Phigros 踩音转谱原型的首个公开研究版，包含源码、测试、英文优先的实验
报告、模型卡、数据/许可声明，以及独立 release 权重包。四折 `x_MAE=0.14860`
（等距基线 `0.14188`），side macro-F1 `0.35055`（多数类 `0.47846`），两项
质量门槛均未通过。11,849,380 字节权重包含四个有音频折模型和一个 fold0
无音频消融模型，SHA-256 为 `BD972F3D7B90456582058BCF5F3C98A55E3DD3D1F5DD044ECB85BD21A8F53250`。
语料无明确许可证，权重可能编码训练内容；不包含原始音频、原谱或生成样例，
详情见 `MODEL_LICENSE.md` 和 `docs/DATA_AND_LICENSES.md`。
