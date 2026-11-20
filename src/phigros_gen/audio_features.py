"""Audio feature extraction for the PhiTapMicro training pipeline.

The contract requires a mono 16 kHz signal transformed into a 64-bin
log-mel spectrogram with ``n_fft=1024`` and ``hop_length=320``, which yields
exactly 50 frames per second.  The per-note feature window (9 x 64) is sampled
by the dataset layer at inference/training time; this module only writes the
full spectrogram plus the metadata required by ``docs/DATA_CONTRACT.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover - environment issue, not logic
    sf = None

try:
    from scipy import signal as sp_signal
except Exception:  # pragma: no cover - environment issue, not logic
    sp_signal = None


SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 64
FRAME_RATE = 50.0
WINDOW_FRAMES = 9
EPS = 1e-6


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Return a triangular mel filterbank of shape ``[n_mels, n_fft//2 + 1]``."""
    fft_bins = n_fft // 2 + 1
    mel_points = np.linspace(
        _hz_to_mel(0.0), _hz_to_mel(float(sr) / 2.0), n_mels + 2
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int64)

    weights = np.zeros((n_mels, fft_bins), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        for j in range(left, center):
            if 0 <= j < fft_bins:
                weights[i, j] = (j - left) / max(1, center - left)
        for j in range(center, right):
            if 0 <= j < fft_bins:
                weights[i, j] = (right - j) / max(1, right - center)

    # Slaney-style area normalization so each filter sums to approximately one.
    widths = hz_points[2 : n_mels + 2] - hz_points[:n_mels]
    weights *= (2.0 / np.maximum(widths, 1e-12))[:, None]
    return weights


def _stft_scipy(
    x: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = N_FFT, hop: int = HOP_LENGTH
) -> np.ndarray:
    """Return complex STFT of shape ``[freq, frames]`` using scipy."""
    if sp_signal is None:
        raise ImportError("scipy is required for STFT; install scipy")
    _, _, z = sp_signal.stft(
        x,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=False,
        detrend=False,
    )
    return z


def _stft_numpy(
    x: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = N_FFT, hop: int = HOP_LENGTH
) -> np.ndarray:
    """Pure-numpy fallback STFT, centered and zero-padded like librosa."""
    x = np.asarray(x, dtype=np.float64)
    window = np.hanning(n_fft)
    pad = n_fft // 2
    x_pad = np.pad(x, (pad, pad), mode="constant")
    n_frames = 1 + (len(x_pad) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x_pad, n_fft)[
        : n_frames * hop : hop
    ]
    z = np.fft.rfft(frames * window, n=n_fft, axis=1).T
    return z


def _stft(
    x: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = N_FFT, hop: int = HOP_LENGTH
) -> np.ndarray:
    if sp_signal is not None:
        return _stft_scipy(x, sr, n_fft, hop)
    return _stft_numpy(x, sr, n_fft, hop)


def resample_mono(
    x: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Resample a 1-D signal to ``target_sr`` using scipy polyphase resampling."""
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr:
        return x
    if sp_signal is not None:
        gcd = int(np.gcd(int(orig_sr), int(target_sr)))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd
        return sp_signal.resample_poly(x, up, down).astype(np.float32)
    # Fallback: linear interpolation keeps the pipeline testable without scipy.
    n_out = int(round(len(x) * float(target_sr) / float(orig_sr)))
    t_in = np.arange(len(x), dtype=np.float64) / float(orig_sr)
    t_out = np.arange(n_out, dtype=np.float64) / float(target_sr)
    return np.interp(t_out, t_in, x).astype(np.float32)


def _load_with_torchaudio(path: str, target_sr: int) -> np.ndarray:
    try:
        import torchaudio

        waveform, orig_sr = torchaudio.load(path)
        mono = waveform.mean(dim=0).numpy().astype(np.float32)
        return resample_mono(mono, int(orig_sr), target_sr)
    except Exception as exc:
        raise RuntimeError(f"Unable to decode audio {path!r}") from exc


def load_audio_mono(
    path: str, target_sr: int = SAMPLE_RATE
) -> "tuple[np.ndarray, int, int]":
    """Load an audio file as mono ``float32`` at ``target_sr``.

    Returns ``(mono, target_sr, original_sr)``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if sf is None:
        mono = _load_with_torchaudio(path, target_sr)
        return mono, target_sr, -1
    try:
        data, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        mono = _load_with_torchaudio(path, target_sr)
        return mono, target_sr, -1
    if data.ndim == 1:
        mono = data
    elif data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = data.mean(axis=1).astype(np.float32)
    mono = resample_mono(mono, int(orig_sr), target_sr)
    return mono, target_sr, int(orig_sr)


def log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Compute a log-mel spectrogram and return ``float32 [T, n_mels]``."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.ndim == 0 or audio.size == 0:
        raise ValueError("empty audio signal")
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode="constant")
    z = _stft(audio, sr, n_fft, hop_length)
    power = np.abs(z) ** 2
    mel_power = mel_filterbank(n_mels, n_fft, sr) @ power
    return np.log(np.maximum(mel_power, EPS)).T.astype(np.float32)


def extract_features(
    audio_path: str,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> "tuple[np.ndarray, float]":
    """Extract ``frames`` and ``audio_seconds`` for an audio file."""
    mono, out_sr, _ = load_audio_mono(audio_path, sr)
    frames = log_mel_spectrogram(mono, out_sr, n_fft, hop_length, n_mels)
    audio_seconds = float(len(mono)) / float(out_sr)
    return frames, audio_seconds


def sample_window(
    frames: np.ndarray,
    frame_rate: float = FRAME_RATE,
    time_sec: float = 0.0,
    n_frames: int = WINDOW_FRAMES,
) -> np.ndarray:
    """Sample a fixed-size window centered on ``time_sec``.

    The window has ``n_frames`` frames (default 9: four before, four after) and
    is zero-padded when the requested time falls outside the spectrogram.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError("frames must be a [T, n_mels] array")
    total = frames.shape[0]
    if total == 0:
        return np.zeros((n_frames, frames.shape[1]), dtype=np.float32)
    center = int(round(float(time_sec) * float(frame_rate)))
    left = n_frames // 2
    start = center - left
    out = np.zeros((n_frames, frames.shape[1]), dtype=np.float32)
    for out_i, frame_i in enumerate(range(start, start + n_frames)):
        if 0 <= frame_i < total:
            out[out_i] = frames[frame_i]
    return out


def write_features_npz(
    out_path: str,
    frames: np.ndarray,
    frame_rate: float = FRAME_RATE,
    audio_seconds: float = 0.0,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=np.asarray(frames, dtype=np.float32),
        frame_rate=np.float32(frame_rate),
        audio_seconds=np.float32(audio_seconds),
    )


def read_features_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "frames": data["frames"],
            "frame_rate": float(data["frame_rate"]),
            "audio_seconds": float(data["audio_seconds"]),
        }


def load_jsonl(path: str) -> "list[dict]":
    rows: "list[dict]" = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_relative(repo_root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


def process_chart(
    audio_path: str,
    out_path: str,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> Dict[str, Any]:
    """Extract and persist one chart's features, returning a report dict."""
    frames, audio_seconds = extract_features(
        audio_path, sr, n_fft, hop_length, n_mels
    )
    write_features_npz(out_path, frames, FRAME_RATE, audio_seconds)
    return {
        "audio_path": audio_path,
        "frames": int(frames.shape[0]),
        "mels": int(frames.shape[1]),
        "frame_rate": FRAME_RATE,
        "audio_seconds": audio_seconds,
        "output": out_path,
    }


def iter_chart_rows(rows: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in rows:
        if "audio_path" in row and "chart_id" in row:
            yield row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch log-mel feature extraction for chart audio."
    )
    parser.add_argument("--dataset", default="data/processed/dataset.jsonl")
    parser.add_argument(
        "--output", default="data/processed/features", help="output directory"
    )
    parser.add_argument("--repo-root", default=".", help="resolve relative paths")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--report",
        default=None,
        help="optional JSON report path (defaults to <output>/../feature_report.json)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = os.path.abspath(args.repo_root)
    output_dir = os.path.abspath(args.output)
    rows = load_jsonl(os.path.join(repo_root, args.dataset))

    report: Dict[str, Any] = {
        "config": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "frame_rate": FRAME_RATE,
        },
        "charts": [],
        "errors": [],
    }

    count = 0
    for row in iter_chart_rows(rows):
        if args.limit is not None and count >= args.limit:
            break
        chart_id = row["chart_id"]
        audio_path = resolve_relative(repo_root, row["audio_path"])
        out_path = os.path.join(output_dir, f"{chart_id}.npz")
        try:
            info = process_chart(audio_path, out_path)
            info["chart_id"] = chart_id
            report["charts"].append(info)
        except Exception as exc:  # pragma: no cover - depends on audio data
            report["errors"].append(
                {"chart_id": chart_id, "audio_path": audio_path, "error": str(exc)}
            )
        count += 1

    report["num_charts"] = len(report["charts"])
    report["num_errors"] = len(report["errors"])
    if report["charts"]:
        report["mean_audio_seconds"] = float(
            np.mean([c["audio_seconds"] for c in report["charts"]])
        )
        report["mean_frames"] = float(
            np.mean([c["frames"] for c in report["charts"]])
        )

    report_path = args.report or os.path.join(
        repo_root, "artifacts", "reports", "feature_report.json"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Wrote features for {report['num_charts']} charts -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
