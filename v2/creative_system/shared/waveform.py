"""Decode uploaded audio into a normalized amplitude envelope."""
from __future__ import annotations

import array
import math
import os
import shutil
import struct
import subprocess
import wave
from typing import List, Optional, Sequence

TARGET_RATE = 22050
FFMPEG_TIMEOUT_S = 90

_AUDIO_EXTENSIONS = {
    "mp3",
    "mpeg",
    "mpga",
    "wav",
    "wave",
    "webm",
    "opus",
    "ogg",
    "oga",
    "m4a",
    "mp4",
    "aac",
    "flac",
    "aiff",
    "aif",
    "amr",
    "3gp",
}


def _file_extension(path: str) -> str:
    name = os.path.basename(path)
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[1].lower()


def load_mono_samples(path: str) -> List[float]:
    if not path or not os.path.isfile(path):
        raise ValueError("Audio file was not found")
    ext = _file_extension(path)
    if ext and ext not in _AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format .{ext}")

    samples: Optional[List[float]] = None
    if ext in {"wav", "wave"}:
        try:
            samples = _read_wav(path)
        except Exception as exc:
            print(f"wav parse failed, trying ffmpeg: {exc}")

    if samples is None:
        samples = _decode_with_ffmpeg(path)

    if not samples:
        raise ValueError("Audio file contained no samples")
    return samples


def rms_envelope(samples: Sequence[float], bins: int, power: float = 0.65) -> List[float]:
    """Downsample audio to `bins` RMS values in 0..1, with a lifted curve for visuals."""
    count = len(samples)
    if count == 0 or bins <= 0:
        return [0.0] * max(1, bins)

    values: List[float] = []
    for i in range(bins):
        start = int(i * count / bins)
        end = int((i + 1) * count / bins)
        if end <= start:
            end = min(count, start + 1)
        chunk = samples[start:end]
        acc = 0.0
        peak = 0.0
        for sample in chunk:
            acc += sample * sample
            abs_s = abs(sample)
            if abs_s > peak:
                peak = abs_s
        rms = math.sqrt(acc / len(chunk)) if chunk else 0.0
        values.append(0.72 * rms + 0.28 * peak)

    ranked = sorted(values)
    cap = ranked[int(len(ranked) * 0.96)] if ranked else 1.0
    if cap <= 1e-8:
        cap = max(values) if values else 1.0
    if cap <= 1e-8:
        return [0.08] * bins

    out = []
    for value in values:
        unit = min(1.0, value / cap)
        out.append(0.06 + 0.94 * (unit ** power))
    return _smooth(out, radius=2)


def _smooth(values: Sequence[float], radius: int = 2) -> List[float]:
    if radius <= 0 or len(values) < 3:
        return list(values)
    n = len(values)
    out: List[float] = []
    for i in range(n):
        total = 0.0
        weight = 0.0
        for j in range(i - radius, i + radius + 1):
            idx = min(n - 1, max(0, j))
            w = 1.0 + radius - abs(j - i)
            total += values[idx] * w
            weight += w
        out.append(total / weight if weight else values[i])
    return out


def _read_wav(path: str) -> List[float]:
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels() or 1
        width = handle.getsampwidth()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    if not raw or n_frames <= 0:
        return []

    if width == 1:
        mono = _mix_down([(b - 128) / 128.0 for b in raw], channels)
    elif width == 2:
        data = array.array("h")
        data.frombytes(raw[: len(raw) - (len(raw) % 2)])
        mono = _mix_down([s / 32768.0 for s in data], channels)
    elif width == 3:
        ints: List[int] = []
        for i in range(0, len(raw) - 2, 3):
            val = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if val & 0x800000:
                val -= 0x1000000
            ints.append(val)
        mono = _mix_down([s / 8388608.0 for s in ints], channels)
    elif width == 4:
        if len(raw) >= 4 and _looks_like_float32(raw):
            count = len(raw) // 4
            floats = struct.unpack("<" + "f" * count, raw[: count * 4])
            mono = _mix_down(floats, channels)
        else:
            data = array.array("i")
            data.frombytes(raw[: len(raw) - (len(raw) % 4)])
            mono = _mix_down([s / 2147483648.0 for s in data], channels)
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    return mono


def _looks_like_float32(raw: bytes) -> bool:
    probe = struct.unpack("<" + "f" * min(32, len(raw) // 4), raw[: min(128, len(raw) - len(raw) % 4)])
    finite = [abs(v) for v in probe if math.isfinite(v)]
    if len(finite) < 4:
        return False
    return max(finite) <= 8.0


def _mix_down(samples: Sequence[float], channels: int) -> List[float]:
    if channels <= 1:
        return [float(s) for s in samples]
    out: List[float] = []
    n = len(samples)
    for i in range(0, n - channels + 1, channels):
        acc = 0.0
        for c in range(channels):
            acc += float(samples[i + c])
        out.append(acc / channels)
    return out


def _ffmpeg_exe() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception:
        return None
    try:
        return get_ffmpeg_exe()
    except Exception:
        return None


def _decode_with_ffmpeg(path: str) -> List[float]:
    exe = _ffmpeg_exe()
    if not exe:
        raise ValueError(
            "Could not decode this audio format. Install ffmpeg, or upload a WAV file."
        )
    cmd = [
        exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_S,
            check=False,
            creationflags=flags,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio decoding timed out") from exc
    except OSError as exc:
        raise ValueError(f"Could not run ffmpeg: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise ValueError(err or "ffmpeg could not decode this audio file")

    raw = proc.stdout
    count = len(raw) // 4
    if count <= 0:
        return []
    try:
        import numpy as np

        return np.frombuffer(raw[: count * 4], dtype="<f4").astype(float).tolist()
    except Exception:
        return list(struct.unpack("<" + "f" * count, raw[: count * 4]))


def sketch_waveform_guide(envelope: Sequence[float], style: str):
    """Plain amplitude sketch so Gemini can follow the real recording. Not the final art."""
    from PIL import Image, ImageDraw

    values = list(envelope) or [0.2]
    if style == "rings":
        size = 1024
        img = Image.new("RGB", (size, size), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx = cy = size / 2
        n = min(240, max(64, len(values)))
        step = len(values) / n
        for i in range(n):
            amp = values[min(len(values) - 1, int(i * step))]
            angle = -math.pi / 2 + (2 * math.pi * i / n)
            inner, outer = 340, 340 + 200 * amp
            x0 = cx + math.cos(angle) * inner
            y0 = cy + math.sin(angle) * inner
            x1 = cx + math.cos(angle) * outer
            y1 = cy + math.sin(angle) * outer
            draw.line([(x0, y0), (x1, y1)], fill=(220, 220, 220), width=2)
        return img

    width, height = 1280, 640
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cy = height / 2
    n = min(width, max(64, len(values)))
    step = len(values) / n
    max_h = height * 0.42
    for i in range(n):
        amp = values[min(len(values) - 1, int(i * step))]
        x = int(i * width / n)
        h = amp * max_h
        draw.line([(x, cy - h), (x, cy + h)], fill=(220, 220, 220), width=2)
    draw.line([(0, cy), (width, cy)], fill=(255, 255, 255), width=2)
    return img

