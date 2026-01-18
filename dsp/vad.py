
from __future__ import annotations
import numpy as np

# Optional dependency: WebRTC VAD
try:
    import webrtcvad  # type: ignore
    _HAVE_WEBRTC = True
except Exception:
    _HAVE_WEBRTC = False


def _to_mono_float32(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # normalize to float32 in [-1, 1] (soundfile already returns float in that range)
    return wav.astype(np.float32, copy=False)


def _frame_stride(sr: int, frame_ms: int) -> int:
    return int(round(sr * frame_ms / 1000.0))


# ---------------------------
# Public API
# ---------------------------

def voiced_mask(
    wav: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    aggressiveness: int = 2,
    energy_db: float = -35.0,
    min_active_ms: int = 120,
) -> np.ndarray:
    """
    Returns a boolean mask per *sample* (same length as `wav`) where True = voiced.
    Uses WebRTC VAD if available; otherwise an energy-based fallback.

    Args
    ----
    wav: 1D or 2D numpy array of audio
    sr: sample rate (ideally 16000)
    frame_ms: analysis frame length (10, 20, or 30 ms for WebRTC; 30 ms recommended)
    aggressiveness: 0..3 (only for WebRTC; 3 = most aggressive speech detection)
    energy_db: fallback energy threshold in dBFS (higher -> more frames voiced)
    min_active_ms: min duration for a voiced run; short blips below this are removed
    """
    wav = _to_mono_float32(wav)
    n = len(wav)
    mask = np.zeros(n, dtype=bool)

    hop = _frame_stride(sr, frame_ms)

    if _HAVE_WEBRTC and sr in (8000, 16000, 32000, 48000) and frame_ms in (10, 20, 30):
        # WebRTC branch (frame decisions -> sample mask)
        vad = webrtcvad.Vad(int(aggressiveness))
        # 16-bit PCM required by webrtcvad
        pcm16 = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16, copy=False).tobytes()

        for i in range(0, n - hop + 1, hop):
            chunk = pcm16[2*i:2*(i+hop)]  # 2 bytes per sample
            is_speech = vad.is_speech(chunk, sr)
            if is_speech:
                mask[i:i+hop] = True
    else:
        # Energy VAD fallback
        # Compute RMS per frame and threshold in dBFS
        eps = 1e-9
        for i in range(0, n - hop + 1, hop):
            frame = wav[i:i+hop]
            rms = np.sqrt(np.mean(frame**2) + eps)
            dbfs = 20 * np.log10(rms + eps)  # wav assumed already in [-1, 1]
            if dbfs > energy_db:
                mask[i:i+hop] = True

    # Post-process: remove tiny active blips, fill tiny gaps
    min_len = _frame_stride(sr, min_active_ms)
    mask = _clean_runs(mask, min_len)
    return mask


def voiced_ratio(wav: np.ndarray, sr: int, **kwargs) -> float:
    """Fraction of samples marked voiced."""
    m = voiced_mask(wav, sr, **kwargs)
    return float(np.mean(m)) if len(m) else 0.0


def trim_silence(
    wav: np.ndarray,
    sr: int,
    pad_ms: int = 50,
    **kwargs
) -> np.ndarray:
    """
    Returns a copy of `wav` trimmed to the tightest span that contains voiced audio.
    Adds `pad_ms` of context on both sides when possible.
    """
    wav = _to_mono_float32(wav)
    m = voiced_mask(wav, sr, **kwargs)
    if not m.any():
        return wav  # nothing voiced; return as-is

    idx = np.flatnonzero(m)
    start = int(max(0, idx[0] - _frame_stride(sr, pad_ms)))
    end   = int(min(len(wav), idx[-1] + _frame_stride(sr, pad_ms)))
    return wav[start:end].copy()


def segments(
    wav: np.ndarray,
    sr: int,
    min_seg_ms: int = 200,
    **kwargs
) -> list[tuple[int, int]]:
    """
    Returns a list of (start_sample, end_sample) for each voiced segment longer than `min_seg_ms`.
    """
    m = voiced_mask(wav, sr, **kwargs)
    if not m.any():
        return []

    segs: list[tuple[int, int]] = []
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            j = i + 1
            while j < n and m[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1

    min_len = _frame_stride(sr, min_seg_ms)
    segs = [(s, e) for (s, e) in segs if (e - s) >= min_len]
    return segs


# ---------------------------
# Helpers
# ---------------------------

def _clean_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Remove very short ON runs and fill very short OFF gaps to reduce flicker.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    m = mask.copy()

    # find runs
    n = len(m)
    if n == 0:
        return m

    # Remove short ON runs
    i = 0
    while i < n:
        if m[i]:
            j = i + 1
            while j < n and m[j]:
                j += 1
            if (j - i) < min_len:
                m[i:j] = False
            i = j
        else:
            i += 1

    # Fill short OFF gaps
    i = 0
    while i < n:
        if not m[i]:
            j = i + 1
            while j < n and not m[j]:
                j += 1
            if (j < n) and (i > 0) and ((j - i) < min_len):
                m[i:j] = True
            i = j
        else:
            i += 1

    return m
