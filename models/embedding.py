from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

# Your existing ECAPA model
from models.ecapa import ECAPA

# ---- utilities ----

def _to_mono_float32(wav: np.ndarray) -> np.ndarray:
    """Ensure mono float32 in [-1, 1]."""
    wav = np.asarray(wav)
    if wav.ndim > 1:
        # average channels
        wav = wav.mean(axis=1)
    # if integer input, scale to [-1,1]
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        wav = wav.astype(np.float32) / max(1, abs(info.max))
    return wav.astype(np.float32, copy=False)

def _l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v) + eps)
    return (v / n).astype(np.float32)

def _resample_linear(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Lightweight linear resampler (no external deps).
    Assumes wav is 1D float32. Good enough for ECAPA inputs.
    """
    if sr == target_sr:
        return wav
    if len(wav) == 0:
        return wav
    ratio = target_sr / float(sr)
    new_len = int(round(len(wav) * ratio))
    # linear interpolation
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=max(1, new_len), endpoint=False, dtype=np.float64)
    y = np.interp(x_new, x_old, wav.astype(np.float64))
    return y.astype(np.float32, copy=False)

# ---- singleton factory ----

_EMBEDDER_SINGLETON: Optional["Embedder"] = None

def get_embedder(device: str = "cpu",
                 target_sr: int = 16000,
                 l2_norm: bool = True) -> "Embedder":
    """
    Return a process-wide singleton embedder (so the model loads once).
    """
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is None:
        _EMBEDDER_SINGLETON = Embedder(device=device, target_sr=target_sr, l2_norm=l2_norm)
    return _EMBEDDER_SINGLETON

# ---- main class ----

class Embedder:
    """
    Thin wrapper around ECAPA providing:
      • automatic mono + resample to target_sr
      • consistent float32
      • L2-normalized embeddings
      • sliding-window embeddings (for timelines)
    """
    def __init__(self, device: str = "cpu", target_sr: int = 16000, l2_norm: bool = True):
        self.device = device
        self.target_sr = int(target_sr)
        self.l2_norm = bool(l2_norm)
        self.model = ECAPA(device=device)  # your existing class

    # ----- core API -----

    def embed_wav(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """
        Embed a single utterance (np.ndarray), returning float32 vector (L2-normalized).
        """
        y = _to_mono_float32(wav)
        y = _resample_linear(y, sr, self.target_sr)
        vec = self.model.embed(y.astype(np.float32))
        vec = np.asarray(vec, dtype=np.float32)
        return _l2_normalize(vec) if self.l2_norm else vec

    def embed_file(self, path: str, reader=None) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Load a file (using provided reader or soundfile-like), resample, and embed.
        Returns: (mono_wav_16k, sr16k, embedding)
        """
        if reader is None:
            import soundfile as sf
            wav, sr = sf.read(path)
        else:
            wav, sr = reader(path)
        y = _to_mono_float32(wav)
        y = _resample_linear(y, sr, self.target_sr)
        emb = self.model.embed(y.astype(np.float32))
        emb = np.asarray(emb, dtype=np.float32)
        if self.l2_norm:
            emb = _l2_normalize(emb)
        return y, self.target_sr, emb

    def embed_windows(self, wav: np.ndarray, sr: int,
                      win_s: float = 2.0, hop_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window embeddings across a long utterance.
        Returns:
          times_sec: [T] centers of windows,
          embs:      [T, D] embeddings (L2-normalized if enabled)
        """
        y = _to_mono_float32(wav)
        y = _resample_linear(y, sr, self.target_sr)
        win = int(self.target_sr * win_s)
        hop = int(self.target_sr * hop_s)
        if win <= 0 or hop <= 0:
            raise ValueError("win_s and hop_s must be > 0")

        times: List[float] = []
        embs: List[np.ndarray] = []
        i = 0
        while i + win <= len(y):
            seg = y[i:i+win]
            vec = self.model.embed(seg.astype(np.float32))
            vec = np.asarray(vec, dtype=np.float32)
            vec = _l2_normalize(vec) if self.l2_norm else vec
            embs.append(vec)
            # center time of this window
            t = (i + win // 2) / float(self.target_sr)
            times.append(t)
            i += hop

        if not embs:
            # Fallback: embed entire utterance if shorter than window
            vec = self.model.embed(y.astype(np.float32))
            vec = np.asarray(vec, dtype=np.float32)
            vec = _l2_normalize(vec) if self.l2_norm else vec
            embs = [vec]
            times = [len(y) / float(self.target_sr) * 0.5]

        return np.asarray(times, dtype=np.float32), np.vstack(embs).astype(np.float32)

    # ----- convenience -----

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("cosine expects 1D vectors")
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
