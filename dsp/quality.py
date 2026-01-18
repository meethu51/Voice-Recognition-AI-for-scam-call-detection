import numpy as np

def snr_db(wav):
    sig = np.mean(wav**2)
    # crude noise floor via median abs deviation
    noise = (np.median(np.abs(wav))**2) * 2.0
    return 10*np.log10((sig + 1e-9)/(noise + 1e-9))

def voiced_ratio(vad_flags: np.ndarray) -> float:
    return float(np.mean(vad_flags)) if len(vad_flags) else 0.0

def voiced_ratio_from_wav(wav: np.ndarray, sr: int, frame_ms: int = 20, energy_thresh: float = 1e-3) -> float:
    """
    Quick, dependency-free VAD proxy:
    - Split audio into frames (default 20 ms)
    - Mark a frame 'voiced' if its mean squared energy > threshold
    - Return fraction of voiced frames
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    n = len(wav) // frame_len
    if n == 0:
        return 0.0
    x = wav[: n * frame_len].reshape(n, frame_len)
    energy = (x**2).mean(axis=1)
    flags = (energy > energy_thresh).astype(np.float32)
    return float(flags.mean())