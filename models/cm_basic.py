import numpy as np

class SimpleCM:
    # Heuristic-ish baseline; treat low SNR, ultra-short, ultra-flat spectra as suspicious
    def score(self, wav_16k: np.ndarray, snr_db: float, dur: float) -> float:
        dur_ok = np.tanh((dur - 1.0))            # below 1s → low
        snr_ok = np.clip((snr_db - 5)/20, 0, 1)  # <5 dB → low, >25 dB → high
        return float(np.clip(0.2*dur_ok + 0.8*snr_ok, 0.0, 1.0))
