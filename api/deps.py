import yaml, os
import librosa, numpy as np
from pathlib import Path
from typing import Dict, Any
from speechbrain.inference import EncoderClassifier
import torch

from models.cm_basic import SimpleCM
from dsp.vad import WebRTCVADWrapper

class Settings:
    def __init__(self, d: Dict[str, Any]):   #It creates a structured config object from the YAML file, ensuring consistent parameters everywhere.
        self.db_url: str = d["database"]["url"]
        self.embedder_hf_id: str = d["models"]["embedder_hf_id"]
        self.device: str = d["runtime"].get("device", "cpu")
        self.sample_rate: int = d["audio"].get("sample_rate", 16000)
        self.min_voiced_sec: float = d["policy"]["min_voiced_sec"]
        self.min_snr_db: float = d["policy"]["min_snr_db"]
        self.min_voiced_ratio: float = d["policy"]["min_voiced_ratio"]
        self.sim_high: float = d["policy"]["sim_thresholds"]["high"]
        self.sim_medium: float = d["policy"]["sim_thresholds"]["medium"]
        self.cm_gate: float = d["policy"]["cm_gate"]

def load_settings() -> Settings:
    cfg_path = os.environ.get("VOICEGUARD_CONFIG", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Settings(cfg)

class EmbeddingModel:
    def __init__(self, hf_id: str, device: str):
        self.device = device
        self.model = EncoderClassifier.from_hparams(source=hf_id, run_opts={"device": device})

    @torch.no_grad()
    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        # expects mono float32 [-1,1] @ 16k
        t = torch.tensor(wav_16k, dtype=torch.float32).unsqueeze(0)
        emb = self.model.encode_batch(t).squeeze(0).squeeze(0)
        v = emb.cpu().numpy().astype("float32")
        v /= (np.linalg.norm(v) + 1e-9)
        return v

class Deps:
    def __init__(self):
        self.settings = load_settings()
        self.vad = WebRTCVADWrapper(sr=self.settings.sample_rate, frame_ms=20, mode=3)
        self.embedder = EmbeddingModel(
            hf_id=self.settings.embedder_hf_id,
            device=self.settings.device
        )
        self.cm = SimpleCM()

# Singleton
deps = Deps()
