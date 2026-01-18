import torch, numpy as np
from speechbrain.inference import EncoderClassifier

class ECAPA:
    def __init__(self, device="cpu", hf_id="speechbrain/spkrec-ecapa-voxceleb"):
        self.device = device
        self.model = EncoderClassifier.from_hparams(source=hf_id, run_opts={"device": device})

    @torch.no_grad()
    def embed(self, wav_16k: np.ndarray) -> np.ndarray:
        # expects mono 16 kHz float32 [-1,1]
        emb = self.model.encode_batch(torch.tensor(wav_16k).unsqueeze(0)).squeeze(0).squeeze(0)
        v = emb.cpu().numpy().astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        return v
