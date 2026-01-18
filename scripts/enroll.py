import argparse, uuid, json, soundfile as sf, numpy as np
from models.ecapa import ECAPA
from dsp.quality import snr_db
from models.cm_basic import SimpleCM
# ... write to JSON or DB ...

p = argparse.ArgumentParser()
p.add_argument("wav"); p.add_argument("--label", required=True)
args = p.parse_args()
wav, sr = sf.read(args.wav)
emb = ECAPA().embed(wav.astype("float32"))
print("embedding_norm", np.linalg.norm(emb))
# save enrollment.json for quick start
