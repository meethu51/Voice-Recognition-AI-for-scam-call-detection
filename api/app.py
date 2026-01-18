import os, yaml, re
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
import numpy as np
import soundfile as sf
from dsp.vad import trim_silence, voiced_ratio #Voice Activity Detection — removes silence, computes how much of the clip is speech.
from models.embedding import get_embedder

from models.cm_basic import SimpleCM #Loads a pretrained voice embedding model (e.g., SpeechBrain / ECAPA-TDNN).
from dsp.quality import snr_db, voiced_ratio_from_wav #Signal quality metrics — computes Signal-to-Noise Ratio (SNR) and voiced ratio again (for consistency).
from storage.db import DB  # our asynchronous PostgreSQL wrapper that handles vector storage & retrieval (using pgvector).
from models.embedding import get_embedder #Loads a pretrained voice embedding model (SpeechBrain / ECAPA-TDNN).

#----test-----
print("### APP BUILD: using enrollment neighbors + cfg thresholds") #console log marker to tell you which build logic the server is running

# ------------ helpers ------------
def l2_normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:  #this normalises vectors like embedding to unit length and ensures cosine similarity works normally(L2 normalized)
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return (v / n).astype("float32")

def mask(url: str) -> str:  #hides the database url password
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:*****@", url)


# ------------ config ------------
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_url = os.getenv("DATABASE_URL", cfg["database"]["url"])
print(">>> Using DB:", mask(db_url))


# ------------ app & db ------------
app = FastAPI() #FastAPI application instance.
db = DB(db_url) #database connection object


# ------------ models ------------
class EnrollResponse(BaseModel):  #returns confirmation and enrollment count.
    label: str
    speaker_id: str
    enroll_count: int

class ScoreResponse(BaseModel):  #returns who the voice most closely matches and how confident the model is.
    top_match_label: str | None
    top_match_score: float | None
    decision: str | None


# ------------ lifecycle ------------
@app.on_event("startup")  #starts db loads speechbrain and initialises Simple CM for quality of audio
async def _startup():
    await db.start()
    global embedder, cm
    embedder = get_embedder(
        device=cfg["runtime"]["device"],  # "cpu" or "cuda"
        target_sr=16000,
        l2_norm=True
    )
    cm = SimpleCM()


@app.on_event("shutdown")
async def _shutdown():
    await db.stop()


# ------------ endpoints ------------
@app.post("/enroll", response_model=EnrollResponse)
async def enroll(label: str = Query(...), file: UploadFile = File(...)):  #reads wav into a numpy array and converts stereo to mono
    wav, sr = sf.read(file.file)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    wav = trim_silence(wav, sr, pad_ms=50, aggressiveness=2)  #Removes silence. Computes how much of the signal is voiced. Extracts embedding where vector representing speaker voice
    vr  = float(voiced_ratio(wav, sr, aggressiveness=2))
    # assume 16 kHz input (resample upstream if needed)
    emb = embedder.embed_wav(wav, sr)
    s = float(snr_db(wav)) #Signal-to-noise ratio (audio clarity). Duration. Countermeasure score (checks for replay/fake audio)
    dur = float(len(wav) / sr)
    vr = float(voiced_ratio_from_wav(wav, sr))
    cm_score = float(cm.score(wav, s, dur))

    # create or fetch speaker row by label
    try:
        speaker_id = await db.upsert_speaker_by_label(label, embedding_dim=len(emb))
    except AttributeError:
        # fallback if your DB wrapper does not have upsert_speaker_by_label
        # can remove this when your DB class has it
        speaker_id = await db.create_speaker(label=label, embedding_dim=len(emb))

    # store enrollment (DB will recompute centroid)
    await db.add_enrollment(
        speaker_id,
        emb,
        meta={
            "clip_path": None,
            "f0_med": None, "f0_iqr": None,
            "snr_db": s, "voiced_ratio": vr,
            "duration_sec": dur, "bandwidth_hz": None,
            "cm_score": cm_score,
            "device_tag": None, "codec_tag": None
        },
    )

    # read back count to return
    speakers = await db.list_speakers()
    count = next((x["enroll_count"] for x in speakers if str(x["speaker_id"]) == str(speaker_id)), 0)
    return EnrollResponse(label=label, speaker_id=str(speaker_id), enroll_count=int(count))


@app.post("/score", response_model=ScoreResponse)
async def score(file: UploadFile = File(...)):
    wav, sr = sf.read(file.file)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    wav = trim_silence(wav, sr, pad_ms=50, aggressiveness=2)
    vr  = float(voiced_ratio(wav, sr, aggressiveness=2))
    emb = embedder.embed_wav(wav, sr)
    s = float(snr_db(wav))
    dur = float(len(wav) / sr)
    vr = float(voiced_ratio_from_wav(wav, sr))
    cm_score = float(cm.score(wav, s, dur))

    # -------- compares to nearest ENROLLMENT vectors --------
    # requires the DB.top_enrollment_neighbors(emb, k_total=30) and to_pgvector_str in DB
    neigh = await db.top_enrollment_neighbors(emb, k_total=30)
    #Retrieves top 30 closest enrolled embeddings using pgvector cosine similarity. Returns a list of tuples (speaker_id, label, similarity).

    # group by label and aggregate (mean of top 3 per label)
    per_label = defaultdict(list)
    for sid, lbl, sim in neigh:
        per_label[lbl].append((sid, sim))

    label_scores: list[tuple[str, float, str | None]] = []
    for lbl, items in per_label.items():
        # sort by sim desc
        items.sort(key=lambda x: x[1], reverse=True)
        top3 = items[:3]
        agg = sum(sim for _, sim in top3) / max(1, len(top3))
        # remember the best speaker_id for this label (for logging if you want)
        best_sid = top3[0][0] if top3 else None
        label_scores.append((lbl, float(agg), str(best_sid) if best_sid else None))

    label_scores.sort(key=lambda x: x[1], reverse=True)

    if label_scores:
        top_label, sim, top_sid = label_scores[0]
    else:
        top_label, sim, top_sid = None, None, None

    # thresholds from config.yaml (you set high=0.70, medium=0.40)
    hi = float(cfg["policy"]["sim_thresholds"]["high"])
    mid = float(cfg["policy"]["sim_thresholds"]["medium"])
    if sim is None:
        decision = "no_match"
    elif sim >= hi:
        decision = "high_match"
    elif sim >= mid:
        decision = "medium_match"
    else:
        decision = "low_match"

    # persist the call
    await db.insert_call({    #Logs this verification attempt into a calls table for auditing
        "clip_path": None,
        "embedding": emb,     # DB handles to_pgvector_str casting
        "cm_score": cm_score,
        "snr_db": s,
        "voiced_ratio": vr,
        "duration_sec": dur,
        "bandwidth_hz": None,
        "top_match_speaker": top_sid,  # optional: best enrollment's speaker_id
        "top_match_score": sim,
        "decision": decision,
        "decision_reason": None
    })

    return ScoreResponse(
        top_match_label=top_label,
        top_match_score=None if sim is None else float(sim),
        decision=decision,
    )
