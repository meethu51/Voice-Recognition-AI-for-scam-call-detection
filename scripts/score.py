import numpy as np

def cosine(a,b): return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def decision(sim, cm, q):
    # q = dict(snr_db, dur, voiced_ratio, bandwidth_hz)
    # Shadow thresholds — tuned later; keep conservative now
    quality_ok = (q["dur"] >= 2.0) and (q["snr"] >= 10.0) and (q["voiced"] >= 0.4)
    cm_ok = cm >= 0.5
    if sim >= 0.75 and cm_ok and quality_ok:
        return "flag_high"   # shadow: would auto-hangup later
    elif sim >= 0.65 and cm_ok:
        return "flag_review"
    else:
        return "allow"
