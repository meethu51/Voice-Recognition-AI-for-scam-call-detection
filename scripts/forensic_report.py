from __future__ import annotations
import os, sys, json, hashlib, argparse, uuid, re, importlib, platform, getpass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import stft
import torch, torchcrepe
from datetime import datetime, timezone

import yaml
from models.embedding import get_embedder
from models.cm_basic import SimpleCM
from dsp.vad import trim_silence, voiced_ratio
from dsp.quality import snr_db
from storage.db import DB, to_pgvector_str

# crisp fonts in PDF/PS
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ----------------- Helpers -----------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def l2n(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return (v / (np.linalg.norm(v) + eps)).astype(np.float32)

def f0_track(y: np.ndarray, sr: int):
    """F0 via torchcrepe (librosa-free)."""
    if len(y) < int(sr * 0.1):
        return np.array([]), np.array([]), None, None
    if sr != 16000:
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=int(round(len(y) * 16000 / sr)), endpoint=False)
        y16 = np.interp(x_new, x_old, y.astype(np.float32)).astype(np.float32)
        sr = 16000
    else:
        y16 = y.astype(np.float32, copy=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = torch.from_numpy(y16).unsqueeze(0).to(device)
    f0 = torchcrepe.predict(
        audio, sr, hop_length=160, fmin=50.0, fmax=500.0,
        model="full", batch_size=2048, device=device, return_periodicity=False
    ).squeeze(0).detach().cpu().numpy()
    t = np.arange(len(f0)) * (160.0 / sr)

    f0_masked = f0.astype(float)
    f0_masked[f0_masked <= 0] = np.nan
    f0v = f0_masked[np.isfinite(f0_masked)]
    med = float(np.nanmedian(f0v)) if f0v.size else None
    iqr = float(np.nanpercentile(f0v, 75) - np.nanpercentile(f0v, 25)) if f0v.size else None
    return t, f0_masked, med, iqr

def sliding_cos(embedder, y: np.ndarray, sr: int, ref_vec: np.ndarray,
                win_s: float = 2.0, hop_s: float = 1.0):
    t_sec, embs = embedder.embed_windows(y, sr, win_s=win_s, hop_s=hop_s)
    sims = embs @ ref_vec
    return t_sec, sims.astype(np.float32)

def load_cfg():
    with open("config.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def file_metadata(path: Path, wav: np.ndarray, sr: int) -> dict:
    nbytes = path.stat().st_size if path.exists() else None
    channels = 1 if wav.ndim == 1 else wav.shape[1]
    bitdepth = 16
    return {
        "filename": path.name,
        "bytes": nbytes,
        "orig_sr": sr,
        "channels": channels,
        "bit_depth": bitdepth,
        "duration_s": float(len(wav) / sr),
        "sha256": sha256(path),
    }

def env_versions() -> dict:
    def ver(mod):
        try: return importlib.import_module(mod).__version__
        except Exception: return None
    return {
        "python": platform.python_version(),
        "numpy": ver("numpy"),
        "torch": ver("torch"),
        "torchaudio": ver("torchaudio"),
        "speechbrain": ver("speechbrain"),
        "torchcrepe": ver("torchcrepe"),
        "os": f"{platform.system()} {platform.release()}",
        "hostname": platform.node(),
        "user": getpass.getuser(),
    }

def methods_text(cfg, vad_cfg) -> str:
    hi = float(cfg["policy"]["sim_thresholds"]["high"])
    mid = float(cfg["policy"]["sim_thresholds"]["medium"])
    vad_mode = int(vad_cfg.get("aggressiveness", 2))
    pad_ms  = int(vad_cfg.get("pad_ms", 50))
    return (
        "Speech isolation (VAD): WebRTC VAD, 20 ms frames, mode "
        f"{vad_mode}, silence trimmed with ±{pad_ms} ms padding. "
        "Voiced ratio = fraction of voiced frames.\n\n"
        "Embedding extraction: ECAPA (speechbrain/spkrec-ecapa-voxceleb) at 16 kHz mono; "
        "embeddings L2-normalized.\n\n"
        "Similarity scoring: primary = cosine to each enrollment for the label; report "
        "best and mean(top-3). Secondary = cosine to label centroid (mean of enrollments).\n\n"
        f"Decision policy: thresholds medium={mid:.2f}, high={hi:.2f} → "
        "label as low/medium/high match.\n\n"
        "Anti-spoofing (optional): CM score is reported if available."
    )

def boxed_text(ax, title: str, body: str, fontsize=9):
    ax.axis("off")
    if title:
        ax.text(0.01, 0.98, title, fontsize=11, weight="bold", va="top")
    ax.text(
        0.01, 0.90, body, fontsize=fontsize, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95)
    )

def build_stats_text(parts: list[str], max_chars: int = 100) -> str:
    """Pack tokens into wrapped lines with ' | ' separators."""
    lines, cur = [], ""
    sep = "  |  "
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + len(sep) + len(p) <= max_chars:
            cur += sep + p
        else:
            lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Path to query WAV")
    ap.add_argument("--label", required=True, help="Watchlist label to compare against")
    ap.add_argument("--topn", type=int, default=10, help="Top-N enrollments to list")
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--win", type=float, default=2.0, help="Sliding window size (s)")
    ap.add_argument("--hop", type=float, default=1.0, help="Sliding hop (s)")
    args = ap.parse_args()

    cfg  = load_cfg()
    vcfg = cfg.get("vad", {})
    hi   = float(cfg["policy"]["sim_thresholds"]["high"])
    mid  = float(cfg["policy"]["sim_thresholds"]["medium"])

    # Audio
    qpath = Path(args.query)
    qwav, qsr = sf.read(qpath)
    if qwav.ndim > 1:
        qwav = qwav.mean(axis=1)
    qwav_raw, qsr_raw = qwav.copy(), int(qsr)

    meta_file   = file_metadata(qpath, qwav_raw, qsr_raw)
    vers        = env_versions()
    methods_str = methods_text(cfg, vcfg)

    # VAD + quality
    qwav = trim_silence(
        qwav, qsr,
        pad_ms=int(vcfg.get("pad_ms", 50)),
        aggressiveness=int(vcfg.get("aggressiveness", 2))
    )
    qvr  = float(voiced_ratio(qwav, qsr, aggressiveness=int(vcfg.get("aggressiveness", 2))))
    qsnr = float(snr_db(qwav))
    qdur = float(len(qwav) / qsr)

    # Models
    embedder = get_embedder(device=cfg["runtime"]["device"], target_sr=16000, l2_norm=True)
    cm = SimpleCM()

    # Embedding & CM
    qemb = embedder.embed_wav(qwav, qsr)
    qcm  = float(cm.score(qwav, qsnr, qdur))

    # Pitch
    t_f0, f0, f0_med, f0_iqr = f0_track(qwav, qsr)

    # DB pulls
    db_url = os.getenv("DATABASE_URL", cfg["database"]["url"])
    db = DB(db_url)
    import asyncio
    async def _fetch():
        await db.start()
        centroid_vec = None
        row = await db.pool.fetchrow("SELECT centroid FROM speakers WHERE label=$1;", args.label)
        if row and row["centroid"] is not None:
            from storage.db import from_pgvector
            try:
                cen = from_pgvector(row["centroid"])
            except Exception:
                s = str(row["centroid"]).strip().strip("[]")
                cen = np.fromstring(s, sep=",", dtype=np.float32)
            centroid_vec = l2n(np.asarray(cen, np.float32))

        rows = await db.pool.fetch(
            """
            SELECT e.enrollment_id, e.created_at,
                   1.0 - (e.embedding <-> $1::vector) AS cosine_sim
            FROM enrollments e
            JOIN speakers s ON s.speaker_id = e.speaker_id
            WHERE s.label=$2
            ORDER BY e.embedding <-> $1::vector
            LIMIT $3;
            """,
            to_pgvector_str(qemb), args.label, max(30, args.topn)
        )
        await db.stop()
        return centroid_vec, rows

    centroid_vec, rows = asyncio.run(_fetch())
    enr_scores = [{
        "enrollment_id": str(r["enrollment_id"])[-8:],
        "created_at": "" if r["created_at"] is None else str(r["created_at"])[:19],
        "cosine": float(r["cosine_sim"])
    } for r in rows]
    enr_scores.sort(key=lambda x: x["cosine"], reverse=True)
    topN_for_table = enr_scores[:args.topn]

    # Global scores
    sim_centroid        = float(qemb @ centroid_vec) if centroid_vec is not None else None
    sims_all            = [x["cosine"] for x in enr_scores]
    sim_best_enrollment = float(sims_all[0]) if sims_all else None
    sim_top3_mean       = float(np.mean(sims_all[:3])) if len(sims_all) >= 1 else None

    # Sliding-window timeline
    ref_vec = centroid_vec if centroid_vec is not None else None
    t_sim, s_sim = (np.array([]), np.array([]))
    if ref_vec is not None:
        t_sim, s_sim = sliding_cos(embedder, qwav, qsr, ref_vec, win_s=args.win, hop_s=args.hop)

    # Segments from timeline
    seg_rows = []
    if len(t_sim) > 0:
        flags = np.zeros_like(s_sim, dtype=int)  # 0=low,1=med,2=high
        flags[s_sim >= mid] = 1
        flags[s_sim >= hi]  = 2
        win = args.win
        starts, ends, kinds, cur = [], [], [], None
        for i, (tc, fl) in enumerate(zip(t_sim, flags)):
            if fl > 0 and cur is None:
                cur = {"start": tc - win/2, "kind": fl}
            elif fl == 0 and cur is not None:
                cur["end"] = t_sim[i-1] + win/2
                starts.append(cur["start"]); ends.append(cur["end"]); kinds.append(cur["kind"])
                cur = None
        if cur is not None:
            cur["end"] = t_sim[-1] + win/2
            starts.append(cur["start"]); ends.append(cur["end"]); kinds.append(cur["kind"])
        qdur = float(len(qwav) / qsr)  # ensure up to date
        for st, en, kd in zip(starts, ends, kinds):
            st = max(0.0, float(st)); en = min(float(qdur), float(en))
            dur = en - st
            if dur < 0.20:
                continue
            mask = (t_sim >= st) & (t_sim <= en)
            seg_cos = float(np.mean(s_sim[mask])) if mask.any() else float("nan")
            seg_rows.append({
                "start_s": st, "end_s": en, "dur_s": dur,
                "cosine": seg_cos,
                "flag": "HIGH" if kd == 2 else "MED"
            })

    # JSON side-car
    now_utc = datetime.now(timezone.utc)
    rid = f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pdf_path  = outdir / f"{rid}.pdf"
    json_path = outdir / f"{rid}.json"

    decision = (
        "no_match" if sim_best_enrollment is None else
        ("high_match" if sim_best_enrollment >= hi else
         ("medium_match" if sim_best_enrollment >= mid else "low_match"))
    )
    payload = {
        "report_id": rid,
        "created_utc": now_utc.isoformat(),
        "inputs": {"query_path": str(qpath), "query_sha256": sha256(qpath), "orig_sr": int(qsr)},
        "processing": {
            "resample_hz": 16000,
            "vad": {
                "method": "webrtcvad_or_energy",
                "aggressiveness": int(vcfg.get("aggressiveness", 2)),
                "pad_ms": int(vcfg.get("pad_ms", 50)),
                "voiced_ratio": qvr,
                "voiced_duration_sec": qdur
            }
        },
        "model": {"embedder": "speechbrain/spkrec-ecapa-voxceleb", "device": cfg["runtime"]["device"]},
        "thresholds": {"high": hi, "medium": mid},
        "scores": {
            "cosine_centroid": sim_centroid,
            "cosine_best_enrollment": sim_best_enrollment,
            "cosine_top3_mean": sim_top3_mean,
            "cm_score": qcm,
            "snr_db": qsnr,
            "f0_median_hz": f0_med,
            "f0_iqr_hz": f0_iqr
        },
        "timeline": {
            "win_s": args.win, "hop_s": args.hop,
            "t_sec": t_sim.tolist() if len(t_sim) else [],
            "cosine": s_sim.tolist() if len(s_sim) else []
        },
        "segments": seg_rows,
        "top_enrollments": topN_for_table,
        "decision": decision,
        "label": args.label,
        "provenance": {"file": meta_file, "environment": vers},
        "methods": methods_text(cfg, vcfg),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ----------------- PDF (two pages, A4 portrait) -----------------
    outdir.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # ---------- Page 1 ----------
        plt.close("all")
        fig1 = plt.figure(
            figsize=(8.27, 11.69),  # A4 portrait (inches)
            dpi=150,
            constrained_layout=True,
        )
        gs1 = GridSpec(
            10, 1, figure=fig1,
            height_ratios=[
                0.55,  # title
                1.20,  # stats band
                1.25,  # waveform
                1.55,  # spectrogram
                1.20,  # similarity timeline
                1.10,  # pitch
                1.20,  # per-segment title
                0.90,  # per-segment table
                0.35,  # footer
                0.10,  # spacer
            ],
        )

        # Title
        ax_title = fig1.add_subplot(gs1[0, 0]); ax_title.axis("off")
        ax_title.text(0.0, 0.95,
                      f"Forensic Voice Report — label: {args.label}",
                      fontsize=16, weight="bold", va="top")

        # Stats band (auto-wrapped)
        ax_stats = fig1.add_subplot(gs1[1, 0]); ax_stats.axis("off")

        def _fmt2(x):
            return "NA" if x is None else f"{x:.2f}"

        stats_parts = [
            f"Report ID: {rid}",
            f"File: {meta_file['filename']}",
            f"SHA256: {meta_file['sha256'][:12]}…",
            f"Voiced: {qdur:.2f}s (VR={qvr:.2f})",
            f"SNR: {qsnr:.1f} dB",
            f"Scores — Best: {_fmt2(sim_best_enrollment)}",
            f"Top3 mean: {_fmt2(sim_top3_mean)}",
            f"Centroid: {_fmt2(sim_centroid)}",
            f"Decision: {decision} (thr: med={mid:.2f}, high={hi:.2f})",
            f"CM: {qcm:.2f}",
            f"F0 median: {_fmt2(f0_med)} Hz (IQR {_fmt2(f0_iqr)} Hz)",
        ]
        stats_text = build_stats_text(stats_parts, max_chars=110)
        ax_stats.text(
            0.01, 0.95, stats_text,
            fontsize=8, family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95),
        )

        # Waveform
        axW = fig1.add_subplot(gs1[2, 0])
        tt = np.arange(len(qwav))/qsr
        axW.plot(tt, qwav, linewidth=0.7)
        axW.set_title("Waveform (trimmed to voiced regions)", fontsize=11)
        axW.set_xlabel("Time (s)"); axW.set_ylabel("Amplitude")

        # Spectrogram
        axS = fig1.add_subplot(gs1[3, 0])
        f, ttt, Z = stft(qwav, fs=qsr, nperseg=512, noverlap=256)
        P = 20*np.log10(np.abs(Z) + 1e-6)
        axS.pcolormesh(ttt, f, P, shading="auto", vmin=-80, vmax=-20)
        axS.set_ylim(0, 4000); axS.set_ylabel("Hz"); axS.set_xlabel("Time (s)")
        axS.set_title("Spectrogram (dB, 0–4 kHz)", fontsize=11)

        # Similarity timeline
        axT = fig1.add_subplot(gs1[4, 0])
        if len(t_sim):
            axT.plot(t_sim, s_sim, linewidth=1.0)
            axT.axhline(mid, linestyle="--", linewidth=0.8)
            axT.axhline(hi,  linestyle="--", linewidth=0.8)
            axT.set_ylim(min(-0.1, float(np.min(s_sim)) - 0.05), 1.05)
            axT.set_title("Sliding-window cosine vs reference (centroid)", fontsize=11)
            axT.set_ylabel("Cosine"); axT.set_xlabel("Time (s)")
            axT.text(0.01, 0.88, f"win={args.win:.1f}s  hop={args.hop:.1f}s",
                     transform=axT.transAxes, fontsize=8)
        else:
            axT.text(0.5, 0.5, "No reference timeline (centroid unavailable).",
                     ha="center", va="center")
            axT.set_axis_off()

        # Pitch
        axP = fig1.add_subplot(gs1[5, 0])
        if len(t_f0):
            axP.plot(t_f0, f0, linewidth=0.8)
            axP.set_ylim(50, 500); axP.set_yscale('log')
            axP.set_title("Pitch track (F0)", fontsize=11)
            axP.set_ylabel("Hz"); axP.set_xlabel("Time (s)")
        else:
            axP.text(0.5, 0.5, "F0 unavailable", ha='center', va='center')
            axP.set_axis_off()

        # Per-segment table title
        axSegTitle = fig1.add_subplot(gs1[6, 0]); axSegTitle.axis("off")
        axSegTitle.text(0.0, 0.9, "Per-segment (voiced ≥ 0.2 s)",
                        fontsize=11, weight="bold", va="top")

        # Per-segment table
        axSegTbl = fig1.add_subplot(gs1[7, 0]); axSegTbl.axis("off")
        if seg_rows:
            cols = ["Start (s)", "End (s)", "Dur (s)", "Cosine", "Flag"]
            cell = [[f"{r['start_s']:.2f}", f"{r['end_s']:.2f}", f"{r['dur_s']:.2f}",
                     f"{r['cosine']:.2f}", r['flag']] for r in seg_rows]
            table = axSegTbl.table(cellText=cell, colLabels=cols, loc='center')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
        else:
            axSegTbl.text(0.5, 0.5, "No segments ≥ 0.2 s", ha='center', va='center')

        # Footer
        axF = fig1.add_subplot(gs1[8, 0]); axF.axis("off")
        axF.text(0.0, 0.6,
                 "See next page for Top Enrollments and Full Provenance / Methods.",
                 fontsize=9, style="italic")

        pdf.savefig(fig1)   # page 1

        # ---------- Page 2 ----------
        
        fig2 = plt.figure(
            figsize=(8.27, 11.69),  # A4 portrait
            dpi=150,
            constrained_layout=True,
        )

        # Single-column layout; boxes stacked vertically
        gs2 = GridSpec(
            8, 1, figure=fig2,
            height_ratios=[
                0.70,  # page title
                0.35,  # "Top enrollments" section title
                2.70,  # enrollments table (full width)
                0.40,  # "Provenance & Methods" section title
                1.60,  # Provenance box
                1.60,  # Methods box
                0.10,  # spacer
                0.10,  # spacer
            ],
        )

        # Page title
        ax2_title = fig2.add_subplot(gs2[0, 0]); ax2_title.axis("off")
        ax2_title.text(0.0, 0.95, "Top Enrollments and Provenance",
                    fontsize=16, weight="bold", va="top")

        # Section title (enrollments)
        ax2_enr_title = fig2.add_subplot(gs2[1, 0]); ax2_enr_title.axis("off")
        ax2_enr_title.text(0.0, 0.9, "Top enrollments by cosine",
                        fontsize=12, weight="bold", va="top")

        # Enrollments table (full width, single column)
        ax_enr = fig2.add_subplot(gs2[2, 0]); ax_enr.axis("off")
        if topN_for_table:
            cols = ["Enroll ID", "Created", "Cosine"]
            cell = [[r["enrollment_id"], r.get("created_at", ""), f"{r['cosine']:.2f}"]
                    for r in topN_for_table]
            table = ax_enr.table(cellText=cell, colLabels=cols, loc='center')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
        else:
            ax_enr.text(0.5, 0.5, "No enrollments", ha='center', va='center')

        # Section title (provenance & methods)
        ax2_boxes_title = fig2.add_subplot(gs2[3, 0]); ax2_boxes_title.axis("off")
        ax2_boxes_title.text(0.0, 0.9, "Provenance & Methods",
                            fontsize=12, weight="bold", va="top")

        # Build provenance text (wrap to avoid horizontal overflow)
        from textwrap import fill
        prov_txt = (
            f"File: {meta_file['filename']}\n"
            f"Bytes: {meta_file['bytes']} | SHA256: {meta_file['sha256']}\n"
            f"Orig sample rate: {meta_file['orig_sr']} Hz | Channels: {meta_file['channels']} "
            f"| Bit depth: {meta_file['bit_depth']}\n"
            f"Duration: {meta_file['duration_s']:.2f} s\n\n"
            f"System: {vers['os']} | Host: {vers['hostname']} | User: {vers['user']}\n"
            f"Python: {vers['python']} | numpy: {vers['numpy']} | torch: {vers['torch']} | "
            f"torchaudio: {vers['torchaudio']}\n"
            f"speechbrain: {vers['speechbrain']} | torchcrepe: {vers['torchcrepe']}\n\n"
            f"Config snapshot: thresholds (med={mid:.2f}, high={hi:.2f}); "
            f"VAD mode={int(vcfg.get('aggressiveness',2))}, pad_ms={int(vcfg.get('pad_ms',50))}; "
            "scoring: best & mean(top-3) vs enrollments; centroid cosine."
        )
        prov_txt = fill(prov_txt, width=110)

        # Provenance box (row 4)
        ax_prov = fig2.add_subplot(gs2[4, 0]); ax_prov.axis("off")
        ax_prov.text(
            0.02, 0.98, prov_txt, va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95)
        )

        # Methods box (row 5) – also wrapped
        methods_wrapped = fill(methods_str, width=110)
        ax_meth = fig2.add_subplot(gs2[5, 0]); ax_meth.axis("off")
        ax_meth.text(
            0.02, 0.98, methods_wrapped, va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95)
        )

        pdf.savefig(fig2)  # page 2


    print(f"Saved:\n  {pdf_path}\n  {json_path}")

if __name__ == "__main__":
    main()
