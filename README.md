# VoiceGuard (Sprint A)

A demo-ready speaker watchlist system:
- ECAPA embeddings (SpeechBrain), stored in **pgvector**.
- Simple anti-spoof proxy (replace with real CM later).
- VAD-based trimming and quality metrics.
- REST API (FastAPI) + simple web demo.
- Shadow-mode policy (optionally enable auto-hangup threshold later).

## 1. Prereqs

- Python 3.10+
- Docker (for Postgres with pgvector)
- FFmpeg (recommended for audio conversions)

## 2. Start Postgres with pgvector

```bash
docker compose up -d db
psql "postgresql://voice:voice@localhost:5432/voiceguard" -f storage/schema.sql
