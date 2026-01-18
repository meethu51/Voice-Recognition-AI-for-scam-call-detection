🎙️ VoiceGuard — Speaker Watchlist System (Sprint A)

VoiceGuard is a demo-ready speaker recognition & watchlist system built with modern speech embeddings and vector search.
Designed for real-time voice risk detection, forensics, and call screening pipelines.

✨ Features

🔊 ECAPA speaker embeddings (SpeechBrain)

🧠 Vector similarity search using pgvector (Postgres)

🛡️ Anti-spoof proxy (placeholder for full CM models)

✂️ VAD-based trimming with audio quality metrics

🌐 REST API built with FastAPI

🧪 Simple web demo for scoring & inspection

👻 Shadow-mode policy

Scores speakers without enforcement

Optional auto-hangup thresholds later

```pgqsl
🧱 System Architecture (High-Level)
Audio Input
   │
   ├─▶ VAD + Quality Filter
   │
   ├─▶ Anti-Spoof Proxy (CM placeholder)
   │
   ├─▶ ECAPA Embedding Extraction
   │
   ├─▶ pgvector Similarity Search
   │
   └─▶ Policy Engine (Shadow / Enforce)
```
📋 Prerequisites

Python 3.10+

Docker (for Postgres + pgvector)

FFmpeg (recommended for audio conversion)

🚀 Quick Start
1️⃣ Clone the repository
```bash
git clone https://github.com/meethu51/voice-recognition-ai.git
cd voice-recognition-ai
```

2️⃣ Start Postgres with pgvector
```bash
docker compose up -d db
```

Initialize the schema:

```bash
psql "postgresql://voice:voice@localhost:5432/voiceguard" -f storage/schema.sql
```

3️⃣ Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
4️⃣ Run the API
```bash
uvicorn api.app:app --reload
```


API: http://localhost:8000

Docs (Swagger): http://localhost:8000/docs

🧪 Scripts
Script	Description
scripts/enroll.py	Enroll a speaker into the watchlist
scripts/score.py	Score an audio file against the DB
scripts/forensic_report.py	Generate JSON/PDF reports
scripts/Check.py	Health & sanity checks
📁 Project Structure
```bash
.
├── api/            # FastAPI app
├── dsp/            # VAD & audio quality metrics
├── models/         # ECAPA + embedding logic
├── storage/        # DB schema & access
├── scripts/        # CLI utilities
├── web/            # Simple demo UI
├── config.example.yaml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

🔐 Configuration

Copy the example config:
```bash
cp config.example.yaml config.yaml
```


⚠️ config.yaml is not tracked by Git — keep secrets safe.

🧠 Design Notes

Shadow-mode by default — no automatic enforcement

Anti-spoofing is a stub (designed for later CM replacement)

Optimized for clarity & extensibility, not benchmark chasing

Intended as a foundation for production-grade voice risk systems

🛣️ Roadmap

 Replace proxy CM with real anti-spoof model

 Streaming audio support

 Threshold calibration tooling

 Multi-tenant watchlists

 Policy-driven enforcement (auto hang-up / alerts)

📄 License

MIT License © 2026 Bhuvan Shrivastava

🙌 Acknowledgements

SpeechBrain — ECAPA models

pgvector — vector similarity in Postgres

FastAPI — clean, fast APIs
