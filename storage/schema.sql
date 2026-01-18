-- Enable extension (once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- We’ll use cosine distance, so store normalized vectors and index with ivfflat (cosine ops)

-- Speakers “watchlist” table
CREATE TABLE IF NOT EXISTS speakers (
  speaker_id UUID PRIMARY KEY,
  label TEXT NOT NULL,                       -- e.g. "scam_profile_001"
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Embedding info
  embedding_dim INT NOT NULL DEFAULT 192,    -- ECAPA default = 192; adjust to your model
  centroid VECTOR(192),                      -- centroid kept L2-normalized

  -- Quality rollups (optional)
  enroll_count INT NOT NULL DEFAULT 0,
  mean_f0_med REAL,
  mean_f0_iqr REAL
);

-- Enrollments (raw examples per speaker)
CREATE TABLE IF NOT EXISTS enrollments (
  enrollment_id UUID PRIMARY KEY,
  speaker_id UUID NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  clip_path TEXT,
  embedding VECTOR(192) NOT NULL,            -- L2-normalized per-insert
  f0_med REAL, f0_iqr REAL,
  snr_db REAL, voiced_ratio REAL,
  duration_sec REAL, bandwidth_hz INT,
  cm_score REAL,

  -- For auditing
  device_tag TEXT, codec_tag TEXT
);

-- Calls scored against the watchlist
CREATE TABLE IF NOT EXISTS calls (
  call_id UUID PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  clip_path TEXT,
  embedding VECTOR(192),                     -- may be NULL on failure
  cm_score REAL,
  snr_db REAL, voiced_ratio REAL,
  duration_sec REAL, bandwidth_hz INT,

  -- Top-1 nearest neighbor at scoring time
  top_match_speaker UUID REFERENCES speakers(speaker_id),
  top_match_score REAL,                      -- cosine similarity in [-1..1]

  -- Policy decision at scoring time (shadow or active)
  decision TEXT,
  decision_reason TEXT
);

-- Helpful: materialized (or plain) view for quick analytics
CREATE OR REPLACE VIEW speaker_overview AS
SELECT
  s.speaker_id,
  s.label,
  s.enroll_count,
  s.created_at,
  s.mean_f0_med,
  s.mean_f0_iqr
FROM speakers s;

-- Indexes (pgvector)
-- NOTE: ivfflat requires ANALYZE and is only used when lists > 0 and index is built.
-- Choose lists ~ (#rows / 10) as a starting point; tune later.
CREATE INDEX IF NOT EXISTS idx_enrollments_embedding_ivfflat
ON enrollments USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_speakers_centroid_ivfflat
ON speakers USING ivfflat (centroid vector_cosine_ops) WITH (lists = 10);

-- Also keep a small btree on label for quick lookups
CREATE INDEX IF NOT EXISTS idx_speakers_label ON speakers (label);
