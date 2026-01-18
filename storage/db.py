import asyncpg, uuid, numpy as np
from typing import Optional, List, Tuple, Dict, Any

def to_pg_vector(v: np.ndarray) -> list:
    return [float(x) for x in v.tolist()]

def to_pgvector_str(v: np.ndarray) -> str:
    # Format like: [0.1,0.2,...] which pgvector accepts, then cast with ::vector
    return "[" + ",".join(f"{float(x):.7g}" for x in v.tolist()) + "]"

def from_pgvector(val) -> np.ndarray:
    """
    Convert a pgvector value returned by asyncpg into a np.float32 array.
    Handles both list-like and text formats like "[0.1,0.2,...]".
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.asarray(val, dtype=np.float32)

    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return np.array([], dtype=np.float32)
        return np.fromstring(s, sep=",", dtype=np.float32)

    raise TypeError(f"Unsupported pgvector value type: {type(val)}")

class DB:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[asyncpg.Pool] = None

    async def start(self):
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=10)

    async def stop(self):
        if self.pool:
            await self.pool.close()

    # ---------- Speakers ----------
    async def create_speaker(self, label: str, embedding_dim: int = 192, notes: str = "") -> uuid.UUID:
        q = """
        INSERT INTO speakers (speaker_id, label, notes, embedding_dim, centroid, enroll_count)
        VALUES ($1, $2, $3, $4, NULL, 0) RETURNING speaker_id;
        """
        sid = uuid.uuid4()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(q, sid, label, notes, embedding_dim)
        return row["speaker_id"]

    async def upsert_speaker_by_label(self, label: str, embedding_dim: int = 192) -> uuid.UUID:
        """Create speaker if label not present, else return existing id."""
        # Requires UNIQUE(label) on speakers (we added that).
        q = """
        INSERT INTO speakers (speaker_id, label, embedding_dim)
        VALUES ($1, $2, $3)
        ON CONFLICT (label) DO UPDATE SET embedding_dim = EXCLUDED.embedding_dim
        RETURNING speaker_id;
        """
        sid = uuid.uuid4()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(q, sid, label, embedding_dim)
        return row["speaker_id"]

    async def list_speakers(self) -> List[Dict[str, Any]]:
        q = "SELECT speaker_id, label, enroll_count, created_at FROM speakers ORDER BY created_at DESC;"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q)
        return [dict(r) for r in rows]

    async def get_speaker(self, speaker_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        q = "SELECT * FROM speakers WHERE speaker_id=$1;"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(q, speaker_id)
        return dict(row) if row else None

    async def update_centroid(self, speaker_id: uuid.UUID):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT embedding FROM enrollments WHERE speaker_id=$1;", speaker_id
            )
            n = len(rows)
            if n == 0:
                await conn.execute(
                    "UPDATE speakers SET centroid = NULL, enroll_count = 0 WHERE speaker_id=$1;",
                    speaker_id,
                )
                return

            vecs = [from_pgvector(r["embedding"]) for r in rows]
            mean_vec = np.mean(vecs, axis=0).astype("float32")

            # IMPORTANT: normalize the centroid
            nrm = np.linalg.norm(mean_vec) + 1e-9
            cen = (mean_vec / nrm).astype("float32")

            cen_txt = to_pgvector_str(cen)
            await conn.execute(
                "UPDATE speakers SET centroid = $1::vector, enroll_count = $2 WHERE speaker_id=$3;",
                cen_txt, n, speaker_id
            )

    # ---------- Enrollments ----------
    async def add_enrollment(self,
                             speaker_id: uuid.UUID,
                             embedding: np.ndarray,
                             meta: Dict[str, Any]) -> uuid.UUID:
        e_id = uuid.uuid4()
        q = """
        INSERT INTO enrollments (
          enrollment_id, speaker_id, clip_path, embedding,
          f0_med, f0_iqr, snr_db, voiced_ratio, duration_sec, bandwidth_hz, cm_score,
          device_tag, codec_tag
        ) VALUES (
          $1, $2, $3, $4,
          $5, $6, $7, $8, $9, $10, $11,
          $12, $13
        );
        """
        emb = to_pg_vector(embedding)
        async with self.pool.acquire() as conn:
            await conn.execute(
                q, e_id, speaker_id,
                meta.get("clip_path"), emb,
                meta.get("f0_med"), meta.get("f0_iqr"),
                meta.get("snr_db"), meta.get("voiced_ratio"),
                meta.get("duration_sec"), meta.get("bandwidth_hz"),
                meta.get("cm_score"),
                meta.get("device_tag"), meta.get("codec_tag"),
            )
        await self.update_centroid(speaker_id)
        return e_id

    async def add_enrollment(self,
                             speaker_id: uuid.UUID,
                             embedding: np.ndarray,
                             meta: Dict[str, Any]) -> uuid.UUID:
        e_id = uuid.uuid4()
        q = """
        INSERT INTO enrollments (
          enrollment_id, speaker_id, clip_path, embedding,
          f0_med, f0_iqr, snr_db, voiced_ratio, duration_sec, bandwidth_hz, cm_score,
          device_tag, codec_tag
        ) VALUES (
          $1, $2, $3, $4::vector,
          $5, $6, $7, $8, $9, $10, $11,
          $12, $13
        );
        """
        emb_txt = to_pgvector_str(embedding)   # <-- convert to text literal
        async with self.pool.acquire() as conn:
            await conn.execute(
                q, e_id, speaker_id,
                meta.get("clip_path"), emb_txt,        # <-- pass string here
                meta.get("f0_med"), meta.get("f0_iqr"),
                meta.get("snr_db"), meta.get("voiced_ratio"),
                meta.get("duration_sec"), meta.get("bandwidth_hz"),
                meta.get("cm_score"),
                meta.get("device_tag"), meta.get("codec_tag"),
            )
        await self.update_centroid(speaker_id)
        return e_id

    async def insert_call(self, rec: Dict[str, Any]) -> uuid.UUID:
        cid = uuid.uuid4()
        q = """
        INSERT INTO calls (
          call_id, clip_path, embedding, cm_score, snr_db, voiced_ratio,
          duration_sec, bandwidth_hz, top_match_speaker, top_match_score, decision, decision_reason
        ) VALUES (
          $1, $2, $3::vector, $4, $5, $6,
          $7, $8, $9, $10, $11, $12
        );
        """
        emb_txt = None if rec.get("embedding") is None else to_pgvector_str(rec["embedding"])
        async with self.pool.acquire() as conn:
            await conn.execute(
                q, cid, rec.get("clip_path"), emb_txt,   # <-- string or None
                rec.get("cm_score"), rec.get("snr_db"), rec.get("voiced_ratio"),
                rec.get("duration_sec"), rec.get("bandwidth_hz"),
                rec.get("top_match_speaker"), rec.get("top_match_score"),
                rec.get("decision"), rec.get("decision_reason")
            )
        return cid

    # ---------- Nearest neighbors ----------
    async def topk_speakers_by_centroid(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[uuid.UUID, str, float]]:
        """
        Returns (speaker_id, label, cosine_sim) sorted by nearest centroid.
        """
        q = """
        SELECT speaker_id, label,
            1.0 - (centroid <-> $1::vector) AS cosine_sim
        FROM speakers
        WHERE centroid IS NOT NULL
        ORDER BY centroid <-> $1::vector
        LIMIT $2;
        """
        # pass as text literal like "[0.1,0.2,...]" and cast to ::vector in SQL
        emb_txt = to_pgvector_str(query_vec)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, emb_txt, k)
        return [(r["speaker_id"], r["label"], float(r["cosine_sim"])) for r in rows]
    
    async def top_enrollment_neighbors(self, query_vec: np.ndarray, k_total: int = 20):
        """
        Return the top-k enrollment neighbors of the query embedding,
        joined with their labels. We pass the embedding as pgvector text.
        """
        q = """
        SELECT e.speaker_id, s.label,
            1.0 - (e.embedding <-> $1::vector) AS cosine_sim
        FROM enrollments e
        JOIN speakers s ON s.speaker_id = e.speaker_id
        ORDER BY e.embedding <-> $1::vector
        LIMIT $2;
        """
        vec_txt = to_pgvector_str(query_vec)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(q, vec_txt, k_total)
        # list of (speaker_id, label, sim)
        return [(r["speaker_id"], r["label"], float(r["cosine_sim"])) for r in rows]
        
    # ---------- Calls ----------
    async def insert_call(self, rec: Dict[str, Any]) -> uuid.UUID:
        cid = uuid.uuid4()
        q = """
        INSERT INTO calls (
        call_id, clip_path, embedding, cm_score, snr_db, voiced_ratio,
        duration_sec, bandwidth_hz, top_match_speaker, top_match_score, decision, decision_reason
        ) VALUES (
        $1, $2, $3::vector, $4, $5, $6,
        $7, $8, $9, $10, $11, $12
        );
        """
        emb_txt = None if rec.get("embedding") is None else to_pgvector_str(rec["embedding"])
        async with self.pool.acquire() as conn:
            await conn.execute(
                q,
                cid,
                rec.get("clip_path"),
                emb_txt,                         # <-- must be string or None
                rec.get("cm_score"),
                rec.get("snr_db"),
                rec.get("voiced_ratio"),
                rec.get("duration_sec"),
                rec.get("bandwidth_hz"),
                rec.get("top_match_speaker"),
                rec.get("top_match_score"),
                rec.get("decision"),
                rec.get("decision_reason"),
            )
        return cid
