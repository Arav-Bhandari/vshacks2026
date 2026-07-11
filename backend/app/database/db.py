import json
import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    nct_id TEXT PRIMARY KEY,
    title TEXT,
    status TEXT,
    phase TEXT,
    conditions TEXT,
    interventions TEXT,
    primary_outcomes TEXT,
    secondary_outcomes TEXT,
    enrollment INTEGER,
    start_date TEXT,
    completion_date TEXT,
    duration_months REAL,
    allocation TEXT,
    masking TEXT,
    arms INTEGER,
    intervention_model TEXT,
    sponsor TEXT,
    eligibility TEXT,
    raw JSON
);
CREATE VIRTUAL TABLE IF NOT EXISTS trials_fts USING fts5(
    nct_id UNINDEXED, title, conditions, interventions,
    primary_outcomes, content=''
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (datetime('now')),
    filename TEXT,
    status TEXT DEFAULT 'created',
    progress JSON,
    usdm JSON,
    similar_trials JSON,
    baseline JSON,
    burden JSON,
    ml_prediction JSON,
    fda_analysis JSON,
    optimized_protocol JSON,
    markdown TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase);
CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
"""


def init_db():
    with get_db() as db:
        db.executescript(SCHEMA)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_trials(rows: list[dict]):
    cols = [
        "nct_id", "title", "status", "phase", "conditions", "interventions",
        "primary_outcomes", "secondary_outcomes", "enrollment", "start_date",
        "completion_date", "duration_months", "allocation", "masking", "arms",
        "intervention_model", "sponsor", "eligibility", "raw",
    ]
    with get_db() as db:
        db.executemany(
            f"INSERT OR REPLACE INTO trials ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        db.executemany(
            "INSERT INTO trials_fts (nct_id, title, conditions, interventions, "
            "primary_outcomes) VALUES (?,?,?,?,?)",
            [
                (r["nct_id"], r.get("title") or "", r.get("conditions") or "",
                 r.get("interventions") or "", r.get("primary_outcomes") or "")
                for r in rows
            ],
        )


def trial_count() -> int:
    with get_db() as db:
        return db.execute("SELECT count(*) FROM trials").fetchone()[0]


def search_trials(query: str, limit: int = 50) -> list[dict]:
    fts = " OR ".join(
        f'"{t}"' for t in query.replace('"', "").split() if len(t) > 2
    )
    if not fts:
        return []
    with get_db() as db:
        rows = db.execute(
            "SELECT t.* FROM trials_fts f JOIN trials t ON t.nct_id = f.nct_id "
            "WHERE trials_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, str) and v[:1] in "{[":
            try:
                d[k] = json.loads(v)
            except ValueError:
                pass
    return d


def update_session(session_id: str, **fields):
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        vals.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
    vals.append(session_id)
    with get_db() as db:
        db.execute(
            f"UPDATE sessions SET {', '.join(sets)} WHERE session_id = ?", vals
        )


def create_session(session_id: str, filename: str):
    with get_db() as db:
        db.execute(
            "INSERT INTO sessions (session_id, filename) VALUES (?, ?)",
            (session_id, filename),
        )


def list_sessions() -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT session_id, created_at, filename, status FROM sessions "
            "ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
