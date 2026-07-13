import json
import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    nct_id TEXT PRIMARY KEY,
    title TEXT,
    official_title TEXT,
    status TEXT,
    study_type TEXT,
    phase TEXT,
    conditions TEXT,
    conditions_json JSON,
    interventions TEXT,
    interventions_json JSON,
    primary_outcomes TEXT,
    primary_outcomes_json JSON,
    secondary_outcomes TEXT,
    secondary_outcomes_json JSON,
    primary_outcome_timeframes TEXT,
    primary_outcome_timeframes_json JSON,
    secondary_outcome_timeframes TEXT,
    secondary_outcome_timeframes_json JSON,
    enrollment INTEGER,
    enrollment_type TEXT,
    start_date TEXT,
    start_date_type TEXT,
    feature_start_date TEXT,
    feature_start_date_type TEXT,
    primary_completion_date TEXT,
    primary_completion_date_type TEXT,
    completion_date TEXT,
    completion_date_type TEXT,
    duration_months REAL,
    allocation TEXT,
    masking TEXT,
    arms INTEGER,
    intervention_model TEXT,
    primary_purpose TEXT,
    observational_model TEXT,
    time_perspective TEXT,
    sponsor TEXT,
    sponsor_class TEXT,
    collaborators_count INTEGER,
    sex TEXT,
    minimum_age TEXT,
    maximum_age TEXT,
    minimum_age_years REAL,
    maximum_age_years REAL,
    healthy_volunteers INTEGER,
    std_ages TEXT,
    sampling_method TEXT,
    study_population TEXT,
    site_count INTEGER,
    country_count INTEGER,
    countries TEXT,
    eligibility TEXT,
    inclusion_criteria_count INTEGER,
    exclusion_criteria_count INTEGER,
    feature_snapshot_date TEXT,
    feature_snapshot_kind TEXT,
    study_first_submit_date TEXT,
    source_updated_at TEXT,
    fetched_at TEXT,
    raw JSON
);
CREATE VIRTUAL TABLE IF NOT EXISTS trials_fts USING fts5(
    nct_id UNINDEXED, title, conditions, interventions,
    primary_outcomes
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

TRIAL_MIGRATIONS = {
    "official_title": "TEXT",
    "study_type": "TEXT",
    "conditions_json": "JSON",
    "interventions_json": "JSON",
    "primary_outcomes_json": "JSON",
    "secondary_outcomes_json": "JSON",
    "primary_outcome_timeframes": "TEXT",
    "primary_outcome_timeframes_json": "JSON",
    "secondary_outcome_timeframes": "TEXT",
    "secondary_outcome_timeframes_json": "JSON",
    "enrollment_type": "TEXT",
    "start_date_type": "TEXT",
    "feature_start_date": "TEXT",
    "feature_start_date_type": "TEXT",
    "primary_completion_date": "TEXT",
    "primary_completion_date_type": "TEXT",
    "completion_date_type": "TEXT",
    "primary_purpose": "TEXT",
    "observational_model": "TEXT",
    "time_perspective": "TEXT",
    "sponsor_class": "TEXT",
    "collaborators_count": "INTEGER",
    "sex": "TEXT",
    "minimum_age": "TEXT",
    "maximum_age": "TEXT",
    "minimum_age_years": "REAL",
    "maximum_age_years": "REAL",
    "healthy_volunteers": "INTEGER",
    "std_ages": "TEXT",
    "sampling_method": "TEXT",
    "study_population": "TEXT",
    "site_count": "INTEGER",
    "country_count": "INTEGER",
    "countries": "TEXT",
    "inclusion_criteria_count": "INTEGER",
    "exclusion_criteria_count": "INTEGER",
    "feature_snapshot_date": "TEXT",
    "feature_snapshot_kind": "TEXT",
    "study_first_submit_date": "TEXT",
    "source_updated_at": "TEXT",
    "fetched_at": "TEXT",
}


def _migrate_trials(db: sqlite3.Connection):
    existing = {row[1] for row in db.execute("PRAGMA table_info(trials)")}
    for column, sql_type in TRIAL_MIGRATIONS.items():
        if column not in existing:
            db.execute(f'ALTER TABLE trials ADD COLUMN "{column}" {sql_type}')
    db.execute("CREATE INDEX IF NOT EXISTS idx_trials_study_type ON trials(study_type)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_trials_start_date ON trials(start_date)")


def init_db():
    with get_db() as db:
        db.executescript(SCHEMA)
        _migrate_trials(db)


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


def upsert_trials(rows: list[dict], *, update_fts: bool = True):
    cols = [
        "nct_id", "title", "official_title", "status", "study_type", "phase",
        "conditions", "conditions_json", "interventions", "interventions_json",
        "primary_outcomes", "primary_outcomes_json", "secondary_outcomes",
        "secondary_outcomes_json", "primary_outcome_timeframes",
        "primary_outcome_timeframes_json", "secondary_outcome_timeframes",
        "secondary_outcome_timeframes_json", "enrollment",
        "enrollment_type", "start_date", "start_date_type",
        "feature_start_date", "feature_start_date_type",
        "primary_completion_date", "primary_completion_date_type", "completion_date",
        "completion_date_type", "duration_months", "allocation", "masking", "arms",
        "intervention_model", "primary_purpose", "observational_model",
        "time_perspective", "sponsor", "sponsor_class", "collaborators_count", "sex",
        "minimum_age", "maximum_age", "minimum_age_years", "maximum_age_years",
        "healthy_volunteers", "std_ages", "sampling_method", "study_population",
        "site_count", "country_count", "countries", "eligibility",
        "inclusion_criteria_count", "exclusion_criteria_count",
        "feature_snapshot_date", "feature_snapshot_kind",
        "study_first_submit_date", "source_updated_at", "fetched_at", "raw",
    ]
    with get_db() as db:
        db.executemany(
            f"INSERT OR REPLACE INTO trials ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        if update_fts:
            ids = [r["nct_id"] for r in rows]
            db.execute(
                f"DELETE FROM trials_fts WHERE nct_id IN ({','.join('?' * len(ids))})",
                ids,
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
