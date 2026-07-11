# Backend Architecture

## Module Map

Backend is organized by concern:

- **`app/main.py`** — FastAPI app, mount routes, health/swagger endpoints
- **`app/config.py`** — settings, paths, env loading
- **`app/pipeline.py`** — async orchestration, progress tracking, error handling
- **`app/database/db.py`** — SQLite CRUD (sessions, trials, FTS)
- **`app/database/fetch_trials.py`** — load trials from ClinicalTrials.gov API
- **`app/routes/protocol.py`** — POST upload, GET/export protocol endpoints
- **`app/routes/search.py`** — /api/trials/search (FTS), POST /api/search/nl
- **`app/routes/ws.py`** — WebSocket /ws/{id}, broadcast progress events
- **`app/ml/`** — XGBoost duration/risk predictor, feature extraction, model training
- **`app/services/`** — pipeline stages (each file = one transformation)

## Pipeline Stages

Each stage in `app/pipeline.py` transforms the protocol and updates the session:

| Stage | Function | Input | Output | Stores in |
|-------|----------|-------|--------|-----------|
| parse | `parse_pdf(pdf_path)` → str | PDF file | markdown text | `session.markdown` |
| usdm | `convert_to_usdm(markdown)` → dict | markdown | USDM JSON | `session.usdm` |
| similar | `find_similar(usdm, limit=50)` → list | USDM JSON | trial NCT IDs + scores | `session.similar_trials` |
| baseline | `compute_baseline(similar_trials, n=10)` → dict | trial list | agg enrollment/duration stats | `session.baseline` |
| burden | `analyze_burden(usdm)` → dict | USDM JSON | participant burden assessment | `session.burden` |
| ml | `predict_duration_risk(usdm, baseline, burden)` → dict | all prior | XGBoost predictions + SHAP | `session.ml_prediction` |
| fda | `analyze_fda_compliance(usdm)` → dict | USDM JSON | gaps vs. FDA guidance (Claude) | `session.fda_analysis` |
| optimize | `optimize_protocol(usdm, similar, fda, burden)` → dict | all prior | revised protocol (Claude extended thinking) | `session.optimized_protocol` |

## Sessions Table Schema

SQLite `sessions` table stores protocol analysis state. All JSON fields are serialized as strings; `db.get_session()` auto-deserializes:

```python
{
  "session_id": str,          # UUID
  "created_at": str,          # ISO timestamp
  "filename": str,            # uploaded PDF filename
  "status": str,              # "created" | "running" | "complete" | "error"
  "progress": {               # current step progress
    "step": str,              # stage name
    "status": str,            # "running" | "done" | "error"
    "detail": str,            # human-readable message
    "pct": int                # 0-100
  },
  "markdown": str,            # extracted PDF text
  "usdm": dict,               # USDM JSON structure
  "similar_trials": list,     # [{"nct_id": ..., "score": ...}, ...]
  "baseline": dict,           # {"avg_enrollment": ..., "avg_duration_months": ...}
  "burden": dict,             # burden assessment output
  "ml_prediction": dict,      # {"duration_months": float, "risk_score": float, "shap": ...}
  "fda_analysis": dict,       # compliance gaps, guidance references
  "optimized_protocol": dict  # revised protocol with improvements
}
```

## Data & Models

**Trials database:**
- Path: `database/clinical_trials.db`
- Source: ClinicalTrials.gov API
- Loader: `app/database/fetch_trials.py` (separate status filters for completed vs. ongoing)
- Size: 591K trials, ~1.7GB
- Indexes: phase, status; FTS5 on title/conditions/interventions

**ML models:**
- Path: `backend/app/ml/models/`
- Type: XGBoost (duration, risk classification)
- Training: 556K completed trials with real duration
- Features: extracted from USDM (phase, enrollment, study arms, etc.)
- Loaded by: `app/ml/predictor.py`
- Training script: `app/ml/create_demo_models.py`

**FDA PDFs:**
- Path: `fda/{category}/{filename}` (general, oncology, genetics)
- Manifest: `fda/manifest.json` (filename, category, title, download URL)
- Downloaded by: `scripts/fetch_fda.sh`
- Used by: `app/services/fda_analyzer.py` (Claude reads & references)
