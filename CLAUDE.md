# TrialScope AI Developer Guide

## Quick Commands

**Backend:**
```bash
cd backend
../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd front-end
npm install && npm run dev
```

**Tests:**
```bash
cd backend
../.venv/bin/python -m pytest tests/ -v
```

**Load data:**
```bash
bash database/setup_database.sh  # venv + deps + trials
bash scripts/fetch_fda.sh         # FDA PDFs
cd backend && ../.venv/bin/python -m app.ml.create_demo_models  # ML models
```

## Code Style

- Python: PEP 8, type hints optional but encouraged
- TypeScript: strict mode, tsconfig.json enforced
- Comments: one line max, only when necessary (not obvious from code)
- No emojis
- Module docstrings for complex files

## Architecture Overview

**Pipeline orchestration:** `backend/app/pipeline.py` runs async stages:
1. parse_pdf → extract text from uploaded protocol
2. convert_to_usdm → Claude transforms markdown to USDM JSON
3. find_similar → sentence-transformers + 4-factor scoring
4. compute_baseline → aggregate metrics from similar trials
5. analyze_burden → participant burden assessment
6. predict_duration_risk → XGBoost + SHAP explanations
7. analyze_fda_compliance → Claude 2-stage (haiku doc select + sonnet gap analysis)
8. optimize_protocol → Claude extended thinking for improvements

**Backend modules:**
- `database/db.py` — SQLite session/trial CRUD
- `ml/predictor.py` — load XGBoost models, generate predictions
- `services/` — each stage has a module; `llm_utils.py` wraps Anthropic API
- `routes/protocol.py` — POST /api/protocol/upload, GET/export endpoints
- `routes/search.py` — /api/trials/search, POST /api/search/nl
- `routes/ws.py` — WebSocket /ws/{id} for progress events

**Database:**
- SQLite at `database/clinical_trials.db`
- `trials` table (591K rows, indexed on phase/status)
- `trials_fts` (FTS5 index on title/conditions/interventions)
- `sessions` table (tracks protocol uploads with JSON columns)

**Session schema** (all JSON-serialized):
- `progress`: `{"step": string, "status": string, "detail": string, "pct": int}`
- `usdm`, `similar_trials`, `baseline`, `burden`, `ml_prediction`, `fda_analysis`, `optimized_protocol`: stage outputs
- `markdown`: extracted PDF text

## Env & Secrets

- `backend/app/.env` needs `ANTHROPIC_API_KEY`
- `front-end/.env.local` needs `NEXT_PUBLIC_API_URL=http://localhost:8000`
- FDA PDFs auto-download to `fda/{category}/` if missing
