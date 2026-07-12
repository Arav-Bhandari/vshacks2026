# CRAP — Comprehensive Review and Analysis Platform

AI-driven clinical trial protocol analysis. Upload a Phase II–III protocol draft PDF and CRAP converts it to CDISC USDM, benchmarks it against 591K+ historical trials, flags FDA compliance gaps, predicts duration risk, and generates an improved, citation-linked draft exportable as USDM JSON/XML.

## Features

- **PDF to USDM conversion** — extracts protocol text and transforms it into structured USDM v3.0 (Schedule of Activities, endpoints, arms, eligibility) using Claude
- **Similar trials discovery** — finds up to 50 comparable trials from 591K+ ClinicalTrials.gov studies via multi-factor semantic scoring (condition 35%, phase 20%, endpoints 25%, design 20%)
- **Baseline benchmarking** — weighted metric aggregation (duration, enrollment, burden) from the most similar trials
- **Burden analysis** — rule-based complexity, recruitment difficulty, and patient burden scoring
- **ML risk prediction** — XGBoost duration overrun model with SHAP explanations
- **FDA compliance analysis** — two-stage Claude pipeline (Haiku document selection, Sonnet gap analysis) over a library of 12 real FDA guidance PDFs
- **Protocol optimization** — AI-regenerated draft with citations and regulatory alignment, exportable as USDM JSON/XML
- **Natural language trial search** — query the trial database in plain English (Claude with tool use), with live ClinicalTrials.gov API fallback
- **MCP server** — TypeScript server exposing the trial database to any MCP-compatible client

## Architecture

```
+------------------------------------------------------------+
|                   Frontend (Next.js)                        |
|   Trial Search | Protocol Upload | Analysis Dashboard       |
|     Real-time progress via WebSockets (polling fallback)    |
+---------------------------+--------------------------------+
                            | HTTP/REST + WebSockets
                            v
+------------------------------------------------------------+
|                  Backend API (FastAPI)                      |
|   Claude | SQLite FTS5 | ML Models | FDA Guidance | SHAP    |
|   Async pipeline | Session persistence | WS progress        |
+---------------------------+--------------------------------+
                            |
                            v
+------------------------------------------------------------+
|              Data Layer & External Services                 |
| 591K Trials (SQLite) | FDA Guidance PDFs | ClinicalTrials.gov|
+------------------------------------------------------------+
```

**Pipeline:** PDF parse → USDM conversion (Claude) → similar trials (semantic 4-factor scoring) → weighted baseline benchmarks → burden analysis → ML duration/risk prediction with SHAP → FDA compliance analysis → optimized protocol draft (Claude extended thinking) → USDM JSON/XML export.

### Tech Stack

**Backend (Python):** FastAPI, Claude (Anthropic API), SQLite + FTS5, sentence-transformers (all-MiniLM-L6-v2), XGBoost + SHAP, PyMuPDF + pdfplumber, WebSockets

**Frontend (TypeScript):** Next.js App Router, TypeScript strict mode, Tailwind CSS, Recharts, Lucide React

**Standards & data:** CDISC USDM v3.0, Model Context Protocol, ClinicalTrials.gov registry, FDA guidance documents

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── database/         # SQLite layer, ClinicalTrials.gov fetcher
│   │   ├── services/         # PDF parser, USDM converter, similarity,
│   │   │                     # burden, baseline, FDA analyzer, optimizer
│   │   ├── ml/               # features, training, XGBoost+SHAP predictor
│   │   ├── routes/           # REST endpoints, WebSocket handler
│   │   ├── pipeline.py       # 8-stage analysis orchestrator
│   │   └── main.py           # FastAPI entrypoint
│   ├── tests/                # pytest suites (services, ML, FDA, API)
│   └── data/uploads/         # uploaded protocol PDFs
├── database/
│   ├── clinical_trials.db    # 591K trials (generated, gitignored)
│   └── setup_database.sh     # venv + deps + full data load
├── front-end/                # Next.js app (search, upload, dashboard)
├── mcp-server/               # TypeScript MCP server (3 trial tools)
├── fda/                      # FDA guidance PDFs + manifest.json
├── scripts/fetch_fda.sh      # re-download guidance PDFs from manifest
└── docs/backend/             # architecture notes
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- Anthropic API key

### Installation

```bash
# 1. Database + dependencies (fetches 591K trials, takes a while)
bash database/setup_database.sh

# 2. FDA guidance PDFs
bash scripts/fetch_fda.sh

# 3. Environment
cp backend/app/.env.example backend/app/.env
# add ANTHROPIC_API_KEY to backend/app/.env
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > front-end/.env.local

# 4. Train ML model (full run needs a GPU box or ~30 min CPU)
cd backend && ../.venv/bin/python -m app.ml.train
# EMB_DEVICE=cuda|mps|cpu overrides encoder device; embeddings
# cache to backend/app/ml/models/ so reruns resume

# 5. Backend (Terminal 1)
cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# 6. Frontend (Terminal 2)
cd front-end && npm install && npm run dev
```

Visit **http://localhost:3000**. API docs at **http://localhost:8000/docs**.

### MCP Server (optional)

```bash
cd mcp-server && npm install && npm run build
```

Register `node mcp-server/dist/index.js` in any MCP client config; see [mcp-server/README.md](mcp-server/README.md).

## Testing

```bash
# backend
cd backend
../.venv/bin/python -m pytest tests/ -v

# frontend
cd front-end
npx tsc --noEmit && npm run build

# mcp server smoke test
cd mcp-server && node scripts/smoke.mjs
```

## Notes on Model Accuracy

The XGBoost duration model trains on 556K real completed trials; holdout R² is 0.12. Real-world trial duration is dominated by factors outside protocol design, so predictions are directional and every prediction ships with SHAP attributions explaining it.

## Documentation

- [CLAUDE.md](CLAUDE.md) — development commands and guidance
- [docs/backend/architecture.md](docs/backend/architecture.md) — module map and pipeline detail
- API docs: http://localhost:8000/docs (FastAPI auto-generated)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite
5. Open a Pull Request

Development guidelines: PEP 8 for Python, TypeScript strict mode, tests for new features, no emojis in code.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Anthropic](https://www.anthropic.com) — Claude API
- [ClinicalTrials.gov](https://clinicaltrials.gov) — public clinical trials registry
- [CDISC](https://www.cdisc.org) — USDM v3.0 standard
- [FDA](https://www.fda.gov) — public guidance documents
