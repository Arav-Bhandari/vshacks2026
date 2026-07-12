# CRAP

**Comprehensive Review and Analysis Platform** — An AI assistant designed to analyze and sharpen your clinical trial protocols. Just upload a Phase II or III PDF, and the platform goes to work: structuring it into CDISC USDM, benchmarking it against over 591,000 historical trials, and ensuring compliance with FDA guidance. It also flags potential timeline risks and generates an optimized draft you can instantly export as USDM JSON or XML.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Repository Layout](#repository-layout)
- [Development](#development)
- [License](#license)
- [Credits](#credits)

## Overview

CRAP turns messy, unstructured protocols into organized, benchmarked data that aligns with regulators.

The platform combines:

| Capability | What it provides |
|---|---|
| USDM conversion | Machine-readable CDISC USDM v3.0 model that captures your endpoints, study arms, eligibility criteria, and Schedule of Activities |
| Trial benchmarking | Baseline metrics for duration, enrollment, and operational burden by analyzing the most similar historical trials. |
| Burden analysis | Rule-based scores for complexity, recruitment difficulty, and overall patient burden. |
| Risk prediction | XGBoost duration-overrun model paired with per-prediction SHAP explanations. |
| FDA compliance | Gap analysis against a library of 12 real FDA guidance PDFs |
| Protocol optimization | An AI-regenerated draft with every modification linked back to source citations. |
| Trial search | Natural-language search over 591K+ trials, featuring a MCP server for external clients |

## How It Works

The backend runs an eight-stage asynchronous pipeline, streaming progress to the frontend over WebSockets:

1. **Parse** — Extracts text from the uploaded protocol PDF using a hybrid PyMuPDF and pdfplumber (no OCR)
2. **Convert** — Transforms the extracted text into structured USDM JSON model with Claude
3. **Match** — Finds highly comparable historical trials via sentence-transformer embeddings and four-factor scoring (condition 35%, endpoints 25%, phase 20%, design 20%)
4. **Baseline** — Aggregates weighted benchmark metrics from those top matched historical trials
5. **Assess** — Scores participant burden and recruitment difficulty
6. **Predict** — Estimates the duration-overrun risk using XGBoost backed by an explanation from SHAP
7. **Check** — Conducts a two-stage FDA compliance analysis: Haiku selects relevant guidance documents, then Sonnet performs the gap analysis
8. **Optimize** — Regenerates an improved protocol draft with Claude extended thinking, ready to export as USDM JSON/XML

```
+------------------------------------------------------------+
|                   Frontend (Next.js)                        |
|   Trial Search | Protocol Upload | Analysis Dashboard       |
+---------------------------+--------------------------------+
                            | REST + WebSockets
                            v
+------------------------------------------------------------+
|                  Backend (FastAPI)                          |
|   8-stage async pipeline | Claude | ML models | Sessions    |
+---------------------------+--------------------------------+
                            |
                            v
+------------------------------------------------------------+
|                       Data Layer                            |
| SQLite (591K trials, FTS5) | FDA PDFs | ClinicalTrials.gov  |
+------------------------------------------------------------+
```

## Getting Started

### Requirements

- Python 3.12+
- Node.js 18+
- An Anthropic API key

### Setup

```bash
# Database + Python dependencies (downloads 591K trials; takes a while)
bash database/setup_database.sh

# FDA guidance PDFs
bash scripts/fetch_fda.sh

# Environment variables
cp backend/app/.env.example backend/app/.env   # then add ANTHROPIC_API_KEY
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > front-end/.env.local

# ML model (full training run: GPU box or ~30 min on CPU;
# EMB_DEVICE=cuda|mps|cpu overrides the encoder device, and
# embeddings cache to backend/app/ml/models/ so reruns resume)
cd backend && ../.venv/bin/python -m app.ml.train
```

### Run

```bash
# Terminal 1 — backend
cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd front-end && npm install && npm run dev
```

Open **http://localhost:3000**. Interactive API docs are at **http://localhost:8000/docs**.

## Usage

- **Analyze a protocol** — upload a Phase II–III protocol PDF from the dashboard and watch the pipeline stages complete in real time; results include benchmarks, burden scores, risk prediction, compliance gaps, and the optimized draft
- **Search trials** — query the trial database with keywords or plain English from the search page
- **Export** — download the analyzed or optimized protocol as USDM v3.0 JSON or XML for CRO/CTMS integration
- **MCP server (optional)** — expose the trial database to any MCP-compatible client:

  ```bash
  cd mcp-server && npm install && npm run build
  # register node mcp-server/dist/index.js in your MCP client config
  ```

  See [mcp-server/README.md](mcp-server/README.md) for tool details.

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | FastAPI, Python 3.12, SQLite + FTS5, WebSockets |
| AI | Claude (Anthropic API): USDM conversion, FDA analysis, optimization, NL search |
| ML | sentence-transformers (all-MiniLM-L6-v2), XGBoost, SHAP |
| PDF | PyMuPDF, pdfplumber |
| Frontend | Next.js (App Router), TypeScript strict, Tailwind CSS, Recharts, Lucide |
| Integration | Model Context Protocol server (TypeScript), CDISC USDM v3.0 export |
| Data | ClinicalTrials.gov (591K+ studies), FDA guidance PDFs |

> **A note on model accuracy:** the duration model trains on 556K completed trials with a holdout R² of about 0.3 — real-world trial duration depends heavily on factors outside protocol design. Predictions are directional, and each ships with SHAP attributions explaining it.

## Repository Layout

```
backend/app/database/    SQLite layer, ClinicalTrials.gov fetcher
backend/app/services/    PDF parser, USDM converter, similarity, burden,
                         baseline, FDA analyzer, optimizer
backend/app/ml/          features, training, XGBoost+SHAP predictor
backend/app/routes/      REST endpoints, WebSocket handler
backend/app/pipeline.py  8-stage analysis orchestrator
backend/tests/           pytest suites (services, ML, FDA, API)
database/                trial DB (generated) + setup script
front-end/               Next.js app (search, upload, dashboard)
mcp-server/              TypeScript MCP server (3 trial tools)
fda/                     FDA guidance PDFs + manifest.json
scripts/                 FDA PDF fetch script
docs/backend/            architecture notes
```

## Development

Run the test suites before opening a PR:

```bash
# Backend
cd backend && ../.venv/bin/python -m pytest tests/ -v

# Frontend
cd front-end && npx tsc --noEmit && npm run build

# MCP server
cd mcp-server && node scripts/smoke.mjs
```

Conventions: PEP 8 for Python, TypeScript strict mode, tests for new features, no emojis in code. More detail in [CLAUDE.md](CLAUDE.md) and [docs/backend/architecture.md](docs/backend/architecture.md).

Contributions are welcome — fork, branch, change, test, and open a pull request.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Credits

Built on the Claude API by [Anthropic](https://www.anthropic.com), public registry data from [ClinicalTrials.gov](https://clinicaltrials.gov), the [CDISC](https://www.cdisc.org) USDM v3.0 standard, and public [FDA](https://www.fda.gov) guidance documents.
