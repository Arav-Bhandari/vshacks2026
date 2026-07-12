# CRAP

**Comprehensive Review and Analysis Platform** — an AI-powered tool for analyzing and improving clinical trial protocols. Upload a Phase II–III protocol PDF; CRAP structures it into CDISC USDM, benchmarks it against 591K+ historical trials, checks it against FDA guidance, predicts duration risk, and regenerates an improved draft you can export as USDM JSON/XML.

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

CRAP takes an unstructured clinical trial protocol document and turns it into structured, benchmarked, regulator-aligned output. It combines:

| Capability | What it gives you |
|---|---|
| USDM conversion | Machine-readable CDISC USDM v3.0 (endpoints, arms, eligibility, Schedule of Activities) |
| Trial benchmarking | Duration, enrollment, and burden baselines from the most similar historical trials |
| Burden analysis | Rule-based scoring of complexity, recruitment difficulty, and patient burden |
| Risk prediction | XGBoost duration-overrun model with per-prediction SHAP explanations |
| FDA compliance | Gap analysis against a library of 12 real FDA guidance PDFs |
| Protocol optimization | AI-regenerated, citation-linked draft with regulatory alignment |
| Trial search | Natural-language search over 591K+ trials, plus an MCP server for external clients |

## How It Works

The backend runs an eight-stage asynchronous pipeline, streaming progress to the frontend over WebSockets:

1. **Parse** — extract text from the uploaded protocol PDF (hybrid PyMuPDF + pdfplumber, no OCR)
2. **Convert** — transform the extracted text into USDM JSON with Claude
3. **Match** — find comparable trials via sentence-transformer embeddings and four-factor scoring (condition 35%, endpoints 25%, phase 20%, design 20%)
4. **Baseline** — aggregate weighted benchmark metrics from the top matches
5. **Assess** — score participant burden and recruitment difficulty
6. **Predict** — estimate duration-overrun risk with XGBoost and explain it with SHAP
7. **Check** — two-stage FDA compliance analysis (Haiku selects relevant guidance documents, Sonnet performs the gap analysis)
8. **Optimize** — regenerate an improved protocol draft with Claude extended thinking, exportable as USDM JSON/XML

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

> **A note on model accuracy:** the duration model trains on 556K completed trials with a holdout R² of 0.12 — real-world trial duration depends heavily on factors outside protocol design. Predictions are directional, and each ships with SHAP attributions explaining it.

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
