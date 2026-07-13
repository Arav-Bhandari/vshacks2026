# CRAP - Comprehensive Review and Analysis Platform

**License**: MIT (Open Source)

---

## Elevator Pitch

**Your AI-driven clinical trial intelligence platform that reviews, benchmarks, and regenerates protocol drafts into USDM-ready, FDA-aligned docs: Reducing amendments, delays, and cost overruns.**

---

## Inspiration

The members of the CRAP team came in with all but a singular question:

"_Why do SO MANY promising discoveries at the bench fail to reach the patients at the bedside?_"

In fact, roughly **9 in 10** clinical developments **fail** between starting Phase I trials and receiving regulatory approval. While many of these failures stem from biological uncertainty, a surprisingly large proportion are lost not in the lab, but in clinical trial design and operations.

While wet-lab innovation races ahead, trial design still lives in sprawling word documents/PDFs - **Even at leading biopharma companies**. These protocols span hundreds of pages, presenting scattered trial design information.
When foundational design choices are made inside such unstructured, manual systems, **trials become vulnerable to avoidable operational risks**: misaligned endpoints, impractical timelines, or regulatory gaps that can compromise even the most promising science.

_The result?_ **Delayed trials, avoidable amendments, and millions of dollars in wasted effort.**

**Enter CRAP.**
_Our mission?_ Controlling the controllables, by **making clinical trial design as intelligent as the science it tests**, narrowing the chasm between therapeutic discovery and approval.

---

## What it does

CRAP transforms messy, unstructured trial drafts into structured and regulator-aligned designs, followed by regenerating improved versions using AI.

### Core Workflow

1. **Upload** any Phase II-III trial draft PDF doc.
2. **Convert** it into a machine-readable USDM structure (Schedule of Activities, endpoints, arms, eligibility, etc.)
3. **Generate insights** on factors that may slow down trial progress using data from **591K+ historical clinical studies**, benchmarking performance metrics such as duration, procedural burden, and amendment likelihood.
4. **Identify missing regulatory elements** by cross-referencing FDA guidance documents, while highlighting compliance gaps and potential design inefficiencies.
5. **Benchmark trial performance** against studies of similar drugs, mechanisms, and phases, providing justification on how design choices (e.g., endpoints, visit frequency, population scope) align with successful precedents.
6. **Regenerate** an improved, citation-linked draft and export it as USDM-ready JSON/XML for CRO or CTMS integration.

### Key Features

#### Protocol Intelligence System
- **PDF Processing**: Automatic PDF to Markdown to USDM conversion using DeepSeek V4 Flash
- **Similar Trials Discovery**: Find up to 50 similar trials from 591K+ studies
- **Similarity Scoring**: Multi-factor semantic analysis (condition 35%, phase 20%, endpoints 25%, design 20%)
- **Baseline Metrics**: Weighted aggregation from top-K most similar trials for realistic benchmarking
- **Burden Analysis**: Rule-based complexity, recruitment difficulty, and patient burden scoring
- **ML Predictions**: XGBoost model with SHAP explainability for duration overrun risk
- **FDA Compliance**: AI-powered regulatory guidance analysis using actual FDA PDF documents
- **Protocol Optimization**: AI-powered regeneration with citations and regulatory alignment
- **USDM Export**: Industry-standard CDISC format export (JSON and XML)

#### Natural Language Trial Search
- Query 591K+ clinical trials using DeepSeek V4 Flash with tool use
- MCP server exposing the same trial database to any MCP-compatible client
- Live ClinicalTrials.gov API fallback

---

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
|                  Backend API (FastAPI)                       |
|DeepSeek V4| SQLite FTS5 | ML Models | FDA Guidance | SHAP  |
|   Async pipeline | Session persistence | WS progress        |
+---------------------------+--------------------------------+
                            |
                            v
+------------------------------------------------------------+
|              Data Layer & External Services                  |
|  591K Trials (SQLite) | FDA Guidance PDFs | ClinicalTrials.gov|
+------------------------------------------------------------+
```

### Technology Stack

#### Backend (Python/FastAPI)
- **FastAPI** - async web framework with automatic API documentation
- **DeepSeek V4 Flash** - USDM conversion, FDA analysis, protocol optimization, NL search
- **SQLite + FTS5** - 591K+ trials from ClinicalTrials.gov, full-text search, zero-install
- **sentence-transformers** - semantic similarity (all-MiniLM-L6-v2, 384-dim embeddings)
- **XGBoost + SHAP** - duration prediction with TreeExplainer attributions
- **PyMuPDF + pdfplumber** - hybrid PDF text extraction (no OCR)
- **WebSockets** - real-time progress updates with HTTP polling fallback

#### Frontend (Next.js/TypeScript)
- **Next.js App Router** + **TypeScript strict**
- **Tailwind CSS**, hand-rolled shadcn-style primitives
- **Recharts** - burden charts, risk gauges, SHAP plots, benchmark CIs
- **Lucide React** icons

#### AI & ML Infrastructure
- **Model Context Protocol (MCP)** - TypeScript MCP server for trial discovery
- **CDISC USDM v3.0** - industry-standard clinical study data model export
- **FDA Guidance Library** - 12 real FDA PDFs (general, oncology, genetics)

### Pipeline

PDF parse -> USDM conversion (DeepSeek V4 Flash) -> similar trials (semantic 4-factor scoring) -> weighted baseline benchmarks -> burden analysis -> ML duration/risk prediction with SHAP -> FDA compliance (guidance selection and gap analysis) -> optimized protocol draft (DeepSeek V4 Flash) -> USDM JSON/XML export.

---

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

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- DeepSeek API key

### Installation

```bash
# 1. Database + dependencies (fetches 591K trials, takes a while)
bash database/setup_database.sh

# 2. FDA guidance PDFs
bash scripts/fetch_fda.sh

# 3. Environment
cp backend/app/.env.example backend/app/.env
# add DEEPSEEK_API_KEY to backend/app/.env
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > front-end/.env.local

# 4. Deep-train the duration stack (GPU recommended)
cd backend
EMB_DEVICE=cuda TRAIN_DEVICE=cuda ../.venv/bin/python -m app.ml.train --deep
# Optional right-censored AFT experiment using active studies
TRAIN_DEVICE=cuda ../.venv/bin/python -m app.ml.survival --deep
# Embeddings and versioned model/metadata artifacts are cached in app/ml/models/

# 5. Backend (Terminal 1)
cd backend && ../.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# 6. Frontend (Terminal 2)
cd front-end && npm install && npm run dev
```

Visit **http://localhost:3000**. API docs at **http://localhost:8000/docs**.

### MCP server (optional)

```bash
cd mcp-server && npm install && npm run build
```

Register `node mcp-server/dist/index.js` in any MCP client config; see mcp-server/README.md.

---

## Performance

- PDF parsing: seconds, no OCR (hybrid PyMuPDF + pdfplumber)
- Keyword search over 591K trials: <100ms (SQLite FTS5)
- Similarity scoring: single batched sentence-transformer encode over candidates
- ML prediction + SHAP: <1s
- Full pipeline: minutes end to end, dominated by DeepSeek V4 Flash stages; progress streamed live

**Honest metrics**: training selects hyperparameters on expanding historical windows and reports an untouched mature-year test, a sponsor-held-out diagnostic, and the old random split separately. The authoritative row counts, raw/log R², MAE/RMSE, calibration, split manifest, data hash, and known snapshot limitations are written to `backend/app/ml/models/train_meta.json` on every run.

---

## Testing

```bash
cd backend
../.venv/bin/python -m pytest tests/ -v

# frontend
cd front-end
npx tsc --noEmit && npm run build

# mcp server smoke test
cd mcp-server && node scripts/smoke.mjs
```

---

## Documentation

- **[Developer guide](CLAUDE.md)** - development commands and guidance
- **[docs/backend/architecture.md](docs/backend/architecture.md)** - module map and pipeline detail
- **API Docs**: http://localhost:8000/docs (FastAPI auto-generated)

---

## Contributing

Contributions welcome. This is an open-source project under MIT license.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite
5. Open a Pull Request

**Development guidelines**: PEP 8 for Python, TypeScript strict mode, tests for new features, comments only when necessary (one line, 60 chars max), no emojis in code.

---

## License

MIT License - see [LICENSE](LICENSE).

---

## Acknowledgments

- **DeepSeek** - DeepSeek V4 Flash API for AI processing
- **ClinicalTrials.gov** - public clinical trials registry (591K+ studies loaded)
- **CDISC** - USDM v3.0 standard for clinical study data
- **FDA** - public guidance documents enabling regulatory intelligence
