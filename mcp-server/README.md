# trialscope-mcp

MCP server (stdio transport) exposing clinical-trial search tools for
TrialScope AI. Reads from the local SQLite DB at
`../database/clinical_trials.db` when present, falling back to the live
ClinicalTrials.gov API v2 otherwise.

## Tools

- `search_trials {query, condition?, phase?, limit?}` — FTS search against
  the local DB (`trials_fts` joined to `trials`); falls back to the live API
  when the DB is missing/empty or returns no rows.
- `get_trial {nct_id}` — looks up a trial by NCT ID in the local DB, falling
  back to `GET /api/v2/studies/{nctId}`.
- `trial_stats {condition?, phase?}` — count, average duration, and average
  enrollment from the local DB only (returns a message if the DB is empty).

## Setup

```sh
npm install
npm run build
```

## Register with Claude

Add to your Claude MCP config (e.g. `claude_desktop_config.json` or a
project `.mcp.json`):

```json
{
  "mcpServers": {
    "trialscope": {
      "command": "node",
      "args": ["/absolute/path/to/mcp-server/dist/index.js"]
    }
  }
}
```

Or via `npx` once published:

```json
{
  "mcpServers": {
    "trialscope": {
      "command": "npx",
      "args": ["-y", "trialscope-mcp"]
    }
  }
}
```

## Smoke test

```sh
npm run build
node scripts/smoke.mjs
```

Sends a JSON-RPC `initialize` + `tools/list` request over stdio and checks
that all three tools are advertised.
