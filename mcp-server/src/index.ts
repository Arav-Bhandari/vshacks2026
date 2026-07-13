#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import Database from "better-sqlite3";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.resolve(__dirname, "..", "..", "database", "clinical_trials.db");
const API_BASE = "https://clinicaltrials.gov/api/v2/studies";

interface Trial {
  nct_id: string;
  title: string | null;
  status: string | null;
  phase: string | null;
  conditions: string | null;
  interventions: string | null;
  primary_outcomes: string | null;
  secondary_outcomes: string | null;
  enrollment: number | null;
  start_date: string | null;
  completion_date: string | null;
  duration_months: number | null;
  allocation: string | null;
  masking: string | null;
  arms: string | null;
  intervention_model: string | null;
  sponsor: string | null;
  eligibility: string | null;
}

function openDb(): Database.Database | null {
  try {
    const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
    return db;
  } catch {
    return null;
  }
}

function tableHasRows(db: Database.Database, table: string): boolean {
  try {
    const row = db.prepare(`SELECT COUNT(*) AS c FROM ${table}`).get() as { c: number };
    return row.c > 0;
  } catch {
    return false;
  }
}

function fetchJson(url: string): Promise<any> {
  return fetch(url).then((r) => {
    if (!r.ok) throw new Error(`ClinicalTrials.gov API error ${r.status}`);
    return r.json();
  });
}

function joinList(arr: unknown): string | null {
  if (!Array.isArray(arr)) return null;
  return arr.join("; ") || null;
}


function normalizeStudy(study: any): Trial {
  const proto = study.protocolSection ?? {};
  const id = proto.identificationModule ?? {};
  const status = proto.statusModule ?? {};
  const design = proto.designModule ?? {};
  const conditions = proto.conditionsModule ?? {};
  const arms = proto.armsInterventionsModule ?? {};
  const outcomes = proto.outcomesModule ?? {};
  const sponsor = proto.sponsorCollaboratorsModule ?? {};
  const eligibility = proto.eligibilityModule ?? {};

  const interventions = Array.isArray(arms.interventions)
    ? arms.interventions.map((i: any) => i.name).filter(Boolean)
    : [];
  const primaryOutcomes = Array.isArray(outcomes.primaryOutcomes)
    ? outcomes.primaryOutcomes.map((o: any) => o.measure).filter(Boolean)
    : [];
  const secondaryOutcomes = Array.isArray(outcomes.secondaryOutcomes)
    ? outcomes.secondaryOutcomes.map((o: any) => o.measure).filter(Boolean)
    : [];
  const armGroups = Array.isArray(arms.armGroups)
    ? arms.armGroups.map((a: any) => a.label).filter(Boolean)
    : [];

  return {
    nct_id: id.nctId ?? "",
    title: id.briefTitle ?? id.officialTitle ?? null,
    status: status.overallStatus ?? null,
    phase: joinList(design.phases),
    conditions: joinList(conditions.conditions),
    interventions: joinList(interventions),
    primary_outcomes: joinList(primaryOutcomes),
    secondary_outcomes: joinList(secondaryOutcomes),
    enrollment: design.enrollmentInfo?.count ?? null,
    start_date: status.startDateStruct?.date ?? null,
    completion_date: status.completionDateStruct?.date ?? null,
    duration_months: null,
    allocation: design.designInfo?.allocation ?? null,
    masking: design.designInfo?.maskingInfo?.masking ?? null,
    arms: joinList(armGroups),
    intervention_model: design.designInfo?.interventionModel ?? null,
    sponsor: sponsor.leadSponsor?.name ?? null,
    eligibility: eligibility.eligibilityCriteria ?? null,
  };
}

async function searchTrialsLive(
  query: string,
  condition?: string,
  limit = 10
): Promise<Trial[]> {
  const params = new URLSearchParams();
  if (query) params.set("query.term", query);
  if (condition) params.set("query.cond", condition);
  params.set("pageSize", String(limit));
  const data = await fetchJson(`${API_BASE}?${params.toString()}`);
  const studies = Array.isArray(data.studies) ? data.studies : [];
  return studies.map(normalizeStudy);
}

async function getTrialLive(nctId: string): Promise<Trial | null> {
  try {
    const data = await fetchJson(`${API_BASE}/${encodeURIComponent(nctId)}`);
    return normalizeStudy(data);
  } catch {
    return null;
  }
}

const server = new McpServer({ name: "crap-mcp", version: "0.1.0" });

server.registerTool(
  "search_trials",
  {
    title: "Search clinical trials",
    description:
      "Full-text search of clinical trials, local DB first, live ClinicalTrials.gov fallback.",
    inputSchema: {
      query: z.string().describe("Free-text search query"),
      condition: z.string().optional().describe("Condition/disease filter"),
      phase: z.string().optional().describe("Trial phase filter, e.g. PHASE2"),
      limit: z.number().int().positive().max(50).optional().default(10),
    },
  },
  async ({ query, condition, phase, limit }) => {
    const db = openDb();
    let rows: Trial[] = [];

    if (db && tableHasRows(db, "trials_fts")) {
      try {
        const clauses: string[] = ["trials_fts MATCH ?"];
        const args: unknown[] = [query];
        if (phase) {
          clauses.push("t.phase = ?");
          args.push(phase);
        }
        if (condition) {
          clauses.push("t.conditions LIKE ?");
          args.push(`%${condition}%`);
        }
        const sql = `
          SELECT t.* FROM trials_fts
          JOIN trials t ON t.nct_id = trials_fts.nct_id
          WHERE ${clauses.join(" AND ")}
          LIMIT ?`;
        args.push(limit);
        rows = db.prepare(sql).all(...args) as Trial[];
      } catch {
        rows = [];
      } finally {
        db.close();
      }
    } else if (db) {
      db.close();
    }

    if (rows.length === 0) {
      const combinedQuery = phase ? `${query} ${phase}` : query;
      rows = await searchTrialsLive(combinedQuery, condition, limit);
    }

    return {
      content: [{ type: "text", text: JSON.stringify({ source: rows.length ? "db-or-live" : "none", results: rows }, null, 2) }],
    };
  }
);

server.registerTool(
  "get_trial",
  {
    title: "Get a clinical trial by NCT ID",
    description: "Look up a single trial in the local DB, falling back to the live API.",
    inputSchema: {
      nct_id: z.string().describe("NCT identifier, e.g. NCT01234567"),
    },
  },
  async ({ nct_id }) => {
    const db = openDb();
    let trial: Trial | null = null;

    if (db) {
      try {
        trial = (db.prepare("SELECT * FROM trials WHERE nct_id = ?").get(nct_id) as Trial) ?? null;
      } catch {
        trial = null;
      } finally {
        db.close();
      }
    }

    if (!trial) {
      trial = await getTrialLive(nct_id);
    }

    if (!trial) {
      return {
        content: [{ type: "text", text: JSON.stringify({ error: `Trial ${nct_id} not found` }) }],
        isError: true,
      };
    }

    return { content: [{ type: "text", text: JSON.stringify(trial, null, 2) }] };
  }
);

server.registerTool(
  "trial_stats",
  {
    title: "Aggregate clinical trial statistics",
    description: "Counts and averages from the local DB, optionally filtered by condition/phase.",
    inputSchema: {
      condition: z.string().optional(),
      phase: z.string().optional(),
    },
  },
  async ({ condition, phase }) => {
    const db = openDb();
    if (!db || !tableHasRows(db, "trials")) {
      db?.close();
      return {
        content: [
          { type: "text", text: "Local database is empty or unavailable; stats require the local DB." },
        ],
      };
    }

    try {
      const clauses: string[] = [];
      const args: unknown[] = [];
      if (condition) {
        clauses.push("conditions LIKE ?");
        args.push(`%${condition}%`);
      }
      if (phase) {
        clauses.push("phase = ?");
        args.push(phase);
      }
      const where = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
      const sql = `
        SELECT COUNT(*) AS count,
               AVG(duration_months) AS avg_duration_months,
               AVG(enrollment) AS avg_enrollment
        FROM trials ${where}`;
      const stats = db.prepare(sql).get(...args);
      return { content: [{ type: "text", text: JSON.stringify(stats, null, 2) }] };
    } finally {
      db.close();
    }
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("Fatal error starting crap-mcp:", err);
  process.exit(1);
});
