"""Bulk-load completed trials from ClinicalTrials.gov API v2."""
import json
import sys
import time
from pathlib import Path

import httpx

from app.database.db import init_db, trial_count, upsert_trials

API = "https://clinicaltrials.gov/api/v2/studies"
TOKEN_FILE = Path(__file__).parent / ".fetch_token"
FIELDS = ",".join([
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.designModule",
    "protocolSection.conditionsModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule",
    "protocolSection.eligibilityModule",
    "protocolSection.sponsorCollaboratorsModule",
])


def _date(d: dict | None) -> str | None:
    return (d or {}).get("date")


def _months(start: str | None, end: str | None) -> float | None:
    if not start or not end or len(start) < 7 or len(end) < 7:
        return None
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    m = (ey - sy) * 12 + (em - sm)
    return float(m) if m > 0 else None


def parse_study(s: dict) -> dict | None:
    p = s.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    nct = ident.get("nctId")
    if not nct:
        return None
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    info = design.get("designInfo", {})
    arms_mod = p.get("armsInterventionsModule", {})
    outcomes = p.get("outcomesModule", {})
    start = _date(status.get("startDateStruct"))
    end = _date(status.get("completionDateStruct"))
    return {
        "nct_id": nct,
        "title": ident.get("briefTitle"),
        "status": status.get("overallStatus"),
        "phase": ", ".join(design.get("phases") or []),
        "conditions": ", ".join(p.get("conditionsModule", {}).get("conditions") or []),
        "interventions": "; ".join(
            f"{i.get('type', '')}: {i.get('name', '')}"
            for i in arms_mod.get("interventions") or []
        ),
        "primary_outcomes": "; ".join(
            o.get("measure", "") for o in outcomes.get("primaryOutcomes") or []
        ),
        "secondary_outcomes": "; ".join(
            o.get("measure", "") for o in outcomes.get("secondaryOutcomes") or []
        ),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "start_date": start,
        "completion_date": end,
        "duration_months": _months(start, end),
        "allocation": info.get("allocation"),
        "masking": (info.get("maskingInfo") or {}).get("masking"),
        "arms": len(arms_mod.get("armGroups") or []) or None,
        "intervention_model": info.get("interventionModel"),
        "sponsor": p.get("sponsorCollaboratorsModule", {})
        .get("leadSponsor", {}).get("name"),
        "eligibility": (p.get("eligibilityModule", {}).get("eligibilityCriteria")
                        or "")[:4000],
        "raw": None,
    }


def fetch_all(max_pages: int | None = None, statuses: str = "COMPLETED"):
    init_db()
    token_file = TOKEN_FILE.with_suffix("." + statuses[:12].lower())
    token = token_file.read_text().strip() if token_file.exists() else None
    if token == "DONE":
        print(f"already complete: {trial_count()} trials")
        return
    pages = 0
    with httpx.Client(timeout=60) as client:
        while True:
            params = {
                "filter.overallStatus": statuses,
                "pageSize": 1000,
                "fields": FIELDS,
            }
            if token:
                params["pageToken"] = token
            for attempt in range(5):
                try:
                    r = client.get(API, params=params)
                    r.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TimeoutException):
                    time.sleep(2 ** attempt)
            else:
                print("giving up after retries", file=sys.stderr)
                return
            data = r.json()
            rows = [x for x in map(parse_study, data.get("studies", [])) if x]
            upsert_trials(rows)
            token = data.get("nextPageToken")
            token_file.write_text(token or "DONE")
            pages += 1
            if pages % 10 == 0:
                print(f"{trial_count()} trials loaded", flush=True)
            if not token or (max_pages and pages >= max_pages):
                break
    print(f"done: {trial_count()} trials")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    st = sys.argv[2] if len(sys.argv) > 2 else "COMPLETED"
    fetch_all(n, st)
