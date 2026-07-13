"""Import trials from ClinicalTrials.gov."""

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from app.database.db import get_db, init_db, trial_count, upsert_trials

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
    "protocolSection.contactsLocationsModule",
])


def _date(d: dict | None) -> str | None:
    return (d or {}).get("date")


def _date_type(d: dict | None) -> str | None:
    return (d or {}).get("type")


def _parsed_date(value: str) -> date | None:
    """Parse full or reduced API dates."""
    try:
        if len(value) >= 10:
            return date.fromisoformat(value[:10])
        if len(value) >= 7:
            return date(int(value[:4]), int(value[5:7]), 1)
    except (TypeError, ValueError):
        pass
    return None


def _months(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    start_day, end_day = _parsed_date(start), _parsed_date(end)
    if not start_day or not end_day or end_day <= start_day:
        return None

    if len(start) >= 10 and len(end) >= 10:
        return round((end_day - start_day).days / (365.2425 / 12), 4)
    months = (end_day.year - start_day.year) * 12 + end_day.month - start_day.month
    return float(months) if months > 0 else None


_AGE_UNITS_IN_YEARS = {
    "year": 1.0,
    "years": 1.0,
    "month": 1 / 12,
    "months": 1 / 12,
    "week": 7 / 365.2425,
    "weeks": 7 / 365.2425,
    "day": 1 / 365.2425,
    "days": 1 / 365.2425,
    "hour": 1 / (365.2425 * 24),
    "hours": 1 / (365.2425 * 24),
    "minute": 1 / (365.2425 * 24 * 60),
    "minutes": 1 / (365.2425 * 24 * 60),
}


def _age_years(value: str | None) -> float | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s*", value)
    if not match:
        return None
    multiplier = _AGE_UNITS_IN_YEARS.get(match.group(2).lower())
    if multiplier is None:
        return None
    return round(float(match.group(1)) * multiplier, 6)


def _bool_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _join(values: list[str]) -> str:
    return "; ".join(value for value in values if value)


def _json_array(values: list) -> str:
    """Serialize a registry array."""
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _count_criteria(text: str) -> int:
    text = str(text or "").strip()
    if not text:
        return 0
    text = re.sub(r"(?i)\b(?:inclusion|exclusion)\s+criteria\s*:?\s*", "", text)
    marked = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", text)
    if marked:
        return len(marked)
    return len([part for part in re.split(r"\s*;\s*|\n+", text) if part.strip()])


def _criteria_counts(text: str | None) -> tuple[int | None, int | None]:
    """Count inclusion and exclusion criteria."""
    if not str(text or "").strip():
        return None, None
    text = str(text)
    pieces = re.split(r"(?i)\bexclusion\s+criteria\s*:?\s*", text, maxsplit=1)
    inclusion_text = re.sub(
        r"(?i)\binclusion\s+criteria\s*:?\s*", "", pieces[0]
    )
    exclusion_text = pieces[1] if len(pieces) > 1 else ""
    return _count_criteria(inclusion_text), _count_criteria(exclusion_text)


def parse_study(
    s: dict,
    *,
    fetched_at: str | None = None,
    include_raw: bool = False,
    feature_snapshot_date: str | None = None,
    feature_snapshot_kind: str = "LATEST",
) -> dict | None:
    p = s.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    nct = ident.get("nctId")
    if not nct:
        return None

    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    info = design.get("designInfo", {})
    enrollment = design.get("enrollmentInfo") or {}
    arms_mod = p.get("armsInterventionsModule", {})
    outcomes = p.get("outcomesModule", {})
    primary_outcomes = outcomes.get("primaryOutcomes") or []
    secondary_outcomes = outcomes.get("secondaryOutcomes") or []
    eligibility = p.get("eligibilityModule", {})
    sponsors = p.get("sponsorCollaboratorsModule", {})
    lead_sponsor = sponsors.get("leadSponsor") or {}
    locations = p.get("contactsLocationsModule", {}).get("locations") or []
    countries = sorted({loc.get("country") for loc in locations if loc.get("country")})

    start_struct = status.get("startDateStruct") or {}
    primary_completion_struct = status.get("primaryCompletionDateStruct") or {}
    completion_struct = status.get("completionDateStruct") or {}
    start = _date(start_struct)
    end = _date(completion_struct)
    minimum_age = eligibility.get("minimumAge")
    maximum_age = eligibility.get("maximumAge")
    conditions = p.get("conditionsModule", {}).get("conditions") or []
    interventions = [
        {"type": item.get("type") or "", "name": item.get("name") or ""}
        for item in arms_mod.get("interventions") or []
        if item.get("type") or item.get("name")
    ]
    primary_measures = [
        outcome.get("measure", "") for outcome in primary_outcomes
        if outcome.get("measure")
    ]
    secondary_measures = [
        outcome.get("measure", "") for outcome in secondary_outcomes
        if outcome.get("measure")
    ]
    primary_timeframes = [
        outcome.get("timeFrame", "") for outcome in primary_outcomes
        if outcome.get("timeFrame")
    ]
    secondary_timeframes = [
        outcome.get("timeFrame", "") for outcome in secondary_outcomes
        if outcome.get("timeFrame")
    ]
    eligibility_text = eligibility.get("eligibilityCriteria") or ""
    inclusion_count, exclusion_count = _criteria_counts(eligibility_text)
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
    snapshot_kind = str(feature_snapshot_kind or "LATEST").upper()
    if feature_snapshot_date is None and snapshot_kind == "INITIAL":
        feature_snapshot_date = status.get("studyFirstSubmitDate")

    return {
        "nct_id": nct,
        "title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "status": status.get("overallStatus"),
        "study_type": design.get("studyType"),
        "phase": ", ".join(design.get("phases") or []),
        "conditions": ", ".join(conditions),
        "conditions_json": _json_array(conditions),
        "interventions": _join([
            f"{item['type']}: {item['name']}".strip(": ")
            for item in interventions
        ]),
        "interventions_json": _json_array(interventions),
        "primary_outcomes": _join(primary_measures),
        "primary_outcomes_json": _json_array(primary_measures),
        "secondary_outcomes": _join(secondary_measures),
        "secondary_outcomes_json": _json_array(secondary_measures),
        "primary_outcome_timeframes": _join(primary_timeframes),
        "primary_outcome_timeframes_json": _json_array(primary_timeframes),
        "secondary_outcome_timeframes": _join(secondary_timeframes),
        "secondary_outcome_timeframes_json": _json_array(secondary_timeframes),
        "enrollment": enrollment.get("count"),
        "enrollment_type": enrollment.get("type"),
        "start_date": start,
        "start_date_type": _date_type(start_struct),
        "feature_start_date": start,
        "feature_start_date_type": _date_type(start_struct),
        "primary_completion_date": _date(primary_completion_struct),
        "primary_completion_date_type": _date_type(primary_completion_struct),
        "completion_date": end,
        "completion_date_type": _date_type(completion_struct),
        "duration_months": _months(start, end),
        "allocation": info.get("allocation"),
        "masking": (info.get("maskingInfo") or {}).get("masking"),
        "arms": len(arms_mod.get("armGroups") or []) or None,
        "intervention_model": info.get("interventionModel"),
        "primary_purpose": info.get("primaryPurpose"),
        "observational_model": info.get("observationalModel"),
        "time_perspective": info.get("timePerspective"),
        "sponsor": lead_sponsor.get("name"),
        "sponsor_class": lead_sponsor.get("class"),
        "collaborators_count": len(sponsors.get("collaborators") or []),
        "sex": eligibility.get("sex"),
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "minimum_age_years": _age_years(minimum_age),
        "maximum_age_years": _age_years(maximum_age),
        "healthy_volunteers": _bool_int(eligibility.get("healthyVolunteers")),
        "std_ages": ", ".join(eligibility.get("stdAges") or []),
        "sampling_method": eligibility.get("samplingMethod"),
        "study_population": eligibility.get("studyPopulation"),
        "site_count": len(locations),
        "country_count": len(countries),
        "countries": ", ".join(countries),
        "eligibility": eligibility_text,
        "inclusion_criteria_count": inclusion_count,
        "exclusion_criteria_count": exclusion_count,
        "feature_snapshot_date": feature_snapshot_date or fetched_at,
        "feature_snapshot_kind": snapshot_kind,
        "study_first_submit_date": status.get("studyFirstSubmitDate"),
        "source_updated_at": (
            status.get("lastUpdateSubmitDate")
            or _date(status.get("lastUpdatePostDateStruct"))
        ),
        "fetched_at": fetched_at,
        "raw": json.dumps(s, ensure_ascii=False, separators=(",", ":"))
        if include_raw else None,
    }


_LABEL_COLUMNS = (
    "status",
    "start_date",
    "start_date_type",
    "primary_completion_date",
    "primary_completion_date_type",
    "completion_date",
    "completion_date_type",
    "duration_months",
)


def _snapshot_studies(path: Path) -> list[dict]:
    """Load studies from a registry export."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        studies = payload
    elif isinstance(payload, dict) and isinstance(payload.get("studies"), list):
        studies = payload["studies"]
    elif isinstance(payload, dict) and "protocolSection" in payload:
        studies = [payload]
    else:
        raise ValueError(
            "snapshot file must contain a study, a list of studies, or {'studies': [...]}"
        )
    if not all(isinstance(study, dict) for study in studies):
        raise ValueError("every snapshot study must be a JSON object")
    return studies


def _existing_labels(nct_ids: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for start in range(0, len(nct_ids), 500):
        block = nct_ids[start:start + 500]
        with get_db() as db:
            rows = db.execute(
                f"SELECT nct_id,{','.join(_LABEL_COLUMNS)} FROM trials "
                f"WHERE nct_id IN ({','.join('?' * len(block))})",
                block,
            ).fetchall()
        result.update({row["nct_id"]: dict(row) for row in rows})
    return result


def import_snapshot_file(
    path: str | Path,
    *,
    snapshot_kind: str = "INITIAL",
    snapshot_date: str | None = None,
    include_raw: bool = False,
) -> int:
    """Import feature snapshots while retaining known labels."""
    init_db()
    path = Path(path)
    studies = _snapshot_studies(path)
    imported_at = datetime.now(timezone.utc).isoformat()
    kind = str(snapshot_kind or "INITIAL").upper()
    rows = [
        row
        for row in (
            parse_study(
                study,
                fetched_at=imported_at,
                include_raw=include_raw,
                feature_snapshot_date=snapshot_date,
                feature_snapshot_kind=kind,
            )
            for study in studies
        )
        if row
    ]
    labels = _existing_labels([row["nct_id"] for row in rows])
    for row in rows:
        existing = labels.get(row["nct_id"])
        if existing and existing.get("duration_months") is not None:
            for column in _LABEL_COLUMNS:
                row[column] = existing.get(column)
    for start in range(0, len(rows), 500):
        upsert_trials(rows[start:start + 500])
    return len(rows)


def _token_path(statuses: str) -> Path:
    suffix = statuses[:12].lower().replace("/", "_").replace("\\", "_")
    return TOKEN_FILE.with_suffix("." + suffix)


def fetch_all(
    max_pages: int | None = None,
    statuses: str = "COMPLETED",
    *,
    force: bool = False,
    refresh: bool = False,
    include_raw: bool = False,
    update_fts: bool = True,
):
    """Fetch studies with checkpoint resume support."""
    init_db()
    token_file = _token_path(statuses)
    token = token_file.read_text().strip() if token_file.exists() else None
    if force or (refresh and token == "DONE"):
        token = None
        token_file.unlink(missing_ok=True)
    if token == "DONE":
        print(f"already complete: {trial_count()} trials (use --refresh to refetch)")
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
                    response = client.get(API, params=params)
                    response.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TimeoutException) as exc:
                    if attempt == 4:
                        print(f"giving up after retries: {exc}", file=sys.stderr)
                        return
                    time.sleep(2 ** attempt)

            data = response.json()
            fetched_at = datetime.now(timezone.utc).isoformat()
            rows = [
                row for row in (
                    parse_study(study, fetched_at=fetched_at, include_raw=include_raw)
                    for study in data.get("studies", [])
                )
                if row
            ]
            if rows:
                upsert_trials(rows, update_fts=update_fts)
            token = data.get("nextPageToken")
            token_file.write_text(token or "DONE")
            pages += 1
            if pages % 10 == 0:
                print(f"{trial_count()} trials loaded", flush=True)
            if not token or (max_pages and pages >= max_pages):
                break
    print(f"done: {trial_count()} trials")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_pages", nargs="?", type=int)
    parser.add_argument("statuses", nargs="?", default="COMPLETED")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refetch when this status checkpoint is already DONE",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard any DONE or partial checkpoint and start at page one",
    )
    parser.add_argument(
        "--store-raw",
        action="store_true",
        help="store selected raw API modules (can substantially increase DB size)",
    )
    parser.add_argument(
        "--skip-fts",
        action="store_true",
        help="skip FTS rewrites when refreshing only non-search enrichment fields",
    )
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        help="import an exported initial/versioned study JSON file instead of fetching",
    )
    parser.add_argument(
        "--snapshot-kind",
        default="INITIAL",
        help="provenance marker for --snapshot-file rows (default: INITIAL)",
    )
    parser.add_argument(
        "--snapshot-date",
        help="ISO date/timestamp represented by --snapshot-file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.snapshot_file:
        count = import_snapshot_file(
            args.snapshot_file,
            snapshot_kind=args.snapshot_kind,
            snapshot_date=args.snapshot_date,
            include_raw=args.store_raw,
        )
        print(f"imported {count} {args.snapshot_kind.upper()} feature snapshots")
    else:
        fetch_all(
            args.max_pages,
            args.statuses,
            force=args.force,
            refresh=args.refresh,
            include_raw=args.store_raw,
            update_fts=not args.skip_fts,
        )
