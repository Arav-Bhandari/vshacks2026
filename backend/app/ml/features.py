"""Shared numeric feature extraction for USDM protocols and DB trial rows."""
import re

FEATURE_ORDER = [
    "phase_num",
    "enrollment",
    "n_arms",
    "n_endpoints_primary",
    "n_endpoints_secondary",
    "n_inclusion",
    "n_exclusion",
    "n_visits",
    "randomized",
    "blinded",
    "n_conditions",
    "n_interventions",
]

# median-ish defaults for missing values
_DEFAULTS = {
    "phase_num": 0,
    "enrollment": 150,
    "n_arms": 2,
    "n_endpoints_primary": 1,
    "n_endpoints_secondary": 3,
    "n_inclusion": 6,
    "n_exclusion": 6,
    "n_visits": 8,
    "randomized": 1,
    "blinded": 0,
    "n_conditions": 1,
    "n_interventions": 1,
}


def _split(text) -> list[str]:
    if not text:
        return []
    return [p for p in str(text).split("; ") if p.strip()]


def _phase_num(text) -> int:
    if not text:
        return 0
    m = re.search(r"[1-4]", str(text))
    return int(m.group()) if m else 0


def usdm_features(usdm: dict, burden: dict | None = None) -> dict:
    study = (usdm or {}).get("study") or {}
    population = study.get("population") or {}
    criteria = population.get("criteria") or {}
    design = study.get("design") or {}
    objectives = study.get("objectives") or []

    n_primary = sum(len(o.get("endpoints") or []) for o in objectives if o.get("level") == "primary")
    n_secondary = sum(len(o.get("endpoints") or []) for o in objectives if o.get("level") == "secondary")

    allocation = str(design.get("allocation") or "").lower()
    masking = str(design.get("masking") or "").lower()

    f = {
        "phase_num": _phase_num(study.get("phase")),
        "enrollment": population.get("plannedEnrollment") or _DEFAULTS["enrollment"],
        "n_arms": len(study.get("arms") or []) or _DEFAULTS["n_arms"],
        "n_endpoints_primary": n_primary or _DEFAULTS["n_endpoints_primary"],
        "n_endpoints_secondary": n_secondary or _DEFAULTS["n_endpoints_secondary"],
        "n_inclusion": len(criteria.get("inclusion") or []) or _DEFAULTS["n_inclusion"],
        "n_exclusion": len(criteria.get("exclusion") or []) or _DEFAULTS["n_exclusion"],
        "n_visits": len((study.get("scheduleOfActivities") or {}).get("visits") or [])
        or _DEFAULTS["n_visits"],
        "randomized": 1 if "randomiz" in allocation else _DEFAULTS["randomized"],
        "blinded": 1 if masking and "open" not in masking and "none" not in masking else _DEFAULTS["blinded"],
        "n_conditions": len(study.get("conditions") or []) or _DEFAULTS["n_conditions"],
        "n_interventions": len(study.get("interventions") or []) or _DEFAULTS["n_interventions"],
    }
    return {k: f[k] for k in FEATURE_ORDER}


def trial_row_features(row: dict) -> dict:
    allocation = str(row.get("allocation") or "").lower()
    masking = str(row.get("masking") or "").lower()
    eligibility = row.get("eligibility") or ""
    # inclusion/exclusion split on markers when present, else whole text counts as 1 block
    incl = re.split(r"exclusion criteria", eligibility, flags=re.IGNORECASE)
    n_inclusion = len(_split(incl[0])) or _DEFAULTS["n_inclusion"]
    n_exclusion = len(_split(incl[1])) if len(incl) > 1 else _DEFAULTS["n_exclusion"]

    f = {
        "phase_num": _phase_num(row.get("phase")),
        "enrollment": row.get("enrollment") or _DEFAULTS["enrollment"],
        "n_arms": row.get("arms") or _DEFAULTS["n_arms"],
        "n_endpoints_primary": len(_split(row.get("primary_outcomes"))) or _DEFAULTS["n_endpoints_primary"],
        "n_endpoints_secondary": len(_split(row.get("secondary_outcomes"))) or _DEFAULTS["n_endpoints_secondary"],
        "n_inclusion": n_inclusion,
        "n_exclusion": n_exclusion,
        "n_visits": _DEFAULTS["n_visits"],
        "randomized": 1 if "randomiz" in allocation else _DEFAULTS["randomized"],
        "blinded": 1 if masking and "open" not in masking and "none" not in masking else _DEFAULTS["blinded"],
        "n_conditions": len(_split(row.get("conditions"))) or _DEFAULTS["n_conditions"],
        "n_interventions": len(_split(row.get("interventions"))) or _DEFAULTS["n_interventions"],
    }
    return {k: f[k] for k in FEATURE_ORDER}
