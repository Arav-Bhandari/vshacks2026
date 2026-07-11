"""Shared feature extraction for USDM protocols and DB trial rows."""
import datetime
import math
import re

from app.ml.embeddings import EMB_DIM

STRUCT_FEATURES = [
    "phase_num",
    "log_enrollment",
    "n_arms",
    "n_endpoints_primary",
    "n_endpoints_secondary",
    "has_secondary_outcomes",
    "n_inclusion",
    "n_exclusion",
    "randomized",
    "blinded",
    "n_conditions",
    "n_interventions",
    "is_drug",
    "is_biological",
    "is_device",
    "is_behavioral",
    "is_procedure",
    "is_other_intervention",
    "is_academic",
    "start_year",
]

FEATURE_ORDER = STRUCT_FEATURES + [f"emb_{i}" for i in range(EMB_DIM)]

_ACADEMIC_KEYWORDS = (
    "university", "hospital", "institute", "college", "center",
    "centre", "foundation", "nih", "national institutes",
)

_TYPE_FLAGS = {
    "DRUG": "is_drug",
    "BIOLOGICAL": "is_biological",
    "DEVICE": "is_device",
    "BEHAVIORAL": "is_behavioral",
    "PROCEDURE": "is_procedure",
}


def _split(text) -> list[str]:
    if not text:
        return []
    return [p for p in str(text).split("; ") if p.strip()]


def _count_criteria(text: str) -> int:
    # eligibility bullets use "* "/"- "/"1." markers, not "; "
    items = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", text)
    if items:
        return len(items)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return len([l for l in lines if "criteria" not in l.lower()])


def _phase_num(text) -> int:
    digits = [int(d) for d in re.findall(r"[1-4]", str(text or ""))]
    return max(digits) if digits else 0


def _sponsor_is_academic(sponsor) -> int:
    s = str(sponsor or "").lower()
    return 1 if any(k in s for k in _ACADEMIC_KEYWORDS) else 0


def _intervention_flags_from_text(interventions: str) -> dict:
    flags = {v: 0 for v in _TYPE_FLAGS.values()}
    other = 0
    for part in _split(interventions):
        if ":" in part:
            kind = part.split(":", 1)[0].strip().upper()
            key = _TYPE_FLAGS.get(kind)
            if key:
                flags[key] = 1
                continue
        other = 1
    flags["is_other_intervention"] = other
    return flags


def _year_from_date(date_str) -> int:
    m = re.match(r"(\d{4})", str(date_str or ""))
    return int(m.group(1)) if m else datetime.date.today().year


def trial_row_features(row: dict) -> dict:
    allocation = str(row.get("allocation") or "").lower()
    masking = str(row.get("masking") or "").lower()
    eligibility = row.get("eligibility") or ""
    incl = re.split(r"exclusion criteria", eligibility, flags=re.IGNORECASE)
    n_inclusion = _count_criteria(incl[0]) or 1
    n_exclusion = _count_criteria(incl[1]) if len(incl) > 1 else 0

    f = {
        "phase_num": _phase_num(row.get("phase")),
        "log_enrollment": math.log1p(row.get("enrollment") or 0),
        "n_arms": row.get("arms") or 1,
        "n_endpoints_primary": len(_split(row.get("primary_outcomes"))) or 1,
        "n_endpoints_secondary": len(_split(row.get("secondary_outcomes"))),
        "has_secondary_outcomes": 1 if row.get("secondary_outcomes") else 0,
        "n_inclusion": n_inclusion,
        "n_exclusion": n_exclusion,
        "randomized": 1 if "randomiz" in allocation else 0,
        "blinded": 1 if masking and "open" not in masking and "none" not in masking else 0,
        "n_conditions": len(_split(row.get("conditions"))) or 1,
        "n_interventions": len(_split(row.get("interventions"))) or 1,
        "is_academic": _sponsor_is_academic(row.get("sponsor")),
        "start_year": _year_from_date(row.get("start_date")),
    }
    f.update(_intervention_flags_from_text(row.get("interventions")))
    return {k: f[k] for k in STRUCT_FEATURES}


def usdm_features(usdm: dict, burden: dict | None = None) -> dict:
    study = (usdm or {}).get("study") or {}
    population = study.get("population") or {}
    criteria = population.get("criteria") or {}
    design = study.get("design") or {}
    objectives = study.get("objectives") or []
    interventions = study.get("interventions") or []

    n_primary = sum(len(o.get("endpoints") or []) for o in objectives if o.get("level") == "primary")
    n_secondary = sum(len(o.get("endpoints") or []) for o in objectives if o.get("level") == "secondary")

    allocation = str(design.get("allocation") or "").lower()
    masking = str(design.get("masking") or "").lower()

    # USDM interventions rarely carry a type prefix; default to drug
    flags = {v: 0 for v in _TYPE_FLAGS.values()}
    flags["is_other_intervention"] = 0
    flags["is_drug"] = 1

    f = {
        "phase_num": _phase_num(study.get("phase")),
        "log_enrollment": math.log1p(population.get("plannedEnrollment") or 150),
        "n_arms": len(study.get("arms") or []) or 2,
        "n_endpoints_primary": n_primary or 1,
        "n_endpoints_secondary": n_secondary,
        "has_secondary_outcomes": 1 if n_secondary else 0,
        "n_inclusion": len(criteria.get("inclusion") or []) or 1,
        "n_exclusion": len(criteria.get("exclusion") or []),
        "randomized": 1 if "randomiz" in allocation else 0,
        "blinded": 1 if masking and "open" not in masking and "none" not in masking else 0,
        "n_conditions": len(study.get("conditions") or []) or 1,
        "n_interventions": len(interventions) or 1,
        "is_academic": 0,
        "start_year": datetime.date.today().year,
    }
    f.update(flags)
    return {k: f[k] for k in STRUCT_FEATURES}


def to_vector(struct_feats: dict, emb) -> list[float]:
    return [struct_feats[k] for k in STRUCT_FEATURES] + [float(v) for v in emb]
