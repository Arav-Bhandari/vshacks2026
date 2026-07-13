"""Extract shared duration model features."""
from __future__ import annotations

import datetime
import json
import math
import re
from typing import Any

from app.ml.embeddings import EMB_DIM


STRUCT_FEATURES = [
    "phase_num",
    "phase_missing",
    "log_enrollment",
    "enrollment_missing",
    "n_arms",
    "arms_missing",
    "n_endpoints_primary",
    "n_endpoints_secondary",
    "has_secondary_outcomes",
    "endpoints_missing",
    "n_inclusion",
    "n_exclusion",
    "eligibility_missing",
    "randomized",
    "allocation_missing",
    "blinded",
    "masking_missing",
    "n_conditions",
    "conditions_missing",
    "n_interventions",
    "interventions_missing",
    "is_drug",
    "is_biological",
    "is_device",
    "is_behavioral",
    "is_procedure",
    "is_other_intervention",
    "intervention_type_missing",
    "is_academic",
    "is_industry",
    "is_government",
    "sponsor_missing",
    "start_year",
    "start_date_missing",
    "is_interventional",
    "is_observational",
    "study_type_missing",
    "purpose_treatment",
    "purpose_prevention",
    "purpose_diagnostic",
    "purpose_supportive_care",
    "purpose_screening",
    "purpose_basic_science",
    "purpose_other",
    "primary_purpose_missing",
    "model_parallel",
    "model_crossover",
    "model_factorial",
    "model_single_group",
    "model_sequential",
    "model_other",
    "intervention_model_missing",
    "female_only",
    "male_only",
    "sex_all",
    "sex_missing",
    "minimum_age_years",
    "maximum_age_years",
    "age_range_years",
    "age_missing",
    "healthy_volunteers",
    "healthy_volunteers_missing",
    "n_sites",
    "sites_missing",
    "n_countries",
    "countries_missing",
    "n_collaborators",
    "collaborators_missing",
    "primary_outcome_timeframe_months",
    "secondary_outcome_timeframe_months",
    "outcome_timeframes_missing",
]

FEATURE_ORDER = STRUCT_FEATURES + [f"emb_{i}" for i in range(EMB_DIM)]

_ACADEMIC_KEYWORDS = (
    "university", "hospital", "institute", "college", "center", "centre",
    "foundation", "nih", "national institutes",
)
_GOVERNMENT_KEYWORDS = (
    "government", "federal", "ministry", "department of", "national health",
    "veterans affairs", "nih", "national institutes",
)
_TYPE_FLAGS = {
    "DRUG": "is_drug",
    "BIOLOGICAL": "is_biological",
    "BIOLOGIC": "is_biological",
    "DEVICE": "is_device",
    "BEHAVIORAL": "is_behavioral",
    "BEHAVIOURAL": "is_behavioral",
    "PROCEDURE": "is_procedure",
}
_FALSE_VALUES = {"0", "false", "no", "n", "none", "not allowed", "not applicable"}
_TRUE_VALUES = {"1", "true", "yes", "y", "allowed", "accepts healthy volunteers"}
_TEMPLATE_ENUM_VALUES = {
    "interventional|observational|expanded_access|",
    "industry|nih|fed|other|",
    "primary|secondary",
}
_TEMPLATE_ENUM_TOKENS = {
    "interventional", "observational", "expanded_access",
    "industry", "nih", "fed", "other", "primary", "secondary",
}


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _first(mapping: dict, *keys: str, default=None):
    for key in keys:
        if key in mapping and not _missing(mapping[key]):
            return mapping[key]
    return default


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value[:1] not in ("[", "{"):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _semantic_prune(value: Any) -> Any:
    """Remove empty template values."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        normalized = stripped.lower()
        enum_tokens = {token for token in normalized.split("|") if token}
        is_template_enum = (
            normalized in _TEMPLATE_ENUM_VALUES
            or (len(enum_tokens) > 1 and enum_tokens <= _TEMPLATE_ENUM_TOKENS)
        )
        return None if not stripped or is_template_enum else stripped
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            pruned = _semantic_prune(item)
            if pruned is not None:
                result[key] = pruned
        return result or None
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            pruned = _semantic_prune(item)
            if pruned is not None:
                result.append(pruned)
        return result or None
    return value


def _items(value: Any, *, comma: bool = False) -> list[Any]:
    """Return a list from JSON/list/registry-delimited values."""
    value = _semantic_prune(_json_value(value))
    if _missing(value):
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if not _missing(v)]
    separator = r"\s*;\s*"
    if comma:
        separator = r"\s*[;,]\s*"
    return [v.strip() for v in re.split(separator, str(value)) if v.strip()]


def _structured_items(
    row: dict,
    structured_key: str,
    legacy_key: str,
    *,
    comma: bool = False,
) -> list[Any]:
    """Read structured or legacy lists."""
    structured = row.get(structured_key)
    if structured is not None:
        parsed = _json_value(structured)
        if isinstance(parsed, list):
            return _items(parsed, comma=comma)
    return _items(row.get(legacy_key), comma=comma)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or "").replace(",", ""))
    return float(match.group()) if match else None


def _count_criteria(text: str) -> int:
    text = str(text or "").strip()
    if not text:
        return 0
    text = re.sub(r"(?i)\b(?:inclusion|exclusion)\s+criteria\s*:?", "", text)
    marked = re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", text)
    if marked:
        return len(marked)
    parts = [p.strip() for p in re.split(r"\s*;\s*|\n+", text) if p.strip()]
    return len(parts)


def _criteria_from_text(text: Any) -> tuple[list[str], list[str]]:
    raw = str(text or "")
    pieces = re.split(r"(?i)\bexclusion\s+criteria\s*:?", raw, maxsplit=1)
    inclusion_text = re.sub(r"(?i)\binclusion\s+criteria\s*:?", "", pieces[0])
    exclusion_text = pieces[1] if len(pieces) > 1 else ""
    return ([""] * _count_criteria(inclusion_text), [""] * _count_criteria(exclusion_text))


def _phase_num(value: Any) -> int:
    values = [int(d) for d in re.findall(r"[1-4]", str(value or ""))]
    if values:
        return max(values)
    roman = re.findall(r"\b(?:iv|iii|ii|i)\b", str(value or "").lower())
    return max(({"i": 1, "ii": 2, "iii": 3, "iv": 4}[v] for v in roman), default=0)


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _year(date_value: Any) -> int | None:
    if isinstance(date_value, dict):
        date_value = _first(date_value, "date", "value")
    match = re.match(r"\s*((?:19|20)\d{2})", str(date_value or ""))
    return int(match.group(1)) if match else None


def _date_parts(date_value: Any) -> tuple[int, int] | None:
    if isinstance(date_value, dict):
        date_value = _first(date_value, "date", "value")
    match = re.match(r"\s*((?:19|20)\d{2})(?:-(\d{1,2}))?", str(date_value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 1)


def _months_between(start: Any, end: Any) -> float | None:
    start_parts, end_parts = _date_parts(start), _date_parts(end)
    if not start_parts or not end_parts:
        return None
    months = (end_parts[0] - start_parts[0]) * 12 + end_parts[1] - start_parts[1]
    return float(months) if months > 0 else None


def _duration_months(value: Any) -> float | None:
    """Parse a duration as months."""
    if _missing(value):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, dict):
        amount = _first(value, "value", "count", "duration", "length")
        unit = _first(value, "unit", "units")
        if amount is not None and unit:
            value = f"{amount} {unit}"
        else:
            value = " ".join(str(v) for v in value.values() if not _missing(v))
    text = str(value).lower()
    iso = re.search(r"\bp(?:(\d+(?:\.\d+)?)y)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)w)?(?:(\d+(?:\.\d+)?)d)?\b", text)
    if iso and any(iso.groups()):
        years, months, weeks, days = (float(v or 0) for v in iso.groups())
        return years * 12 + months + weeks * 7 / 30.4375 + days / 30.4375
    units = {
        "year": 12.0, "yr": 12.0,
        "month": 1.0, "mo": 1.0,
        "week": 7 / 30.4375, "wk": 7 / 30.4375,
        "day": 1 / 30.4375,
        "hour": 1 / (24 * 30.4375),
    }
    candidates = []
    for amount, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*[-–]?\s*"
        r"(years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?)\b",
        text,
    ):
        singular = unit.rstrip("s")
        candidates.append(float(amount) * units[singular])
    for unit, amount in re.findall(
        r"\b(years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?)\s*[-–]?\s*"
        r"(\d+(?:\.\d+)?)",
        text,
    ):
        singular = unit.rstrip("s")
        candidates.append(float(amount) * units[singular])
    for unit, _start, end in re.findall(
        r"\b(years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:-|–|to|through)\s*"
        r"(?:years?|yrs?|months?|mos?|weeks?|wks?|days?|hours?)?\s*"
        r"(\d+(?:\.\d+)?)",
        text,
    ):
        singular = unit.rstrip("s")
        candidates.append(float(end) * units[singular])
    return max(candidates) if candidates else None


def _max_duration(values: Any) -> float | None:
    parsed = [_duration_months(value) for value in _items(values, comma=False)]
    parsed = [value for value in parsed if value is not None]
    return max(parsed) if parsed else None


def _age_years(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").lower()
    number = _number(text)
    if number is None:
        return None
    if "week" in text:
        return number / 52.1775
    if "month" in text:
        return number / 12
    if "day" in text:
        return number / 365.25
    return number


def _age_pair(minimum: Any, maximum: Any, age_range: Any = None) -> tuple[float | None, float | None]:
    min_years, max_years = _age_years(minimum), _age_years(maximum)
    if min_years is None and max_years is None and age_range:
        matches = re.findall(r"\d+(?:\.\d+)?", str(age_range).replace(",", ""))
        if matches:
            min_years = _age_years(f"{matches[0]} {age_range}")
            max_years = _age_years(f"{matches[-1]} {age_range}") if len(matches) > 1 else None
    return min_years, max_years


def _interventions(value: Any) -> list[dict[str, str]]:
    result = []
    for item in _items(value):
        if isinstance(item, dict):
            name = _first(item, "name", "interventionName", "label", default="")
            kind = _first(item, "type", "interventionType", "category", default="")
        else:
            text = str(item).strip()
            prefix, separator, remainder = text.partition(":")
            if separator and prefix.strip().upper() in (set(_TYPE_FLAGS) | {"OTHER"}):
                kind, name = prefix.strip(), remainder.strip()
            else:
                kind, name = "", text
        if name or kind:
            result.append({"name": str(name), "type": str(kind)})
    return result


def _endpoint_data(objectives: Any) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    primary, secondary, primary_tf, secondary_tf = [], [], [], []
    for objective in _items(objectives):
        if not isinstance(objective, dict):
            continue
        level = str(_first(objective, "level", "type", default="")).lower()
        endpoints = _items(objective.get("endpoints"))
        target, timeframes = (secondary, secondary_tf) if "secondary" in level else (primary, primary_tf)
        target.extend(endpoints)
        for endpoint in endpoints:
            if isinstance(endpoint, dict):
                timeframe = _first(endpoint, "timeframe", "timeFrame", "time_frame")
                if not _missing(timeframe):
                    timeframes.append(timeframe)
    return primary, secondary, primary_tf, secondary_tf


def _schedule_data(schedule: Any) -> tuple[list[Any], list[Any], bool]:
    if isinstance(schedule, dict):
        visits_value = _first(schedule, "visits", "encounters", default=[])
        explicitly_present = any(key in schedule for key in ("visits", "encounters"))
    else:
        visits_value = schedule
        explicitly_present = not _missing(schedule)
    visits = _items(visits_value)
    procedures = []
    for visit in visits:
        if isinstance(visit, dict):
            procedures.extend(_items(_first(visit, "procedures", "activities", default=[])))
    return visits, procedures, explicitly_present


def _canonical_row(row: dict) -> dict:
    interventions = _interventions(_structured_items(
        row, "interventions_json", "interventions"
    ))
    conditions = _structured_items(
        row, "conditions_json", "conditions", comma=True
    )
    primary = _structured_items(
        row, "primary_outcomes_json", "primary_outcomes"
    )
    secondary = _structured_items(
        row, "secondary_outcomes_json", "secondary_outcomes"
    )
    parsed_inclusion, parsed_exclusion = _criteria_from_text(row.get("eligibility"))
    inclusion_count = _number(row.get("inclusion_criteria_count"))
    exclusion_count = _number(row.get("exclusion_criteria_count"))
    if inclusion_count is None:
        inclusion_count = len(parsed_inclusion)
    if exclusion_count is None:
        exclusion_count = len(parsed_exclusion)

    start = _first(row, "feature_start_date", "start_date", "planned_start_date")
    start_type = _first(
        row, "feature_start_date_type", "start_date_type", "planned_start_date_type"
    )
    primary_completion = _first(row, "primary_completion_date", "planned_primary_completion_date")
    primary_completion_type = _first(
        row, "primary_completion_date_type", "planned_primary_completion_date_type"
    )
    completion = row.get("completion_date")
    completion_type = row.get("completion_date_type")
    estimated_end = None
    if "estimat" in str(primary_completion_type or "").lower():
        estimated_end = primary_completion
    elif "estimat" in str(completion_type or "").lower():
        estimated_end = completion
    planned_duration = _months_between(start, estimated_end)

    minimum_age = _first(row, "minimum_age_years", "minimum_age", "min_age")
    maximum_age = _first(row, "maximum_age_years", "maximum_age", "max_age")
    min_age, max_age = _age_pair(minimum_age, maximum_age)
    n_visits = _number(_first(row, "visit_count", "n_visits"))
    n_procedures = _number(_first(row, "procedure_count", "n_procedures"))

    enrollment_type = row.get("enrollment_type")
    enrollment = None if "actual" in str(enrollment_type or "").lower() else row.get("enrollment")

    return {
        "phase": row.get("phase"),
        "enrollment": enrollment,
        "enrollment_type": enrollment_type,
        "arms": row.get("arms"),
        "primary": primary,
        "secondary": secondary,
        "n_inclusion": max(0, int(inclusion_count)),
        "n_exclusion": max(0, int(exclusion_count)),
        "eligibility_present": not _missing(row.get("eligibility")),
        "allocation": row.get("allocation"),
        "masking": row.get("masking"),
        "conditions": conditions,
        "interventions": interventions,
        "sponsor": row.get("sponsor"),
        "sponsor_class": _first(row, "sponsor_class", "sponsor_type"),
        "start": start,
        "start_type": start_type,
        "study_type": row.get("study_type"),
        "primary_purpose": row.get("primary_purpose"),
        "intervention_model": row.get("intervention_model"),
        "sex": row.get("sex"),
        "min_age": min_age,
        "max_age": max_age,
        "healthy_volunteers": row.get("healthy_volunteers"),
        "sites": _number(_first(row, "site_count", "n_sites")),
        "countries": _number(_first(row, "country_count", "n_countries")),
        "collaborators": _number(_first(row, "collaborators_count", "n_collaborators")),
        "visits": n_visits,
        "visits_present": n_visits is not None,
        "procedures": n_procedures,
        "procedures_present": n_procedures is not None,
        "primary_timeframe": _max_duration(_structured_items(
            row,
            "primary_outcome_timeframes_json",
            "primary_outcome_timeframes",
        ) or _items(row.get("primary_outcomes_timeframes"))),
        "secondary_timeframe": _max_duration(_structured_items(
            row,
            "secondary_outcome_timeframes_json",
            "secondary_outcome_timeframes",
        ) or _items(row.get("secondary_outcomes_timeframes"))),
        "planned_duration": planned_duration,
    }


def _canonical_usdm(usdm: dict) -> dict:
    cleaned = _semantic_prune(usdm or {}) or {}
    study = cleaned.get("study") or {}
    population = study.get("population") or {}
    eligibility = study.get("eligibility") or {}
    criteria = population.get("criteria") or {}
    inclusion = _items(_first(
        criteria, "inclusion", "inclusionCriteria", "inclusion_criteria",
        default=_first(eligibility, "inclusion", "inclusionCriteria", "inclusion_criteria", default=[]),
    ))
    exclusion = _items(_first(
        criteria, "exclusion", "exclusionCriteria", "exclusion_criteria",
        default=_first(eligibility, "exclusion", "exclusionCriteria", "exclusion_criteria", default=[]),
    ))
    design = study.get("design") or {}
    primary, secondary, primary_tf, secondary_tf = _endpoint_data(study.get("objectives"))
    visits, procedures, schedule_present = _schedule_data(
        _first(study, "scheduleOfActivities", "schedule_of_activities")
    )

    min_age_value = _first(population, "minimumAge", "minAge", "minimum_age", "min_age")
    max_age_value = _first(population, "maximumAge", "maxAge", "maximum_age", "max_age")
    if min_age_value is None:
        min_age_value = _first(eligibility, "minimumAge", "minAge", "minimum_age", "min_age")
    if max_age_value is None:
        max_age_value = _first(eligibility, "maximumAge", "maxAge", "maximum_age", "max_age")
    min_age, max_age = _age_pair(
        min_age_value, max_age_value, _first(population, "ageRange", "age_range")
    )

    sponsor_value = _first(study, "sponsor", "leadSponsor", default={})
    if isinstance(sponsor_value, dict):
        sponsor = _first(sponsor_value, "name", "label")
        sponsor_class = _first(sponsor_value, "class", "type", "category")
    else:
        sponsor = sponsor_value
        sponsor_class = _first(study, "sponsorClass", "sponsorType", "sponsor_class")
    sites_value = _first(study, "sites", "studySites", "study_sites")
    countries_value = _first(study, "countries", "studyCountries", "study_countries")
    collaborators_value = _first(study, "collaborators", "studyCollaborators")

    start_value = _first(study, "startDate", "plannedStartDate", "start_date")
    start_type = _first(study, "startDateType", "plannedStartDateType", "start_date_type")
    if isinstance(start_value, dict):
        start_type = _first(start_value, "type", "dateType", default=start_type)

    estimated_duration = _first(
        study, "estimatedDuration", "plannedDuration", "durationMonths", "estimated_duration"
    )
    if "durationMonths" in study and not _missing(study.get("durationMonths")):
        planned_duration = _number(study["durationMonths"])
    else:
        planned_duration = _duration_months(estimated_duration)

    return {
        "phase": study.get("phase"),
        "enrollment": _first(population, "plannedEnrollment", "planned_enrollment", default=_first(
            study, "plannedEnrollment", "planned_enrollment", "enrollment"
        )),
        "enrollment_type": "ANTICIPATED",
        "arms": len(_items(study.get("arms"))) if "arms" in study else None,
        "primary": primary,
        "secondary": secondary,
        "n_inclusion": len(inclusion),
        "n_exclusion": len(exclusion),
        "eligibility_present": bool(criteria or eligibility),
        "allocation": design.get("allocation"),
        "masking": design.get("masking"),
        "conditions": _items(study.get("conditions"), comma=True),
        "interventions": _interventions(study.get("interventions")),
        "sponsor": sponsor,
        "sponsor_class": sponsor_class,
        "start": start_value,
        "start_type": start_type,
        "study_type": _first(study, "studyType", "study_type"),
        "primary_purpose": _first(design, "primaryPurpose", "primary_purpose", default=_first(
            study, "primaryPurpose", "primary_purpose"
        )),
        "intervention_model": _first(design, "interventionModel", "intervention_model"),
        "sex": _first(population, "sex", "gender", default=study.get("sex")),
        "min_age": min_age,
        "max_age": max_age,
        "healthy_volunteers": _first(
            population, "healthyVolunteers", "healthy_volunteers",
            default=_first(eligibility, "healthyVolunteers", "healthy_volunteers"),
        ),
        "sites": _number(sites_value) if not isinstance(sites_value, (list, tuple, set)) else len(sites_value),
        "countries": (_number(countries_value) if not isinstance(countries_value, (list, tuple, set))
                      else len(countries_value)),
        "collaborators": (_number(collaborators_value)
                          if not isinstance(collaborators_value, (list, tuple, set))
                          else len(collaborators_value)),
        "visits": len(visits) if schedule_present else None,
        "visits_present": schedule_present,
        "procedures": len(procedures) if schedule_present else None,
        "procedures_present": schedule_present,
        "primary_timeframe": _max_duration(primary_tf),
        "secondary_timeframe": _max_duration(secondary_tf),
        "planned_duration": planned_duration,
    }


def _model_features(canonical: dict) -> dict:
    phase_missing = _missing(canonical["phase"])
    enrollment = _number(canonical["enrollment"])
    arms = _number(canonical["arms"])
    primary = canonical["primary"]
    secondary = canonical["secondary"]
    conditions = canonical["conditions"]
    interventions = canonical["interventions"]
    allocation = str(canonical["allocation"] or "").lower()
    masking = str(canonical["masking"] or "").lower()

    intervention_flags = {feature: 0 for feature in _TYPE_FLAGS.values()}
    other_intervention = 0
    known_intervention_types = 0
    for intervention in interventions:
        kind = str(intervention.get("type") or "").strip().upper().replace(" ", "_")
        flag = _TYPE_FLAGS.get(kind)
        if flag:
            intervention_flags[flag] = 1
            known_intervention_types += 1
        elif kind:
            other_intervention = 1
            known_intervention_types += 1

    sponsor = str(canonical["sponsor"] or "").lower()
    sponsor_class = str(canonical["sponsor_class"] or "").lower()
    sponsor_missing = not sponsor and not sponsor_class
    is_industry = "industry" in sponsor_class
    is_government = any(key in sponsor_class or key in sponsor for key in _GOVERNMENT_KEYWORDS)
    is_academic = any(key in sponsor_class or key in sponsor for key in _ACADEMIC_KEYWORDS)

    study_type = str(canonical["study_type"] or "").lower()
    purpose = str(canonical["primary_purpose"] or "").lower().replace("_", " ")
    model = str(canonical["intervention_model"] or "").lower().replace("_", " ")
    sex = str(canonical["sex"] or "").strip().lower()
    sex_missing = not sex
    female_only = bool(sex and ("female" in sex or "women" in sex) and "all" not in sex)
    male_only = bool(sex and ("male" in sex or "men" in sex) and "female" not in sex and "women" not in sex and "all" not in sex)
    sex_all = bool(sex and any(token in sex for token in ("all", "both")))

    min_age, max_age = canonical["min_age"], canonical["max_age"]
    age_missing = min_age is None and max_age is None
    age_range = max(0.0, max_age - min_age) if min_age is not None and max_age is not None else 0.0
    healthy = _bool(canonical["healthy_volunteers"])
    start_year = _year(canonical["start"])
    primary_timeframe = canonical["primary_timeframe"]
    secondary_timeframe = canonical["secondary_timeframe"]

    f = {
        "phase_num": _phase_num(canonical["phase"]),
        "phase_missing": int(phase_missing),
        "log_enrollment": math.log1p(max(0.0, enrollment or 0.0)),
        "enrollment_missing": int(enrollment is None),
        "n_arms": arms or 0.0,
        "arms_missing": int(arms is None),
        "n_endpoints_primary": len(primary),
        "n_endpoints_secondary": len(secondary),
        "has_secondary_outcomes": int(bool(secondary)),
        "endpoints_missing": int(not primary and not secondary),
        "n_inclusion": canonical["n_inclusion"],
        "n_exclusion": canonical["n_exclusion"],
        "eligibility_missing": int(not canonical["eligibility_present"]),
        "randomized": int("random" in allocation),
        "allocation_missing": int(not allocation),
        "blinded": int(bool(masking) and not any(x in masking for x in ("open", "none", "no masking"))),
        "masking_missing": int(not masking),
        "n_conditions": len(conditions),
        "conditions_missing": int(not conditions),
        "n_interventions": len(interventions),
        "interventions_missing": int(not interventions),
        **intervention_flags,
        "is_other_intervention": other_intervention,
        "intervention_type_missing": int(bool(interventions) and known_intervention_types == 0),
        "is_academic": int(is_academic),
        "is_industry": int(is_industry),
        "is_government": int(is_government),
        "sponsor_missing": int(sponsor_missing),
        "start_year": start_year or datetime.date.today().year,
        "start_date_missing": int(start_year is None),
        "is_interventional": int("interventional" in study_type),
        "is_observational": int("observational" in study_type),
        "study_type_missing": int(not study_type),
        "purpose_treatment": int("treatment" in purpose),
        "purpose_prevention": int("prevention" in purpose),
        "purpose_diagnostic": int("diagnostic" in purpose),
        "purpose_supportive_care": int("supportive" in purpose),
        "purpose_screening": int("screening" in purpose),
        "purpose_basic_science": int("basic science" in purpose),
        "purpose_other": int(bool(purpose) and not any(x in purpose for x in (
            "treatment", "prevention", "diagnostic", "supportive", "screening", "basic science",
        ))),
        "primary_purpose_missing": int(not purpose),
        "model_parallel": int("parallel" in model),
        "model_crossover": int("crossover" in model or "cross over" in model),
        "model_factorial": int("factorial" in model),
        "model_single_group": int("single group" in model),
        "model_sequential": int("sequential" in model),
        "model_other": int(bool(model) and not any(x in model for x in (
            "parallel", "crossover", "cross over", "factorial", "single group", "sequential",
        ))),
        "intervention_model_missing": int(not model),
        "female_only": int(female_only),
        "male_only": int(male_only),
        "sex_all": int(sex_all),
        "sex_missing": int(sex_missing),
        "minimum_age_years": min_age or 0.0,
        "maximum_age_years": max_age or 0.0,
        "age_range_years": age_range,
        "age_missing": int(age_missing),
        "healthy_volunteers": int(healthy is True),
        "healthy_volunteers_missing": int(healthy is None),
        "n_sites": canonical["sites"] or 0.0,
        "sites_missing": int(canonical["sites"] is None),
        "n_countries": canonical["countries"] or 0.0,
        "countries_missing": int(canonical["countries"] is None),
        "n_collaborators": canonical["collaborators"] or 0.0,
        "collaborators_missing": int(canonical["collaborators"] is None),
        "primary_outcome_timeframe_months": primary_timeframe or 0.0,
        "secondary_outcome_timeframe_months": secondary_timeframe or 0.0,
        "outcome_timeframes_missing": int(primary_timeframe is None and secondary_timeframe is None),
    }
    return {name: f[name] for name in STRUCT_FEATURES}


def trial_row_features(row: dict) -> dict:
    return _model_features(_canonical_row(row or {}))


def usdm_features(usdm: dict, burden: dict | None = None) -> dict:
    return _model_features(_canonical_usdm(usdm or {}))


def to_vector(struct_feats: dict, emb) -> list[float]:
    missing = [name for name in STRUCT_FEATURES if name not in struct_feats]
    if missing:
        raise ValueError(f"structured feature schema is missing: {', '.join(missing)}")
    emb_values = [float(value) for value in emb]
    return [float(struct_feats[name]) for name in STRUCT_FEATURES] + emb_values
