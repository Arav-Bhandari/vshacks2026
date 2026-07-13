"""Score protocol complexity and patient burden."""
import re

_COMMON_CONDITIONS = {
    "cancer", "diabetes", "hypertension", "asthma", "depression", "obesity",
    "arthritis", "copd", "heart disease", "stroke", "alzheimer", "migraine",
    "allergy", "anxiety", "back pain", "osteoporosis", "influenza",
}

_INVASIVE_KEYWORDS = [
    "biopsy", "infusion", "lumbar", "mri", "blood draw", "catheter",
    "endoscopy", "injection", "surgical", "spinal tap", "venipuncture",
]


def _clamp(x: float) -> int:
    return int(round(max(0.0, min(100.0, x))))


def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def analyze_burden(usdm: dict) -> dict:
    study = usdm.get("study", {}) or {}

    arms = study.get("arms", []) or []
    n_arms = len(arms)

    endpoints = []
    for obj in study.get("objectives", []) or []:
        endpoints.extend(obj.get("endpoints", []) or [])
    n_endpoints = len(endpoints)

    visits = _get(study, "scheduleOfActivities", "schedule_of_activities", default=[]) or []
    n_visits = len(visits)
    procedures_total = sum(len(v.get("procedures", []) or []) for v in visits if isinstance(v, dict))
    avg_procedures = procedures_total / n_visits if n_visits else 0.0

    design = study.get("design", {}) or {}
    model_str = (design.get("interventionModel") or "").lower()
    is_crossover_factorial = "crossover" in model_str or "factorial" in model_str

    complexity_factors = [
        ("arms", n_arms * 8, f"{n_arms} treatment arm(s)"),
        ("endpoints", n_endpoints * 5, f"{n_endpoints} endpoint(s)"),
        ("visits", n_visits * 3, f"{n_visits} scheduled visit(s)"),
        ("procedures_per_visit", avg_procedures * 4,
         f"avg {avg_procedures:.1f} procedures per visit"),
        ("crossover_factorial", 15 if is_crossover_factorial else 0,
         "crossover/factorial design" if is_crossover_factorial else "parallel design"),
    ]
    complexity_score = _clamp(sum(v for _, v, _ in complexity_factors))

    eligibility = study.get("eligibility", {}) or {}
    inclusion = eligibility.get("inclusionCriteria") or eligibility.get("inclusion_criteria") or []
    exclusion = eligibility.get("exclusionCriteria") or eligibility.get("exclusion_criteria") or []
    n_criteria = len(inclusion) + len(exclusion)

    enrollment = _get(study, "plannedEnrollment", "planned_enrollment", default=None)
    if enrollment:
        enrollment_score = 25 if enrollment < 50 else 15 if enrollment < 200 else 5
    else:
        enrollment_score = 10

    min_age = eligibility.get("minAge") or eligibility.get("min_age")
    max_age = eligibility.get("maxAge") or eligibility.get("max_age")
    age_score = 0
    if isinstance(min_age, (int, float)) and isinstance(max_age, (int, float)):
        age_score = 15 if (max_age - min_age) < 20 else 0

    conditions = [c.lower() for c in (study.get("conditions") or [])]
    is_rare = bool(conditions) and not any(
        common in c for c in conditions for common in _COMMON_CONDITIONS
    )
    rare_score = 20 if is_rare else 0

    recruitment_factors = [
        ("eligibility_criteria", n_criteria * 3, f"{n_criteria} eligibility criteria"),
        ("enrollment_size", enrollment_score,
         f"planned enrollment {enrollment}" if enrollment else "enrollment unspecified"),
        ("age_restriction", age_score, "narrow eligible age range" if age_score else "broad age range"),
        ("rare_condition", rare_score,
         "condition outside common-disease list" if is_rare else "common condition"),
    ]
    recruitment_difficulty = _clamp(sum(v for _, v, _ in recruitment_factors))

    duration_months = _get(study, "durationMonths", "duration_months", default=None) or 0

    all_text = " ".join(
        str(v.get("name", "")) + " " + " ".join(str(p) for p in (v.get("procedures") or []))
        for v in visits if isinstance(v, dict)
    ).lower()
    invasive_hits = sum(1 for kw in _INVASIVE_KEYWORDS if kw in all_text)

    burden_factors = [
        ("visit_count", n_visits * 4, f"{n_visits} scheduled visit(s)"),
        ("invasive_procedures", invasive_hits * 8,
         f"{invasive_hits} invasive procedure type(s)"),
        ("duration", min(duration_months, 24) * 2, f"study duration {duration_months} month(s)"),
    ]
    patient_burden = _clamp(sum(v for _, v, _ in burden_factors))

    all_factors = complexity_factors + recruitment_factors + burden_factors
    top = sorted(all_factors, key=lambda f: f[1], reverse=True)[:5]
    factors = [{"name": name, "score": round(score, 1), "detail": detail}
               for name, score, detail in top if score > 0]

    return {
        "complexity_score": complexity_score,
        "recruitment_difficulty": recruitment_difficulty,
        "patient_burden": patient_burden,
        "factors": factors,
    }
