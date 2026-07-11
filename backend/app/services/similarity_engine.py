"""Find historical trials similar to a USDM protocol."""
import re

from app.database.db import search_trials

_WEIGHTS = {"condition": 0.35, "phase": 0.20, "endpoints": 0.25, "design": 0.20}

_model = None
_model_load_failed = False


def _get_model():
    global _model, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _model_load_failed = True
        _model = None
    return _model


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _cosine(a, b) -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _study_conditions_text(usdm: dict) -> str:
    return " ".join(usdm.get("study", {}).get("conditions", []) or [])


def _study_intervention_names(usdm: dict) -> list[str]:
    ivs = usdm.get("study", {}).get("interventions", []) or []
    return [iv.get("name", "") for iv in ivs if iv.get("name")]


def _study_endpoint_text(usdm: dict) -> str:
    texts = []
    for obj in usdm.get("study", {}).get("objectives", []) or []:
        for ep in obj.get("endpoints", []) or []:
            if isinstance(ep, dict):
                texts.append(ep.get("description") or ep.get("name") or "")
            else:
                texts.append(str(ep))
    return " ".join(t for t in texts if t)


_PHASE_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}


def _parse_phase_numbers(phase: str | None) -> list[int]:
    if not phase:
        return []
    nums = [int(n) for n in re.findall(r"\d+", phase)]
    if nums:
        return nums
    tokens = re.findall(r"[ivx]+", phase.lower())
    return [_PHASE_ROMAN[t] for t in tokens if t in _PHASE_ROMAN]


def _phase_score(usdm_phase: str | None, trial_phase: str | None) -> float:
    a, b = _parse_phase_numbers(usdm_phase), _parse_phase_numbers(trial_phase)
    if not a or not b:
        return 0.0
    if set(a) & set(b):
        return 1.0
    if min(abs(x - y) for x in a for y in b) == 1:
        return 0.5
    return 0.0


def _design_score(usdm: dict, trial: dict) -> float:
    design = usdm.get("study", {}).get("design", {}) or {}
    matches = 0
    total = 4

    u_alloc = (design.get("allocation") or "").lower()
    t_alloc = (trial.get("allocation") or "").lower()
    u_rand = "random" in u_alloc or bool(design.get("randomization"))
    t_rand = "random" in t_alloc
    if u_rand == t_rand:
        matches += 1

    u_mask = (design.get("masking") or "").lower()
    t_mask = (trial.get("masking") or "").lower()
    u_none = u_mask in ("", "none", "open", "open label", "open-label")
    t_none = t_mask in ("", "none", "open", "open label", "open-label")
    if u_none == t_none:
        matches += 1

    u_model = (design.get("interventionModel") or "").strip().lower()
    t_model = (trial.get("intervention_model") or "").strip().lower()
    if u_model and t_model and u_model == t_model:
        matches += 1

    u_arms = len(usdm.get("study", {}).get("arms", []) or [])
    t_arms = trial.get("arms")
    if u_arms and t_arms is not None and abs(u_arms - t_arms) <= 1:
        matches += 1

    return matches / total


def _candidates(usdm: dict, limit: int = 400) -> list[dict]:
    queries = []
    cond_text = _study_conditions_text(usdm)
    if cond_text:
        queries.append(cond_text)
    for name in _study_intervention_names(usdm):
        queries.append(name)

    seen: dict[str, dict] = {}
    per_query = max(50, limit // max(1, len(queries) or 1))
    for q in queries or [""]:
        if not q:
            continue
        for trial in search_trials(q, per_query):
            seen.setdefault(trial["nct_id"], trial)
        if len(seen) >= limit:
            break
    return list(seen.values())[:limit]


def find_similar(usdm: dict, limit: int = 50) -> list[dict]:
    candidates = _candidates(usdm)
    if not candidates:
        return []

    usdm_cond_text = _study_conditions_text(usdm)
    usdm_endpoint_text = _study_endpoint_text(usdm)
    usdm_phase = usdm.get("study", {}).get("phase")

    model = _get_model()
    cond_embs = endpoint_embs = None
    if model is not None:
        cand_cond_texts = [" ".join(c.get("conditions") or []) if isinstance(c.get("conditions"), list)
                            else (c.get("conditions") or "") for c in candidates]
        cand_endpoint_texts = [c.get("primary_outcomes") or "" for c in candidates]
        batch = [usdm_cond_text, usdm_endpoint_text] + cand_cond_texts + cand_endpoint_texts
        embs = model.encode(batch)
        usdm_cond_emb, usdm_endpoint_emb = embs[0], embs[1]
        n = len(candidates)
        cond_embs = embs[2:2 + n]
        endpoint_embs = embs[2 + n:2 + 2 * n]

    scored = []
    for i, trial in enumerate(candidates):
        trial_cond_text = " ".join(trial.get("conditions") or []) if isinstance(trial.get("conditions"), list) \
            else (trial.get("conditions") or "")
        trial_endpoint_text = trial.get("primary_outcomes") or ""

        if model is not None:
            cond_sim = max(0.0, _cosine(usdm_cond_emb, cond_embs[i]))
            endpoint_sem = max(0.0, _cosine(usdm_endpoint_emb, endpoint_embs[i]))
        else:
            cond_sim = _jaccard(usdm_cond_text, trial_cond_text)
            endpoint_sem = _jaccard(usdm_endpoint_text, trial_endpoint_text)

        endpoint_jac = _jaccard(usdm_endpoint_text, trial_endpoint_text)
        endpoint_score = 0.6 * endpoint_sem + 0.4 * endpoint_jac

        phase_score = _phase_score(usdm_phase, trial.get("phase"))
        design_score = _design_score(usdm, trial)

        total = (
            _WEIGHTS["condition"] * cond_sim
            + _WEIGHTS["phase"] * phase_score
            + _WEIGHTS["endpoints"] * endpoint_score
            + _WEIGHTS["design"] * design_score
        )
        scored.append({
            **trial,
            "similarity": {
                "total": round(total, 3),
                "condition": round(cond_sim, 3),
                "phase": round(phase_score, 3),
                "endpoints": round(endpoint_score, 3),
                "design": round(design_score, 3),
            },
        })

    scored.sort(key=lambda t: t["similarity"]["total"], reverse=True)
    return scored[:limit]
