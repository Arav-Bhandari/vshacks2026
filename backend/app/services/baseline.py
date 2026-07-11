"""Weighted baseline duration/enrollment estimate from similar trials."""
import statistics

_Z_90 = 1.645


def _weighted_stats(pairs: list[tuple[float, float]]) -> tuple[float | None, float | None, float | None]:
    """pairs = [(value, weight), ...]. Returns (mean, ci_low, ci_high)."""
    if not pairs:
        return None, None, None
    total_w = sum(w for _, w in pairs) or 1.0
    mean = sum(v * w for v, w in pairs) / total_w
    values = [v for v, _ in pairs]
    if len(pairs) < 3:
        return mean, min(values), max(values)
    variance = sum(w * (v - mean) ** 2 for v, w in pairs) / total_w
    std = variance ** 0.5
    return mean, mean - _Z_90 * std, mean + _Z_90 * std


def compute_baseline(similar_trials: list[dict], k: int = 10) -> dict:
    if not similar_trials:
        return {
            "expected_duration_months": None,
            "ci_low": None,
            "ci_high": None,
            "median_enrollment": None,
            "n_trials": 0,
            "trials_used": [],
        }

    ranked = sorted(
        similar_trials, key=lambda t: t.get("similarity", {}).get("total", 0), reverse=True
    )[:k]

    duration_pairs = [
        (t["duration_months"], t.get("similarity", {}).get("total", 0) or 1e-6)
        for t in ranked if t.get("duration_months") is not None
    ]
    enrollments = [t["enrollment"] for t in ranked if t.get("enrollment") is not None]

    mean_duration, ci_low, ci_high = _weighted_stats(duration_pairs)

    return {
        "expected_duration_months": round(mean_duration, 1) if mean_duration is not None else None,
        "ci_low": round(ci_low, 1) if ci_low is not None else None,
        "ci_high": round(ci_high, 1) if ci_high is not None else None,
        "median_enrollment": statistics.median(enrollments) if enrollments else None,
        "n_trials": len(ranked),
        "trials_used": [t["nct_id"] for t in ranked],
    }
