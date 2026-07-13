"""Auxiliary duration-model components."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors


def _finite_array(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("model inputs contain NaN or infinite values")
    return result


@dataclass
class RidgeTextHead:
    """Linear log-duration model for text embeddings."""

    alpha: float
    coef_: np.ndarray
    intercept_: float

    @classmethod
    def fit(cls, embeddings: np.ndarray, y_log: np.ndarray, alpha: float) -> "RidgeTextHead":
        X = _finite_array(embeddings)
        y = np.asarray(y_log, dtype=np.float64)
        model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4, max_iter=3000)
        model.fit(X, y)
        return cls(
            alpha=float(alpha),
            coef_=np.asarray(model.coef_, dtype=np.float32),
            intercept_=float(model.intercept_),
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        X = _finite_array(embeddings)
        if X.shape[1] != len(self.coef_):
            raise ValueError(
                f"Ridge head expects {len(self.coef_)} embedding dimensions, got {X.shape[1]}"
            )
        return np.asarray(X @ self.coef_ + self.intercept_, dtype=np.float64)

    def save(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(temp, "wb") as file:
            np.savez_compressed(
                file,
                alpha=np.asarray(self.alpha, dtype=np.float64),
                coef=self.coef_,
                intercept=np.asarray(self.intercept_, dtype=np.float64),
                schema_version=np.asarray(1, dtype=np.int64),
            )
        os.replace(temp, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "artifact": path.name,
            "artifact_sha256": digest,
            "artifact_schema_version": 1,
            "alpha": self.alpha,
            "embedding_dim": int(len(self.coef_)),
            "target": "log1p_duration_months",
        }

    @classmethod
    def load(cls, path: Path) -> "RidgeTextHead":
        with np.load(path, allow_pickle=False) as artifact:
            version = int(artifact["schema_version"])
            if version != 1:
                raise ValueError(f"Unsupported Ridge artifact schema version {version}")
            return cls(
                alpha=float(artifact["alpha"]),
                coef_=np.asarray(artifact["coef"], dtype=np.float32),
                intercept_=float(artifact["intercept"]),
            )


def fit_nonnegative_blend(
    components: Iterable[np.ndarray],
    target: np.ndarray,
    component_names: Iterable[str],
) -> dict[str, Any]:
    """Fit a nonnegative linear blend."""

    arrays = [np.asarray(component, dtype=np.float64) for component in components]
    names = list(component_names)
    if len(arrays) != len(names) or not arrays:
        raise ValueError("components and component_names must have the same nonzero length")
    X = np.column_stack(arrays)
    y = np.asarray(target, dtype=np.float64)
    if len(y) != len(X) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("blend inputs must be aligned and finite")
    model = LinearRegression(positive=True, fit_intercept=True)
    model.fit(X, y)
    weights = {name: float(weight) for name, weight in zip(names, model.coef_)}
    return {
        "weights": weights,
        "intercept": float(model.intercept_),
        "nonnegative_weights": True,
    }


def apply_blend(blend: dict[str, Any], **components: np.ndarray) -> np.ndarray:
    weights = blend.get("weights") or {}
    if set(weights) - set(components):
        raise ValueError(f"Missing blend components: {sorted(set(weights) - set(components))}")
    first = np.asarray(next(iter(components.values())), dtype=np.float64)
    result = np.full(first.shape, float(blend.get("intercept", 0.0)), dtype=np.float64)
    for name, weight in weights.items():
        value = np.asarray(components[name], dtype=np.float64)
        if value.shape != result.shape:
            raise ValueError("blend components must have identical shapes")
        result += float(weight) * value
    return result


class NeighborDurationBaseline:
    """Cosine-neighbor duration baseline."""

    def __init__(
        self,
        n_neighbors: int = 25,
        max_reference: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.max_reference = int(max_reference)
        self.seed = int(seed)
        self.backend = "unfitted"
        self.reference_size = 0
        self._index: Any = None
        self._durations: np.ndarray | None = None

    @staticmethod
    def _normalize(value: np.ndarray) -> np.ndarray:
        X = _finite_array(value).copy()
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        X /= norm
        return X

    def fit(self, embeddings: np.ndarray, durations: np.ndarray) -> "NeighborDurationBaseline":
        X = self._normalize(embeddings)
        y = np.asarray(durations, dtype=np.float32)
        if len(X) != len(y) or not np.isfinite(y).all():
            raise ValueError("neighbor reference embeddings/durations must be aligned and finite")

        rng = np.random.default_rng(self.seed)
        try:
            import faiss  # type: ignore

            limit = min(len(X), self.max_reference)
            self.backend = "faiss_ivf_cosine"
        except ImportError:
            faiss = None
            fallback_limit = int(os.getenv("TRAIN_NEIGHBOR_SKLEARN_MAX_REFERENCE", "5000"))
            limit = min(len(X), self.max_reference, fallback_limit)
            self.backend = "sklearn_bounded_cosine"

        selected = (
            np.arange(len(X), dtype=int)
            if len(X) <= limit
            else np.sort(rng.choice(len(X), size=limit, replace=False))
        )
        X = np.ascontiguousarray(X[selected], dtype=np.float32)
        self._durations = y[selected]
        self.reference_size = int(len(X))
        k = max(1, min(self.n_neighbors, self.reference_size))

        if faiss is not None and self.reference_size >= 256:
            nlist = min(256, max(16, int(np.sqrt(self.reference_size))))
            quantizer = faiss.IndexFlatIP(X.shape[1])
            index = faiss.IndexIVFFlat(quantizer, X.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(X)
            index.add(X)
            index.nprobe = min(16, nlist)
            self._index = index
            self._k = k
        else:
            self.backend = "sklearn_bounded_cosine"
            index = NearestNeighbors(
                n_neighbors=k,
                metric="cosine",
                algorithm="brute",
                n_jobs=int(os.getenv("TRAIN_N_JOBS", "-1")),
            )
            index.fit(X)
            self._index = index
            self._k = k
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._index is None or self._durations is None:
            raise RuntimeError("neighbor baseline has not been fitted")
        X = np.ascontiguousarray(self._normalize(embeddings), dtype=np.float32)
        batch_size = int(os.getenv("TRAIN_NEIGHBOR_QUERY_BATCH", "4096"))
        predictions = []
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            if self.backend == "faiss_ivf_cosine":
                similarity, indices = self._index.search(batch, self._k)
            else:
                distance, indices = self._index.kneighbors(batch, return_distance=True)
                similarity = 1.0 - distance
            neighbor_y = self._durations[indices]
            weights = np.maximum(np.asarray(similarity, dtype=np.float64), 0.0) + 1e-6
            predictions.append(np.sum(neighbor_y * weights, axis=1) / np.sum(weights, axis=1))
        return np.concatenate(predictions).astype(np.float64)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reference_size": self.reference_size,
            "n_neighbors": int(getattr(self, "_k", self.n_neighbors)),
            "max_reference": self.max_reference,
            "metric": "cosine",
        }


def coverage_preflight(
    X_struct: np.ndarray,
    feature_names: list[str],
    n_rows: int,
) -> dict[str, Any]:
    """Summarize structured feature coverage."""

    X = np.asarray(X_struct, dtype=np.float64)
    if X.shape != (n_rows, len(feature_names)):
        raise ValueError("coverage matrix does not match row/feature counts")
    finite = np.isfinite(X)
    nonfinite = {
        name: int((~finite[:, index]).sum())
        for index, name in enumerate(feature_names)
        if (~finite[:, index]).any()
    }
    safe = np.where(finite, X, np.nan)
    std = np.nanstd(safe, axis=0)
    constant = [name for name, value in zip(feature_names, std) if value < 1e-12]
    near_constant = [
        name
        for index, name in enumerate(feature_names)
        if name not in constant
        and max(
            np.mean(X[:, index] == np.nanmin(X[:, index])),
            np.mean(X[:, index] == np.nanmax(X[:, index])),
        )
        >= 0.995
    ]
    missing_flag_rates = {
        name: round(float(np.mean(X[:, index] > 0.5)), 6)
        for index, name in enumerate(feature_names)
        if name.endswith("_missing")
    }
    warnings = []
    if nonfinite:
        warnings.append(f"Nonfinite structured values found in {len(nonfinite)} features")
    if constant:
        warnings.append(f"Constant structured features: {', '.join(constant)}")
    almost_unavailable = [
        name for name, rate in missing_flag_rates.items() if rate >= 0.95
    ]
    if almost_unavailable:
        warnings.append(
            "At least 95% missing in training: " + ", ".join(almost_unavailable)
        )
    return {
        "n_rows": int(n_rows),
        "nonfinite_counts": nonfinite,
        "constant_features": constant,
        "near_constant_features": near_constant,
        "missing_flag_rates": missing_flag_rates,
        "warnings": warnings,
        "passed": not nonfinite,
    }


def metadata_sha256(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
