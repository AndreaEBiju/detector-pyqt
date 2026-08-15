"""Utilities for ranking held-out multi-model results."""

from __future__ import annotations

from typing import Optional


# Weighted business score used to rank models after held-out comparison.
# Positive metrics are better as-is; error-rate metrics are inverted
# (1 - rate) before aggregation so larger is always better.
GLOBAL_SCORE_WEIGHTS = {
    "recall": 0.30,
    "precision": 0.20,
    "f1": 0.20,
    "agreement": 0.15,
    "fp_rate": 0.075,
    "fn_rate": 0.075,
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def compute_global_score(
    metrics: dict, weights: Optional[dict] = None,
) -> Optional[float]:
    """Compute a normalized weighted score in [0, 1].

    Missing metrics are skipped and remaining weights are renormalized.
    Returns None only if no weighted metric is available.
    """
    ws = dict(weights or GLOBAL_SCORE_WEIGHTS)
    total_weight = 0.0
    score_sum = 0.0
    for key, weight in ws.items():
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            v = _clamp01(float(raw))
        except Exception:
            continue
        contrib = v if key not in {"fp_rate", "fn_rate"} else (1.0 - v)
        score_sum += float(weight) * contrib
        total_weight += float(weight)
    if total_weight <= 0:
        return None
    return _clamp01(score_sum / total_weight)


def rank_models_from_report(
    report: dict, weights: Optional[dict] = None,
) -> list[dict]:
    """Build score/ranking rows from a multi-model held-out report."""
    rows: list[dict] = []
    for model in (report.get("models", []) or []):
        version = model.get("model_version", "?")
        micro = (
            ((model.get("report") or {}).get("aggregate") or {}).get("micro")
            or {}
        )
        score = compute_global_score(micro, weights=weights)
        rows.append({
            "model_version": str(version),
            "score": score,
            "metrics": {
                "agreement": micro.get("agreement"),
                "precision": micro.get("precision"),
                "recall": micro.get("recall"),
                "f1": micro.get("f1"),
                "fp_rate": micro.get("fp_rate"),
                "fn_rate": micro.get("fn_rate"),
            },
        })

    rows.sort(
        key=lambda r: (
            r["score"] is None,
            -float(r["score"]) if r["score"] is not None else 0.0,
            r["model_version"],
        )
    )
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows
