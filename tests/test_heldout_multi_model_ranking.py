from ui.data.model_ranking import (
    compute_global_score,
    rank_models_from_report,
)


def _mk_model(version: str, **micro_metrics):
    return {
        "model_version": version,
        "report": {"aggregate": {"micro": dict(micro_metrics)}},
    }


def test_rank_models_orders_by_weighted_score():
    report = {
        "models": [
            _mk_model(
                "v0.1.0",
                recall=0.86, precision=0.84, f1=0.85,
                agreement=0.88, fp_rate=0.08, fn_rate=0.14,
            ),
            _mk_model(
                "v0.2.0",
                recall=0.91, precision=0.89, f1=0.90,
                agreement=0.92, fp_rate=0.06, fn_rate=0.09,
            ),
            _mk_model(
                "v0.3.0",
                recall=0.82, precision=0.80, f1=0.81,
                agreement=0.86, fp_rate=0.11, fn_rate=0.18,
            ),
        ]
    }
    rows = rank_models_from_report(report)
    assert [r["model_version"] for r in rows] == ["v0.2.0", "v0.1.0", "v0.3.0"]
    assert rows[0]["score"] is not None
    assert 0.0 <= rows[0]["score"] <= 1.0


def test_global_score_renormalizes_when_some_metrics_missing():
    score = compute_global_score({
        "recall": 0.90,
        "precision": 0.85,
        # Missing f1/agreement/fp_rate/fn_rate on purpose.
    })
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_rank_models_puts_unscorable_models_last():
    report = {
        "models": [
            _mk_model("v0.1.0"),  # no metrics => no score
            _mk_model(
                "v0.2.0",
                recall=0.8, precision=0.8, f1=0.8,
                agreement=0.8, fp_rate=0.2, fn_rate=0.2,
            ),
        ]
    }
    rows = rank_models_from_report(report)
    assert rows[0]["model_version"] == "v0.2.0"
    assert rows[1]["model_version"] == "v0.1.0"
    assert rows[1]["score"] is None
