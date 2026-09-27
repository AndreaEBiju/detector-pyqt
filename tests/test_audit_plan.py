"""The audit plan must be reproducible, stratified, and blind to the detector."""

from __future__ import annotations

import pytest

from ui.audit.plan import (
    EDGE_GUARD_S,
    MIN_ANIMALS,
    N_SPANS,
    SPAN_S,
    Assessable,
    plan_audit,
)


def _pool(n_animals: int = 3, per_animal: int = 2, dur: float = 600.0):
    return [
        Assessable(
            recording_id=f"rec_{chr(65 + a)}{i}",
            animal=chr(65 + a),
            condition="baseline" if i % 2 == 0 else "stim_recovery",
            regions=((0.0, dur),),
        )
        for a in range(n_animals)
        for i in range(per_animal)
    ]


def test_the_same_seed_gives_the_same_plan() -> None:
    """The seed is recorded in provenance so the plan can be rebuilt exactly."""
    a = plan_audit(_pool(), seed=7)
    b = plan_audit(_pool(), seed=7)

    assert a.spans == b.spans
    assert a.to_json()["seed"] == 7


def test_a_different_seed_gives_a_different_plan() -> None:
    """Otherwise the seed is decoration and every audit looks at the same minutes."""
    assert plan_audit(_pool(), seed=1).spans != plan_audit(_pool(), seed=2).spans


def test_the_plan_is_ten_minutes_in_five_contiguous_two_minute_spans() -> None:
    """The labelling budget: 1-3 hours once, at one keystroke per event."""
    plan = plan_audit(_pool(), seed=0)

    assert len(plan.spans) == N_SPANS == 5
    assert all(s.duration_s == pytest.approx(SPAN_S) for s in plan.spans)
    assert plan.total_s == pytest.approx(600.0)


def test_spans_are_stratified_across_animals() -> None:
    """A plan that lands on one animal measures that animal's noise floor."""
    plan = plan_audit(_pool(n_animals=3), seed=3)

    assert len({s.animal for s in plan.spans}) >= MIN_ANIMALS


def test_a_single_animal_pool_raises_rather_than_returning_a_weak_plan() -> None:
    """Silently returning a one-animal plan would hide that it proves less."""
    with pytest.raises(ValueError, match="at least"):
        plan_audit(_pool(n_animals=1), seed=0)


def test_spans_keep_clear_of_region_edges() -> None:
    """A span straddling an unassessable boundary shows filter settling, and the
    labeller would mark it as an artifact."""
    plan = plan_audit(_pool(dur=400.0), seed=11)

    for s in plan.spans:
        assert s.start_s >= EDGE_GUARD_S
        assert s.stop_s <= 400.0 - EDGE_GUARD_S


def test_a_region_too_short_for_a_span_is_never_drawn_from() -> None:
    """Placeability is span + BOTH guards, not span alone."""
    short = Assessable("s", "A", "baseline", ((0.0, SPAN_S + EDGE_GUARD_S),))
    assert short.placeable() == []

    long = Assessable("l", "A", "baseline", ((0.0, SPAN_S + 2 * EDGE_GUARD_S + 1),))
    assert long.placeable() != []


def test_stim_epochs_are_excluded_by_construction() -> None:
    """The caller passes assessable regions; a gap for stim can never be drawn.

    Recorded as a test because 'excluding stim' is a requirement, and the way it
    is met - by never offering those seconds to the sampler - is easy to lose in
    a later refactor.
    """
    pool = [
        Assessable("r1", "A", "stim_recovery", ((0.0, 300.0), (500.0, 900.0))),
        Assessable("r2", "B", "baseline", ((0.0, 900.0),)),
    ]
    plan = plan_audit(pool, seed=5, n_spans=4)

    for s in plan.spans:
        if s.recording_id == "r1":
            in_first = s.start_s >= EDGE_GUARD_S and s.stop_s <= 300.0 - EDGE_GUARD_S
            in_second = s.start_s >= 500.0 + EDGE_GUARD_S and s.stop_s <= 900.0 - EDGE_GUARD_S
            assert in_first or in_second, f"span {s} fell in the excluded gap"


def test_the_rule_is_recorded_and_says_it_is_not_candidate_dense() -> None:
    """The selection rule travels with the plan, so a reader of the audit record
    can see the audit did not choose its own spans using the detector."""
    doc = plan_audit(_pool(), seed=0).to_json()

    assert "NOT candidate-dense" in doc["selection_rule"]
    assert doc["edge_guard_s"] == EDGE_GUARD_S


# ---------------------------------------------------------------------------
# the precondition is checked, not documented
# ---------------------------------------------------------------------------


def test_a_region_overlapping_an_excluded_span_raises() -> None:
    """The failure this prevents is silent: a span inside a stim epoch shows the
    labeller stimulation artifact, they mark it, and recall looks fine while
    measuring something else entirely."""
    with pytest.raises(ValueError, match="overlaps excluded span"):
        Assessable(
            "r", "A", "stim_recovery",
            regions=((0.0, 300.0),),
            excluded=((120.0, 240.0),),   # stim epoch left inside the region
        )


def test_exclusions_that_were_genuinely_applied_are_accepted() -> None:
    """The regions sit either side of the excluded span, as a real split gives."""
    a = Assessable(
        "r", "A", "stim_recovery",
        regions=((0.0, 120.0), (240.0, 900.0)),
        excluded=((120.0, 240.0),),
    )

    assert a.placeable() != []


def test_unsorted_or_overlapping_regions_raise() -> None:
    """Overlapping regions would let the same seconds be drawn twice and counted
    as two independent observations."""
    with pytest.raises(ValueError, match="sorted and disjoint"):
        Assessable("r", "A", "baseline", regions=((0.0, 300.0), (200.0, 400.0)))


def test_an_empty_or_reversed_region_raises() -> None:
    with pytest.raises(ValueError, match="empty or reversed"):
        Assessable("r", "A", "baseline", regions=((300.0, 300.0),))
