"""Which minutes the recall audit asks a human to look at, and why those.

**Span selection must not use the thing under test.** Choosing candidate-dense
regions would re-introduce, at the span level, exactly the anchoring that blind
marking exists to prevent: the audit would measure recall over the regions the
detector already likes. The selection is therefore uniform-random within the
assessable part of each recording, and the seed is recorded so the same plan can be
rebuilt. Two strata, and only these (ruling 2026-09-29): **animals** by rotation
within a plan, and **conditions** across the gate's eligible pool - the window asks
``gems_blanking_v2.detect.recall.condition_plan`` which condition each span must
have, from the composition of earlier eligible plans alone, never their scores.

Budget: five contiguous 2-minute spans, ~10 minutes of signal per plan. The task's
labelling budget is 500-1000 judged events in 1-3 hours across ~12 recordings, and
a human cannot sustain one-keystroke judgement over much more than that.

Contiguous rather than scattered seconds: an artifact is recognised by its context,
and a 2-minute run lets the labeller see onset, body and recovery. Scattered
one-second windows would cost the same wall-clock and produce worse labels.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Final

SPAN_S: Final = 120.0
"""One contiguous span, seconds."""

N_SPANS: Final = 5
"""Spans per plan, so ~10 minutes total."""

EDGE_GUARD_S: Final = 20.0
"""Kept clear of both ends of an assessable region.

A span straddling the boundary of an unassessable region would show the labeller
filter settling rather than signal, and they would mark it. The guard is larger
than the longest consumer edge buffer in play (15 s, slow wave).
"""

MIN_ANIMALS: Final = 2
MAX_ANIMALS: Final = 3
"""Stratification width. Fewer than two and the plan measures one animal's noise."""


@dataclass(frozen=True, slots=True)
class Span:
    """One contiguous stretch a human will mark, blind."""

    recording_id: str
    animal: str
    condition: str
    start_s: float
    stop_s: float

    @property
    def duration_s(self) -> float:
        return self.stop_s - self.start_s


@dataclass(frozen=True, slots=True)
class AuditPlan:
    """A reproducible set of spans, plus everything needed to rebuild it."""

    spans: tuple[Span, ...]
    seed: int
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return sum(s.duration_s for s in self.spans)

    def to_json(self) -> dict[str, object]:
        """Serialise for the audit record. The seed is the point of this."""
        return {
            "seed": self.seed,
            "n_spans": len(self.spans),
            "span_s": SPAN_S,
            "total_s": self.total_s,
            "selection_rule": (
                "uniform-random start within each assessable region; animals rotated "
                "within the plan; conditions "
                + ("as requested per span, balancing the gate's eligible pool from "
                   "earlier eligible plans' composition (ruling 2026-09-29)"
                   if "conditions_requested" in self.provenance else
                   "not stratified (drawn by recording, the pre-2026-09-29 rule)")
                + "; NOT candidate-dense, because selecting spans with the detector "
                "under test would measure recall over the regions it already likes"
            ),
            "edge_guard_s": EDGE_GUARD_S,
            "spans": [asdict(s) for s in self.spans],
            **self.provenance,
        }


@dataclass(frozen=True, slots=True)
class Assessable:
    """One recording's usable stretches, with stim and unassessable edges removed.

    ``regions`` are ``[start, stop)`` seconds a span may be drawn from. ``excluded``
    is what was taken out - stim epochs from task 03B's split, unassessable edges
    from the band settling times. Neither is this module's to compute.

    **``excluded`` is checked, not trusted.** "The caller passes assessable regions
    only" is a comment, and the gate's validity rests on it being true: a span that
    overlaps a stim epoch shows the labeller stimulation artifact, they mark it,
    and recall comes out looking fine while measuring the wrong thing. The caller
    therefore declares what it removed and this class asserts the removal actually
    happened.
    """

    recording_id: str
    animal: str
    condition: str
    regions: tuple[tuple[float, float], ...]
    excluded: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate the regions and prove the exclusions were applied."""
        last = -float("inf")
        for lo, hi in self.regions:
            if not hi > lo:
                msg = (
                    f"{self.recording_id}: region [{lo}, {hi}) is empty or "
                    "reversed; a span cannot be drawn from it"
                )
                raise ValueError(msg)
            if lo < last:
                msg = (
                    f"{self.recording_id}: regions must be sorted and disjoint, "
                    f"got [{lo}, {hi}) after {last}. Overlapping regions would let "
                    "the same seconds be drawn twice and counted as independent."
                )
                raise ValueError(msg)
            last = hi

        for elo, ehi in self.excluded:
            for lo, hi in self.regions:
                if lo < ehi and elo < hi:
                    msg = (
                        f"{self.recording_id}: assessable region [{lo}, {hi}) "
                        f"overlaps excluded span [{elo}, {ehi}). Stim epochs and "
                        "unassessable edges must be removed BEFORE planning: a "
                        "span drawn there shows the labeller stimulation artifact "
                        "or filter settling, they mark it as motion, and the "
                        "recall number looks fine while measuring the wrong thing."
                    )
                    raise ValueError(msg)

    def placeable(self, span_s: float = SPAN_S) -> list[tuple[float, float]]:
        """Sub-ranges in which a span START may legally fall."""
        out: list[tuple[float, float]] = []
        for lo, hi in self.regions:
            a = lo + EDGE_GUARD_S
            b = hi - EDGE_GUARD_S - span_s
            if b > a:
                out.append((a, b))
        return out


def plan_audit(
    pool: list[Assessable],
    seed: int,
    *,
    n_spans: int = N_SPANS,
    span_s: float = SPAN_S,
    conditions: Sequence[str] | None = None,
) -> AuditPlan:
    """Draw ``n_spans`` contiguous spans: animals rotated, conditions as asked.

    Animals are rotated round-robin (at most :data:`MAX_ANIMALS`, shuffled), and
    within an animal its recordings are taken in shuffled order, so a plan cannot
    land all five spans on one recording. Within the chosen recording the start is
    uniform over the placeable range; all randomness is seeded.

    ``conditions`` - one per span - is how conditions are stratified: span ``i`` is
    drawn from a recording of ``conditions[i]``, from the first animal in the
    rotation that has one. The window passes
    ``recall.condition_plan``'s answer, which balances conditions across the gate's
    eligible pool. Without it the conditions are whatever the recordings drawn happen
    to be - the pre-2026-09-29 behaviour, kept byte-for-byte so a recorded seed
    still rebuilds its plan.

    Raises
    ------
    ValueError
        If the pool (restricted to the asked conditions) spans fewer than
        :data:`MIN_ANIMALS` animals, if ``conditions`` has the wrong length, or if no
        recording has room for a span. Each means the plan would not measure what it
        claims to, and quietly returning a shorter plan would hide that.
    """
    if conditions is not None:
        if len(conditions) != n_spans:
            msg = f"{len(conditions)} conditions given for {n_spans} spans"
            raise ValueError(msg)
        return _plan_by_condition(pool, seed, n_spans=n_spans, span_s=span_s,
                                  conditions=tuple(conditions))
    usable = [a for a in pool if a.placeable(span_s)]
    if not usable:
        msg = (
            f"no recording in the pool has an assessable region longer than "
            f"{span_s + 2 * EDGE_GUARD_S:.0f} s (span plus both edge guards)"
        )
        raise ValueError(msg)

    animals = sorted({a.animal for a in usable})
    if len(animals) < MIN_ANIMALS:
        msg = (
            f"pool covers {len(animals)} animal(s) ({animals}); at least "
            f"{MIN_ANIMALS} are needed or the plan measures one animal's noise"
        )
        raise ValueError(msg)

    rng = random.Random(seed)
    # Round-robin over animals, so the first MIN_ANIMALS..MAX_ANIMALS spans are
    # guaranteed to come from different animals before any animal repeats.
    order = animals[:MAX_ANIMALS] if len(animals) > MAX_ANIMALS else animals
    rng.shuffle(order)

    by_animal: dict[str, list[Assessable]] = {}
    for a in usable:
        if a.animal in order:
            by_animal.setdefault(a.animal, []).append(a)
    for recs in by_animal.values():
        rng.shuffle(recs)

    spans: list[Span] = []
    used: set[tuple[str, float]] = set()
    cursor = {k: 0 for k in by_animal}
    guard = 0
    while len(spans) < n_spans and guard < n_spans * 50:
        guard += 1
        animal = order[len(spans) % len(order)]
        recs = by_animal[animal]
        rec = recs[cursor[animal] % len(recs)]
        cursor[animal] += 1

        ranges = rec.placeable(span_s)
        lo, hi = ranges[rng.randrange(len(ranges))]
        start = rng.uniform(lo, hi)
        # Do not draw the same recording twice at an overlapping position.
        if any(
            r == rec.recording_id and abs(s - start) < span_s for r, s in used
        ):
            continue
        used.add((rec.recording_id, start))
        spans.append(
            Span(
                recording_id=rec.recording_id,
                animal=rec.animal,
                condition=rec.condition,
                start_s=start,
                stop_s=start + span_s,
            )
        )

    if len(spans) < n_spans:
        msg = (
            f"could only place {len(spans)} of {n_spans} non-overlapping spans "
            f"across {len(usable)} recordings"
        )
        raise ValueError(msg)

    return AuditPlan(
        spans=tuple(spans),
        seed=seed,
        provenance={
            "animals": sorted({s.animal for s in spans}),
            "conditions": sorted({s.condition for s in spans}),
            "recordings": sorted({s.recording_id for s in spans}),
        },
    )


def _plan_by_condition(
    pool: list[Assessable], seed: int, *, n_spans: int, span_s: float,
    conditions: tuple[str, ...],
) -> AuditPlan:
    """:func:`plan_audit` with span ``i`` drawn from a ``conditions[i]`` recording."""
    wanted = set(conditions)
    usable = [a for a in pool if a.placeable(span_s) and a.condition in wanted]
    animals = sorted({a.animal for a in usable})
    if len(animals) < MIN_ANIMALS:
        msg = (f"the {sorted(wanted)} recordings cover {len(animals)} animal(s) "
               f"({animals}); at least {MIN_ANIMALS} are needed")
        raise ValueError(msg)
    rng = random.Random(seed)
    order = animals[:MAX_ANIMALS] if len(animals) > MAX_ANIMALS else animals
    rng.shuffle(order)
    by_key: dict[tuple[str, str], list[Assessable]] = {}
    for a in sorted(usable, key=lambda a: (a.animal, a.condition, a.recording_id)):
        if a.animal in order:
            by_key.setdefault((a.animal, a.condition), []).append(a)
    for recs in by_key.values():
        rng.shuffle(recs)
    spans: list[Span] = []
    used: set[tuple[str, float]] = set()
    cursor = dict.fromkeys(by_key, 0)
    guard = 0
    while len(spans) < n_spans and guard < n_spans * 50:
        guard += 1
        cond = conditions[len(spans)]
        first = len(spans) % len(order)
        animal = next((order[(first + k) % len(order)] for k in range(len(order))
                       if (order[(first + k) % len(order)], cond) in by_key), None)
        if animal is None:
            msg = f"no animal in the rotation {order} has a {cond} recording with room"
            raise ValueError(msg)
        recs = by_key[(animal, cond)]
        rec = recs[cursor[(animal, cond)] % len(recs)]
        cursor[(animal, cond)] += 1
        ranges = rec.placeable(span_s)
        lo, hi = ranges[rng.randrange(len(ranges))]
        start = rng.uniform(lo, hi)
        if any(r == rec.recording_id and abs(s - start) < span_s for r, s in used):
            continue
        used.add((rec.recording_id, start))
        spans.append(Span(recording_id=rec.recording_id, animal=rec.animal,
                          condition=rec.condition, start_s=start, stop_s=start + span_s))
    if len(spans) < n_spans:
        msg = (f"could only place {len(spans)} of {n_spans} non-overlapping spans "
               f"of conditions {list(conditions)} across {len(usable)} recordings")
        raise ValueError(msg)
    return AuditPlan(
        spans=tuple(spans), seed=seed,
        provenance={"animals": sorted({s.animal for s in spans}),
                    "conditions": sorted({s.condition for s in spans}),
                    "conditions_requested": list(conditions),
                    "recordings": sorted({s.recording_id for s in spans})},
    )


def replace_spans(
    spans: list[Span], indices: list[int], pool: list[Assessable], seed: int,
    *, span_s: float = SPAN_S,
) -> list[Span]:
    """Replace ``spans[i]`` for each ``i`` in ``indices``; the others are kept as-is.

    For a plan whose recordings became ineligible after it was drawn (an animal
    excluded mid-round). Each replacement keeps the replaced span's CONDITION, comes
    from a recording not already in the plan - chosen uniformly among the
    recordings with room for a span - and starts uniformly within its placeable
    range, exactly as :func:`plan_audit` places spans. Seeded; the caller records
    the seed.

    Raises
    ------
    ValueError
        If no eligible recording of the needed condition has room for a span.
    """
    rng = random.Random(seed)
    in_plan = {sp.recording_id for sp in spans}
    out = list(spans)
    for i in sorted(indices):
        condition = spans[i].condition
        candidates = sorted(
            (a for a in pool
             if a.condition == condition and a.recording_id not in in_plan
             and a.placeable(span_s)),
            key=lambda a: a.recording_id,
        )
        if not candidates:
            msg = f"no eligible {condition} recording has room to replace span {i + 1}"
            raise ValueError(msg)
        rec = candidates[rng.randrange(len(candidates))]
        ranges = rec.placeable(span_s)
        lo, hi = ranges[rng.randrange(len(ranges))]
        start = rng.uniform(lo, hi)
        out[i] = Span(recording_id=rec.recording_id, animal=rec.animal,
                      condition=rec.condition, start_s=start, stop_s=start + span_s)
        in_plan.add(rec.recording_id)
    return out
