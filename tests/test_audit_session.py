"""The blind-then-reveal ordering is the audit's whole value. It must be enforced."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ui.audit.session import Mark, Phase, PhaseError, SpanSession


def _session() -> SpanSession:
    return SpanSession("s0", "rec_A", 100.0, 220.0)


def test_reveal_before_commit_raises() -> None:
    """The circularity guard. If candidates are visible while marks can still be
    edited, the gate measures the detector against labels the detector shaped."""
    s = _session()
    s.add_mark(110.0, 111.0)

    with pytest.raises(PhaseError, match="circular"):
        s.reveal(candidates=np.zeros((0, 2)), traces=[])


def test_marks_are_frozen_after_commit(tmp_path: Path) -> None:
    """Editing a mark after seeing the answer is the same contamination, later."""
    s = _session()
    s.add_mark(110.0, 111.0)
    s.commit(tmp_path)

    with pytest.raises(PhaseError, match="frozen"):
        s.add_mark(150.0, 151.0)
    with pytest.raises(PhaseError, match="frozen"):
        s.remove_mark(0)


def test_marks_are_on_disk_before_anything_is_revealed(tmp_path: Path) -> None:
    """commit() writes. A crash between commit and reveal still leaves a usable,
    uncontaminated record rather than losing the human's work."""
    s = _session()
    s.add_mark(110.0, 112.5)
    path = s.commit(tmp_path)

    assert path.exists()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["blind"] is True
    assert body["marks"] == [{"start_s": 110.0, "stop_s": 112.5}]
    assert body["committed_at"] is not None
    assert s.phase is Phase.COMMITTED


def test_candidates_and_traces_are_empty_until_revealed(tmp_path: Path) -> None:
    """Empty, not raising: the viewer asks every repaint, and a phase check at
    each call site would be one more place to get it wrong."""
    s = _session()
    assert s.candidates.shape == (0, 2)
    assert s.traces == []

    s.commit(tmp_path)
    assert s.candidates.shape == (0, 2), "still hidden between commit and reveal"

    s.reveal(candidates=np.array([[1.0, 2.0]]), traces=["t"])
    assert s.candidates.shape == (1, 2)
    assert s.traces == ["t"]


def test_committing_twice_raises(tmp_path: Path) -> None:
    """A second commit would overwrite the timestamp that evidences the ordering."""
    s = _session()
    s.commit(tmp_path)

    with pytest.raises(PhaseError, match="already committed"):
        s.commit(tmp_path)


def test_a_zero_length_mark_is_refused() -> None:
    """A click that did not drag is not an artifact, and would become a zero-width
    interval that no consumer can act on."""
    with pytest.raises(ValueError, match="positive duration"):
        Mark(110.0, 110.0)


def test_an_empty_span_commits_cleanly(tmp_path: Path) -> None:
    """'I found nothing here' is a real and important answer - it is what makes a
    false positive measurable. It must not be confused with 'not yet done'."""
    s = _session()
    path = s.commit(tmp_path)

    assert json.loads(path.read_text(encoding="utf-8"))["marks"] == []
    assert s.phase is Phase.COMMITTED


def test_marks_array_is_the_shape_the_viewer_overlays_want() -> None:
    s = _session()
    s.add_mark(110.0, 111.0)
    s.add_mark(150.0, 152.0)

    np.testing.assert_allclose(s.marks_array(), [[110.0, 111.0], [150.0, 152.0]])
    assert _session().marks_array().shape == (0, 2)
