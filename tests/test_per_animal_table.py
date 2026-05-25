"""Unit tests for ui.widgets.per_animal_table.

Covers the pure-Python helpers (`normalize_animal_letter`,
`compute_summary`) plus the widget's data-binding contract: populating
from a recordings list, editing the Animal column, summary updating
live, and the `animal_edited` signal firing.

The Qt-dependent tests skip themselves when no Qt platform is
available -- the parity-with-region_table style.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
_DETECTOR_CORE = ROOT / "detector-core"
if _DETECTOR_CORE.exists() and str(_DETECTOR_CORE) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_CORE))

from ui.widgets.per_animal_table import (
    compute_summary, normalize_animal_letter,
)


# ----------------------------------------------------------------------
# normalize_animal_letter -- pure
# ----------------------------------------------------------------------

def test_normalize_lowercase_single_char():
    assert normalize_animal_letter("j") == "J"


def test_normalize_first_char_of_long_string():
    """'JEL', 'Jel', 'jel' all collapse to 'J'."""
    assert normalize_animal_letter("JEL") == "J"
    assert normalize_animal_letter("Jel") == "J"
    assert normalize_animal_letter("jel") == "J"


def test_normalize_strips_whitespace():
    assert normalize_animal_letter("  L  ") == "L"


def test_normalize_empty_returns_none():
    assert normalize_animal_letter("") is None
    assert normalize_animal_letter("   ") is None


def test_normalize_non_letter_returns_none():
    assert normalize_animal_letter("5") is None
    assert normalize_animal_letter("!abc") is None


# ----------------------------------------------------------------------
# compute_summary -- pure
# ----------------------------------------------------------------------

def test_summary_basic_eligible_and_skipped():
    """Three Js + one F + one unknown + one held-out L:
      eligible = 1 (J),
      skipped  = 1 (F),
      unknown  = 1 (the missing-letter row),
      groups   = {J: 3, F: 1}  (held-out L is dropped)
    """
    recs = [
        {"recording_id": "a", "animal": "J", "held_out": False},
        {"recording_id": "b", "animal": "J", "held_out": False},
        {"recording_id": "c", "animal": "J", "held_out": False},
        {"recording_id": "d", "animal": "F", "held_out": False},
        {"recording_id": "e", "animal": None, "held_out": False},
        {"recording_id": "f", "animal": "L", "held_out": True},
    ]
    s = compute_summary(recs, min_recordings_per_animal=3)
    assert s == {
        "n_eligible": 1,
        "n_skipped": 1,
        "unknown": 1,
        "groups": {"J": 3, "F": 1},
    }


def test_summary_threshold_changes():
    """With threshold=4, the same J group (count 3) drops from
    eligible to skipped -- guards against off-by-one in the
    >= comparison."""
    recs = [{"recording_id": f"r{i}", "animal": "J"} for i in range(3)]
    s3 = compute_summary(recs, min_recordings_per_animal=3)
    s4 = compute_summary(recs, min_recordings_per_animal=4)
    assert s3["n_eligible"] == 1 and s3["n_skipped"] == 0
    assert s4["n_eligible"] == 0 and s4["n_skipped"] == 1


def test_summary_treats_lowercase_animal_as_same_group():
    """A 'j' and a 'J' should collapse into one group, not two."""
    recs = [
        {"recording_id": "a", "animal": "J"},
        {"recording_id": "b", "animal": "j"},
        {"recording_id": "c", "animal": "Jel"},
    ]
    s = compute_summary(recs, min_recordings_per_animal=3)
    assert s["groups"] == {"J": 3}
    assert s["n_eligible"] == 1


# ----------------------------------------------------------------------
# Widget behavior -- runs only when a real Qt display is available
# ----------------------------------------------------------------------
#
# The widget tests use pytest-qt's `qtbot` fixture (auto-creates a
# QApplication). Plain construction of QApplication crashes on truly
# headless boxes (no Qt platform plugin can initialize) so we mark
# these tests `qt_display` and default-skip them; run with
# `pytest -m qt_display` from a Mac terminal with a real display.


@pytest.mark.qt_display
def test_widget_populates_from_recordings(qtbot):
    """set_recordings populates the table with one row per recording
    and orders by (known-animals-first, animal, held-out-last)."""
    from ui.widgets.per_animal_table import (
        PerAnimalTable, ANIMAL_COL, RID_COL,
    )
    t = PerAnimalTable()
    qtbot.addWidget(t)
    recs = [
        {"recording_id": "rec_F1", "animal": "F", "rec_type": "baseline"},
        {"recording_id": "rec_J2", "animal": "J", "rec_type": "stim_rec"},
        {"recording_id": "rec_J1", "animal": "J", "rec_type": "baseline"},
        {"recording_id": "rec_unk", "animal": None,
         "rec_type": "baseline"},
    ]
    t.set_recordings(recs)
    assert t.rowCount() == 4
    # Order: F, J1, J2, unknown (unknown last; J sorted by RID within).
    actual_rids = [t.item(i, RID_COL).text() for i in range(4)]
    assert actual_rids == ["rec_F1", "rec_J1", "rec_J2", "rec_unk"]
    # The unknown row's Animal cell is empty (not the string "None").
    assert t.item(3, ANIMAL_COL).text() == ""


@pytest.mark.qt_display
def test_widget_edit_normalizes_and_emits_signal(qtbot):
    """When the user types 'jel' into the Animal cell, the widget
    normalizes it to 'J', fires animal_edited(recording_id, 'J')
    and updates the cached recording state."""
    from PySide6.QtWidgets import QTableWidgetItem    # noqa: F401
    from ui.widgets.per_animal_table import PerAnimalTable, ANIMAL_COL
    t = PerAnimalTable()
    qtbot.addWidget(t)
    t.set_recordings([
        {"recording_id": "rec_a", "animal": None},
        {"recording_id": "rec_b", "animal": "F"},
    ])
    captured: list[tuple[str, object]] = []
    t.animal_edited.connect(
        lambda rid, letter: captured.append((rid, letter))
    )
    # Find rec_a's row (sort order: known animals first, so rec_a is
    # last) and simulate the user typing "jel".
    a_row = None
    for i in range(t.rowCount()):
        if t.item(i, 1).text() == "rec_a":
            a_row = i
            break
    assert a_row is not None
    t.item(a_row, ANIMAL_COL).setText("jel")
    assert captured == [("rec_a", "J")]
    # The cell was snapped to the normalized form so the user sees
    # the saved value.
    assert t.item(a_row, ANIMAL_COL).text() == "J"
    # And the cached recordings list reflects the edit so summary()
    # picks it up.
    s = t.summary()
    # rec_b has F (1 rec), rec_a now has J (1 rec).
    assert s["groups"] == {"F": 1, "J": 1}
    assert s["unknown"] == 0


@pytest.mark.qt_display
def test_widget_summary_live_during_edits(qtbot):
    """Edits to the Animal column should update summary() output
    without a manual refresh."""
    from ui.widgets.per_animal_table import PerAnimalTable, ANIMAL_COL
    t = PerAnimalTable()
    qtbot.addWidget(t)
    t.set_min_recordings_per_animal(2)
    t.set_recordings([
        {"recording_id": "x", "animal": None},
        {"recording_id": "y", "animal": None},
        {"recording_id": "z", "animal": None},
    ])
    assert t.summary() == {
        "n_eligible": 0, "n_skipped": 0, "unknown": 3, "groups": {},
    }
    # Assign all three to animal J.
    for i in range(3):
        t.item(i, ANIMAL_COL).setText("J")
    s = t.summary()
    assert s["unknown"] == 0
    assert s["groups"] == {"J": 3}
    assert s["n_eligible"] == 1
    assert s["n_skipped"] == 0


@pytest.mark.qt_display
def test_widget_held_out_rows_excluded_from_summary(qtbot):
    """Held-out rows show up in the table but never count toward
    eligibility, matching the orchestrator's behavior."""
    from ui.widgets.per_animal_table import PerAnimalTable
    t = PerAnimalTable()
    qtbot.addWidget(t)
    t.set_recordings([
        {"recording_id": "a", "animal": "J", "held_out": False},
        {"recording_id": "b", "animal": "J", "held_out": False},
        {"recording_id": "c", "animal": "J", "held_out": True},
    ])
    s = t.summary()
    assert s["groups"] == {"J": 2}
    assert s["n_eligible"] == 0  # only 2 non-heldout, threshold=3
