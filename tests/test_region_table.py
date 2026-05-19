"""Unit tests for ui.widgets.region_table.

Tests only the pure-Python pieces (`format_time`) so they run without
a display. The QTableWidget itself is exercised by manual UX checks
+ a pytest-qt test in M2's DoD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ui.widgets.region_table import format_time


def test_format_time_basic():
    assert format_time(0.0) == "00:00.000"
    assert format_time(7.5) == "00:07.500"
    assert format_time(60.0) == "01:00.000"
    assert format_time(125.123) == "02:05.123"


def test_format_time_negative():
    assert format_time(-5.0) == "-00:05.000"


def test_format_time_milliseconds_three_decimals():
    # `%.3f` truncates after 3 decimals (with %f's standard round-half-
    # to-even); verify the truncation behavior so callers know what
    # they're getting.
    assert format_time(0.0001) == "00:00.000"
    assert format_time(0.4994) == "00:00.499"
    # Exactly 0.5 rounds to 500 (half-to-even on the boundary).
    assert format_time(0.5) == "00:00.500"
