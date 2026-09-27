"""Only sync_viewer may touch the candidate overlay. Asserted structurally.

The offscreen test proves the CONTROLLER keeps the scene clean. It does not
prove nothing else touches the viewer. A second call path - a dock restoring
state, a worker finishing late, a convenience helper added in six months -
bypasses every phase guard and that test still passes, because it only ever
exercises the path it knows about.

So this walks the audit package's ASTs and asserts the call site is unique. It
is the same kind of check as the bands/stim_split import-graph assertion: a
property of the code's shape, which a reviewer reading a diff cannot reliably
verify by eye.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "ui"
OVERLAY_SETTER = "set_model_intervals"
OWNER = UI / "audit" / "controller.py"


def _calls_to(path: Path, name: str) -> list[int]:
    """Line numbers of every ``*.name(...)`` call in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def _audit_modules() -> list[Path]:
    return sorted((UI / "audit").rglob("*.py"))


def test_only_the_controller_sets_the_candidate_overlay() -> None:
    """Every phase guard lives in sync_viewer. A second caller bypasses them all."""
    offenders = {
        p.relative_to(UI).as_posix(): _calls_to(p, OVERLAY_SETTER)
        for p in _audit_modules()
        if p != OWNER and _calls_to(p, OVERLAY_SETTER)
    }

    assert not offenders, (
        f"{OVERLAY_SETTER} is called outside controller.py: {offenders}. "
        "Route it through sync_viewer, which is the only place that consults "
        "the phase - a second path shows candidates during blind marking and "
        "the offscreen test cannot see it."
    )


def test_the_controller_sets_it_in_exactly_one_function() -> None:
    """Two call sites inside the owner is the same hazard, one file in."""
    tree = ast.parse(OWNER.read_text(encoding="utf-8"))
    owners = [
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == OVERLAY_SETTER
            for n in ast.walk(fn)
        )
    ]

    assert owners == ["sync_viewer"], f"overlay set in {owners}, expected sync_viewer"


def test_the_z_dock_never_touches_the_candidate_overlay() -> None:
    """The dock renders z; candidates are not its business. Keeping them apart
    is why the reveal has exactly one gate rather than two to keep in step."""
    dock = UI / "widgets" / "ztrace_dock.py"

    assert not _calls_to(dock, OVERLAY_SETTER)


def test_nothing_in_the_audit_package_reads_session_private_reveal_state() -> None:
    """``_revealed`` is the store the phase property guards. Reading it directly
    would return candidates regardless of phase - the guard bypassed from below
    rather than from above."""
    offenders = {}
    for p in _audit_modules():
        if p.name == "session.py":
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        hits = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "_revealed"
        ]
        if hits:
            offenders[p.name] = hits

    assert not offenders, f"_revealed read outside session.py: {offenders}"
