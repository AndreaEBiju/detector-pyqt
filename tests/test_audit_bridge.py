"""The bridge must reduce correctly and must not lose NaN meaning."""

from __future__ import annotations

import numpy as np

from ui.audit.bridge import BandTrace, reduce_to_band_traces


def test_six_bands_collapse_to_six_traces_not_fifty_four() -> None:
    """Nine signals x six bands is unreadable; the panel shows one per band."""
    bands = ("0-2", "0.5-3", "2-50", "10-150", "100-300", "300-3000")
    signals = [f"S{i}" for i in range(9)]
    z = {(s, b): np.zeros(100) for s in signals for b in bands}

    traces = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)

    assert len(z) == 54
    assert len(traces) == 6
    assert sorted(t.band for t in traces) == sorted(bands)


def test_the_reduction_is_a_maximum_over_every_signal() -> None:
    """Including raw contacts, per invariant 6 - motion appears on them too, and
    a reduction that skipped them could hide the evidence being looked for."""
    z = {
        ("RVN1", "10-150"): np.array([1.0, 5.0, 2.0]),
        ("T", "10-150"): np.array([4.0, 2.0, 2.0]),
        ("ANT1", "10-150"): np.array([0.0, 0.0, 9.0]),
    }

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    np.testing.assert_allclose(t.z_max, [4.0, 5.0, 9.0])
    assert t.winner == ("T", "RVN1", "ANT1")


def test_which_signal_won_is_kept_per_frame_for_click_to_expand() -> None:
    """Discarding it would make the panel unable to answer 'which channel?'."""
    z = {("A", "2-50"): np.array([1.0, 7.0]), ("B", "2-50"): np.array([6.0, 2.0])}

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    assert t.at(0.000) == (6.0, "B")
    assert t.at(0.010) == (7.0, "A")


def test_a_nan_on_one_signal_does_not_poison_the_frame() -> None:
    """NaN means 'not assessable here', not 'low z'. A plain max would turn a
    frame that is perfectly valid on eight signals into NaN and hide it."""
    z = {
        ("A", "0-2"): np.array([np.nan, 2.0]),
        ("B", "0-2"): np.array([3.0, np.nan]),
    }

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    np.testing.assert_allclose(t.z_max, [3.0, 2.0])
    assert t.winner == ("B", "A")


def test_a_frame_invalid_on_every_signal_stays_nan_with_no_winner() -> None:
    """It must not be reported as a low-z frame - that is the opposite claim."""
    z = {("A", "0-2"): np.array([np.nan]), ("B", "0-2"): np.array([np.nan])}

    t = reduce_to_band_traces(z, z_enter=3.0, grid_s=0.010)[0]

    assert np.isnan(t.z_max[0])
    assert t.winner == ("",)


def test_z_enter_travels_with_the_trace() -> None:
    """The threshold has to be drawable on the plot: 'z was low' and 'z was high
    but under the line' need different fixes and are the audit's whole point."""
    t = reduce_to_band_traces({("A", "0-2"): np.zeros(3)}, z_enter=2.5, grid_s=0.01)[0]

    assert t.z_enter == 2.5


def test_a_time_outside_the_trace_returns_nan_rather_than_wrapping() -> None:
    """Negative indexing would silently report the end of the record."""
    t = BandTrace("0-2", np.array([1.0, 2.0]), ("A", "A"), 0.010, 3.0)

    assert np.isnan(t.at(-1.0)[0])
    assert np.isnan(t.at(99.0)[0])


def _stub_reveal(monkeypatch, signals: list[str], screened: dict[str, tuple]) -> dict:
    """Stub reveal_for_region's stages; return what it passed downstream.

    ``screened`` maps a contact label (``"L3"``) to the reasons its screen fired.
    """
    import dataclasses
    from types import SimpleNamespace

    from gems_blanking_v2.derive import contact_quality, derivations
    from gems_blanking_v2.physio import rpeaks

    from ui.audit import bridge

    seen: dict = {}
    monkeypatch.setattr(derivations, "build_derivations",
                        lambda rec: ({n: np.zeros(rec.data.shape[0]) for n in signals}, {}))
    quality = {lab: SimpleNamespace(cuff_id=lab[0], contact_index=int(lab[1:]), reasons=why,
                                    screened=bool(why))
               for lab, why in screened.items()}
    monkeypatch.setattr(contact_quality, "assess_contacts", lambda rec: quality)
    monkeypatch.setattr(rpeaks, "detect_rpeaks", lambda x, fs: None)

    def z_by_pair(stack, fs, names):
        seen["names"] = list(names)
        return {(n, "0-2"): np.zeros(10) for n in names}

    def capture(z, beats, **kw):
        seen["z_enter"] = kw["z_enter"]
        return SimpleNamespace(candidates=[])

    monkeypatch.setattr(bridge, "z_by_pair", z_by_pair)
    monkeypatch.setattr(bridge, "candidates_for", capture)
    monkeypatch.setattr(dataclasses, "replace", lambda r, data: SimpleNamespace(fs=r.fs, data=data))
    rec = SimpleNamespace(fs=100.0, data=np.zeros((1000, 1)))
    _intervals, seen["traces"] = bridge.reveal_for_region(rec, (0.0, 10.0))
    return seen


def test_the_reveal_uses_the_generators_pinned_threshold(monkeypatch) -> None:
    """One source for z_enter: the reveal must follow ``recall.Z_ENTER``.

    A literal default here would keep revealing at the old threshold the day task
    09 pins a new one, while the miss diagnosis moved with the generator - two
    halves of one score using two thresholds, and nothing would raise.
    """
    from gems_blanking_v2.detect import recall

    monkeypatch.setattr(recall, "Z_ENTER", 2.25)
    seen = _stub_reveal(monkeypatch, ["R_T"], {})

    assert seen["z_enter"] == 2.25
    assert {t.z_enter for t in seen["traces"]} == {2.25}


def test_a_screened_contact_leaves_the_max_with_its_cuffs_tripole(monkeypatch) -> None:
    """Invariant 41: an off-cuff contact's z means nothing, and neither does its T."""
    signals = ["L_T", "L_V1", "L_V2", "L_V3", "R_T", "R_V1", "R_V2", "R_V3", "stomach_ref"]
    seen = _stub_reveal(monkeypatch, signals, {"L3": ("uncorrelated",), "L1": (), "R2": ()})

    assert seen["names"] == ["L_V1", "L_V2", "R_T", "R_V1", "R_V2", "R_V3", "stomach_ref"]
    assert _stub_reveal(monkeypatch, signals, {"L3": ()})["names"] == signals
