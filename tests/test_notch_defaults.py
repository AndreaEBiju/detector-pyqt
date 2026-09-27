"""Change 3: the mains-notch default is 60 Hz only, and harmonics stay off."""

from __future__ import annotations

from ui.data.settings import DEFAULTS


def test_only_the_sixty_hertz_notch_is_on_by_default() -> None:
    """A 120 Hz notch rings INSIDE the 300-3000 Hz ENG band at 2.41 uV per mV of
    excursion, so a 7.5 mV motion excursion puts its ringing at the 4.5-sigma
    spike threshold - the filter manufactures the spikes the consumer then counts.
    60 Hz contributes 0.221 uV/mV, an order of magnitude less, because the
    bandpass attenuates it by -36.4 dB through filtfilt.
    """
    freqs = DEFAULTS["preprocessing_default_freqs_hz"]

    assert freqs == [60.0], f"harmonics must not be on by default, got {freqs}"
    assert 120.0 not in freqs
    assert 180.0 not in freqs


def test_no_harmonic_autodetection_setting_exists() -> None:
    """Detection would re-enable the harmonics this default exists to remove."""
    assert not any("harmonic" in k and "detect" in k for k in DEFAULTS)
