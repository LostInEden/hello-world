"""Overlay animation math and state plumbing (no display needed)."""

from whisperflow.overlay import BAR_MAX, BAR_MIN, Overlay, bar_heights


def test_hidden_state_is_flat():
    assert bar_heights(Overlay.HIDDEN, phase=1.23, level=0.9) == [BAR_MIN] * 5


def test_recording_bars_scale_with_level():
    quiet = bar_heights(Overlay.RECORDING, phase=1.0, level=0.0)
    loud = bar_heights(Overlay.RECORDING, phase=1.0, level=1.0)
    assert sum(loud) > sum(quiet)
    # even silence breathes a little, so the pill reads as "listening"
    assert sum(quiet) > BAR_MIN * 5
    assert all(BAR_MIN <= h <= BAR_MAX for h in quiet + loud)


def test_processing_wave_moves_over_time():
    a = bar_heights(Overlay.PROCESSING, phase=0.0, level=0.0)
    b = bar_heights(Overlay.PROCESSING, phase=0.4, level=0.0)
    assert a != b  # the wave travels regardless of mic level
    assert all(BAR_MIN <= h <= BAR_MAX for h in a + b)


def test_set_state_and_close_are_plain_flags():
    overlay = Overlay({"position": "bottom_center"})
    assert overlay._state == Overlay.HIDDEN
    overlay.set_state(Overlay.RECORDING)
    assert overlay._state == Overlay.RECORDING
    overlay.close()
    assert overlay._closing


def test_level_provider_defaults_to_zero():
    overlay = Overlay({})
    assert overlay._level_provider() == 0.0
