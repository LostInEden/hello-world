"""Overlay animation math and state plumbing (no display needed)."""

from whisperflow.overlay import (
    BAR_COUNT,
    BAR_MAX,
    BAR_MIN,
    Overlay,
    level_to_height,
    wave_heights,
)


def test_level_to_height_bounds_and_monotonic():
    assert level_to_height(0.0) == BAR_MIN
    assert level_to_height(1.0) == BAR_MAX
    assert level_to_height(-5.0) == BAR_MIN  # clamped
    assert level_to_height(5.0) == BAR_MAX
    samples = [level_to_height(x / 10) for x in range(11)]
    assert samples == sorted(samples)


def test_quiet_speech_is_visibly_lifted():
    # the gamma curve should make a 20% level clearly taller than minimum
    assert level_to_height(0.2) > BAR_MIN + 0.25 * (BAR_MAX - BAR_MIN)


def test_wave_moves_over_time_and_stays_in_bounds():
    a = wave_heights(0.0)
    b = wave_heights(0.4)
    assert len(a) == BAR_COUNT
    assert a != b  # the wave travels
    assert all(BAR_MIN <= h <= BAR_MAX for h in a + b)


def test_recording_targets_scroll_the_level_history():
    overlay = Overlay({})
    overlay.set_state(Overlay.RECORDING)
    overlay._level_provider = lambda: 0.8
    t = 100.0
    first = overlay._targets(t)
    assert first[-1] == level_to_height(0.8)  # newest sample enters on the right
    assert first[:-1] == [BAR_MIN] * (BAR_COUNT - 1)  # rest still silent

    overlay._level_provider = lambda: 0.2
    second = overlay._targets(t + 0.06)  # push interval elapsed -> scrolls left
    assert second[-1] == level_to_height(0.2)
    assert second[-2] == level_to_height(0.8)  # previous sample moved left


def test_recording_targets_hold_between_pushes():
    overlay = Overlay({})
    overlay.set_state(Overlay.RECORDING)
    overlay._level_provider = lambda: 0.8
    t = 100.0
    first = overlay._targets(t)
    again = overlay._targets(t + 0.01)  # within the push interval: no scroll
    assert first == again


def test_hidden_targets_are_flat():
    overlay = Overlay({})
    assert overlay._targets(0.0) == [BAR_MIN] * BAR_COUNT


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
