from __future__ import annotations

import math

import pytest

from anyloads import CaseContext, Envelope, LoadError


def case(number, time=None, **parameters):
    return CaseContext(number=number, name=f"c{number}", time=time, parameters=parameters)


def test_no_window_and_no_factor_is_always_one():
    envelope = Envelope()
    assert envelope.factor(case(1)) == 1.0
    assert envelope.factor(case(999, time=None)) == 1.0
    assert not envelope.uses_time


def test_window_by_number_is_inclusive_at_both_ends():
    envelope = Envelope(start=20, stop=300)
    factors = {n: envelope.factor(case(n)) for n in (19, 20, 21, 299, 300, 301)}
    assert factors == {19: 0.0, 20: 1.0, 21: 1.0, 299: 1.0, 300: 1.0, 301: 0.0}


def test_either_end_may_be_open():
    assert Envelope(start=5).factor(case(4)) == 0.0 and Envelope(start=5).factor(case(10**6)) == 1.0
    assert Envelope(stop=5).factor(case(5)) == 1.0 and Envelope(stop=5).factor(case(6)) == 0.0


def test_window_by_time_tolerates_floating_point_steps():
    envelope = Envelope(start=0.3, stop=1.0, by="time")
    third = 3 * 0.1  # 0.30000000000000004
    assert envelope.factor(case(4, time=third)) == 1.0
    assert envelope.factor(case(3, time=0.2)) == 0.0
    assert envelope.factor(case(11, time=1.0)) == 1.0
    assert envelope.factor(case(12, time=1.1)) == 0.0
    with pytest.raises(LoadError, match="needs a case with a time"):
        envelope.factor(case(1))


def test_expression_reads_time_parameters_and_the_window_clock():
    envelope = Envelope(expression="mean + swing * sin(omega * t)", start=1, stop=100)
    value = envelope.factor(case(5, time=2.0, mean=0.3, swing=0.4, omega=1.5))
    assert value == pytest.approx(0.3 + 0.4 * math.sin(3.0))
    steps = Envelope(expression="k", start=20)
    assert steps.factor(case(25)) == 5.0 and steps.factor(case(19)) == 0.0
    clock = Envelope(expression="tau", start=2.0, by="time")
    assert clock.factor(case(9, time=5.5)) == pytest.approx(3.5)
    by_number = Envelope(expression="tau", start=20)
    assert by_number.factor(case(25, time=2.5), start_time=2.0) == pytest.approx(0.5)
    with pytest.raises(LoadError, match="tau"):
        by_number.factor(case(25, time=2.5))
    assert Envelope(expression="t").uses_time and not Envelope(expression="2").uses_time


def test_points_ramp_in_hold_and_ramp_out():
    ramp = Envelope(points=((0, 0), (10, 1), (50, 1), (60, 0)))
    assert ramp.factor(case(1, time=5.0)) == pytest.approx(0.5)
    assert ramp.factor(case(1, time=30.0)) == 1.0
    assert ramp.factor(case(1, time=55.0)) == pytest.approx(0.5)
    assert ramp.factor(case(1, time=99.0)) == 0.0  # held at the last point
    windowed = Envelope(points=((0, 0), (10, 1)), start=3, stop=4)
    assert windowed.factor(case(2, time=5.0)) == 0.0  # outside the window
    with pytest.raises(LoadError, match="no"):
        ramp.factor(case(1))


def test_invalid_envelopes_are_refused():
    with pytest.raises(LoadError, match="before its start"):
        Envelope(start=10, stop=5)
    with pytest.raises(LoadError, match="number or time|by"):
        Envelope(by="step")
    with pytest.raises(LoadError, match="not both"):
        Envelope(expression="1", points=((0, 1), (1, 1)))
    with pytest.raises(LoadError, match="at least two"):
        Envelope(points=((0, 1),))
    with pytest.raises(LoadError, match="increase"):
        Envelope(points=((0, 1), (0, 2)))
    with pytest.raises(LoadError):
        Envelope(expression="__import__('os')")
    with pytest.raises(LoadError, match="finite"):
        Envelope(start=float("nan"))
    with pytest.raises(LoadError, match="flag"):
        Envelope(start=True)
    with pytest.raises(LoadError, match="unknown name"):
        Envelope(expression="undefined * 2").factor(case(1, time=1.0))


def test_round_trip_and_coercion_and_names():
    envelope = Envelope(start=20, stop=300, expression="a + sin(omega * t)")
    assert Envelope.from_dict(envelope.to_dict()) == envelope
    points = Envelope(points=((0, 0), (1, 1)), by="time", start=0.5)
    assert Envelope.from_dict(points.to_dict()) == points
    assert Envelope.coerce(None) is None and Envelope.coerce(envelope) is envelope
    assert Envelope.coerce({"start": 3}).start == 3.0
    assert envelope.names() == {"a", "omega"}
    with pytest.raises(LoadError, match="unknown envelope field"):
        Envelope.from_dict({"begin": 1})
    hash(envelope)
