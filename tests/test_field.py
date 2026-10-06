from __future__ import annotations

import numpy as np
import pytest

from anyloads import (
    CaseContext,
    LoadError,
    PressureField,
    PressureTable,
    UntrustedCodeError,
    check_fluid_side,
    fluid_sign,
    trust_code,
)

POINTS = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])


def test_exactly_one_source_is_required():
    with pytest.raises(LoadError, match="exactly one"):
        PressureField()
    with pytest.raises(LoadError, match="exactly one"):
        PressureField(value=1.0, expression="1")
    with pytest.raises(LoadError, match="finite"):
        PressureField(value=float("nan"))
    with pytest.raises(LoadError, match="finite"):
        PressureField(value=1.0, scale=float("inf"))
    with pytest.raises(LoadError, match="table case only applies"):
        PressureField(value=1.0, table_case=1)
    with pytest.raises(LoadError):
        PressureField(expression="__import__('os')")
    with pytest.raises(LoadError):
        PressureField(code="def nope(): pass")


def test_every_source_evaluates_with_the_cases_identity(tmp_path):
    case = CaseContext(number=2, name="storm", time=3.0, parameters={"draft": 2.0})
    assert PressureField(value=500.0).evaluate(case, POINTS) == pytest.approx([500.0] * 3)
    assert PressureField(expression="10 * max(0, draft - z) * t").evaluate(
        case, POINTS
    ) == pytest.approx([45.0, 15.0, 0.0])
    source = "def pressure(case, z):\n    return 100.0 * case + z\n"
    trust_code(source)
    assert PressureField(code=source).evaluate(case, POINTS) == pytest.approx(
        [200.5, 201.5, 202.5]
    )

    path = tmp_path / "p.csv"
    path.write_text(
        "case,x,y,z,pressure\n" + "\n".join(f"2,0,0,{z},{p}" for z, p in ((0.5, 10), (1.5, 20), (2.5, 30))),
        encoding="utf-8",
    )
    table = PressureTable.from_file(path, key="number")
    assert PressureField(table=table).evaluate(case, POINTS) == pytest.approx([10, 20, 30])
    # pinned to another identifier regardless of the case
    other = CaseContext(number=9)
    assert PressureField(table=table, table_case=2).evaluate(other, POINTS) == pytest.approx(
        [10, 20, 30]
    )


def test_scale_multiplies_the_result():
    field = PressureField(expression="z", scale=1000.0)
    assert field.evaluate(CaseContext(), POINTS) == pytest.approx([500.0, 1500.0, 2500.0])


def test_a_missing_time_is_explained():
    with pytest.raises(LoadError, match="unknown name 't'.*no time"):
        PressureField(expression="sin(t)").evaluate(CaseContext(), POINTS)


def test_untrusted_code_in_a_field_does_not_run():
    field = PressureField(code="def pressure(z):\n    return 1.0\n")
    with pytest.raises(UntrustedCodeError):
        field.evaluate(CaseContext(), POINTS)


def test_plain_data_round_trip_parses_but_never_runs_code(tmp_path):
    marker = tmp_path / "ran.txt"
    source = (
        f"open({str(marker)!r}, 'w').write('x')\n"
        "def pressure(z):\n    return 1.0\n"
    )
    field = PressureField(code=source, code_vectorized=True, scale=2.0)
    data = field.to_dict()
    assert PressureField.from_dict(data) == field
    assert not marker.exists()
    assert PressureField.from_dict(PressureField(expression="z").to_dict()).kind == "expression"
    assert PressureField.from_dict(PressureField(value=1.5).to_dict()).value == 1.5


def test_case_parameters_cannot_shadow_the_vocabulary():
    with pytest.raises(LoadError, match="reserved"):
        CaseContext(parameters={"max": 1.0})
    with pytest.raises(LoadError, match="reserved"):
        CaseContext(parameters={"t": 1.0})
    with pytest.raises(LoadError, match="identifier"):
        CaseContext(parameters={"not a name": 1.0})
    with pytest.raises(LoadError, match="finite"):
        CaseContext(parameters={"a": float("nan")})


def test_fluid_sign_turns_a_magnitude_into_a_normal_signed_pressure():
    normal = np.array([0.0, 0.0, 1.0])
    centre = np.array([0.0, 0.0, 0.0])
    assert fluid_sign("positive_normal", None, centre, normal) == -1.0
    assert fluid_sign("negative_normal", None, centre, normal) == 1.0
    below = np.array([0.0, 0.0, -5.0])  # interior below: fluid above, pushes down
    assert fluid_sign("away_from_point", below, centre, normal) == -1.0
    above = np.array([0.0, 0.0, 5.0])
    assert fluid_sign("away_from_point", above, centre, normal) == 1.0
    with pytest.raises(LoadError, match="plane of an element"):
        fluid_sign("away_from_point", np.array([1.0, 0.0, 0.0]), centre, normal, label="hull")


def test_fluid_side_validation():
    with pytest.raises(LoadError, match="fluid side"):
        check_fluid_side("inside", None)
    with pytest.raises(LoadError, match="needs that point"):
        check_fluid_side("away_from_point", None)
    with pytest.raises(LoadError, match="only applies"):
        check_fluid_side("positive_normal", (0, 0, 0))
    with pytest.raises(LoadError, match="three finite"):
        check_fluid_side("away_from_point", (0, 0))
    assert check_fluid_side("away_from_point", (1, 2, 3)).tolist() == [1.0, 2.0, 3.0]
