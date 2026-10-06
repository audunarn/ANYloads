from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from anyloads import (
    UntrustedCodeError,
    UserCodeError,
    check_code,
    code_hash,
    is_trusted,
    run_pressure_code,
    trust_code,
    trust_store_path,
)

POINTS = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])


def code(text):
    return textwrap.dedent(text).strip() + "\n"


def run(source, **overrides):
    arguments = dict(case=1, name="c", time=None, parameters={}, centroids=POINTS)
    arguments.update(overrides)
    trust_code(source)
    return run_pressure_code(source, **arguments)


@pytest.mark.parametrize(
    "source, message",
    [
        ("", "empty"),
        ("def pressure(case, x, y, z):\n  return (", "line"),
        ("x = 1", "must define"),
        ("def other(x): return 1", "must define"),
        ("def pressure(case, q): return 1", "unsupported argument"),
        ("def pressure(*a): return 1", "plain positional"),
        ("def pressure(case, x=1): return 1", "defaults"),
        ("def pressure(case, **k): return 1", "plain positional"),
    ],
)
def test_malformed_code_is_refused_without_running_it(source, message):
    with pytest.raises(UserCodeError, match=message):
        check_code(source)


def test_check_returns_the_declared_arguments():
    assert check_code("def pressure(case, x, y, z): return 0") == ("case", "x", "y", "z")
    assert check_code("def pressure(z, t, name, parameters): return 0") == (
        "z", "t", "name", "parameters",
    )


def test_code_gets_case_number_and_position_per_point():
    values = run(code(
        """
        def pressure(case, x, y, z):
            return 1000.0 * case + z
        """
    ), case=3)
    assert values == pytest.approx([3000.5, 3001.5, 3002.5])


def test_only_declared_arguments_are_passed():
    values = run(code(
        """
        def pressure(t, name, parameters):
            assert name == "wave"
            return parameters["amp"] * t
        """
    ), name="wave", time=2.0, parameters={"amp": 500.0})
    assert values == pytest.approx([1000.0] * 3)


def test_vectorized_code_is_called_once_with_arrays():
    values = run(code(
        """
        def pressure(x, y, z):
            return np.maximum(0.0, 2.0 - z)
        """
    ), vectorized=True)
    assert values == pytest.approx([1.5, 0.5, 0.0])


def test_modules_are_available_and_imports_work():
    values = run(code(
        """
        import math

        def pressure(z):
            return math.sqrt(z) + np.float64(0.0)
        """
    ))
    assert values == pytest.approx(np.sqrt(POINTS[:, 2]))


def test_a_runtime_error_names_the_line():
    with pytest.raises(UserCodeError, match=r"line 3.*ZeroDivisionError"):
        run(code(
            """
            def pressure(case, x, y, z):
                ratio = 1.0
                return ratio / (z - z)
            """
        ))


def test_bad_returns_are_refused():
    with pytest.raises(UserCodeError, match="non-finite"):
        run("def pressure(z):\n    return float('nan')\n")
    with pytest.raises(UserCodeError, match="ValueError"):
        run("def pressure(z):\n    return 'a lot'\n")


def test_untrusted_code_does_not_run(tmp_path):
    marker = tmp_path / "ran.txt"
    source = (
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "def pressure(z):\n    return 1.0\n"
    )
    check_code(source)  # parsing is fine and runs nothing
    assert not marker.exists()
    with pytest.raises(UntrustedCodeError, match="not been trusted"):
        run_pressure_code(source, case=1, name="c", time=None, parameters={}, centroids=POINTS)
    assert not marker.exists()
    trust_code(source)
    run_pressure_code(source, case=1, name="c", time=None, parameters={}, centroids=POINTS)
    assert marker.exists()


def test_trust_is_per_exact_text():
    source = "def pressure(z):\n    return 1.0\n"
    trust_code(source)
    assert is_trusted(source)
    assert not is_trusted(source + "# edited\n")
    assert code_hash(source) != code_hash(source + " ")


def test_trust_refuses_code_that_does_not_parse():
    with pytest.raises(UserCodeError):
        trust_code("def pressure(:")
    assert not is_trusted("def pressure(:")


def test_the_trust_store_survives_garbage_and_records_only_hashes(isolated_trust_store):
    isolated_trust_store.write_text("not json", encoding="utf-8")
    source = "def pressure(z):\n    return 1\n"
    assert not is_trusted(source)
    trust_code(source)
    stored = json.loads(isolated_trust_store.read_text())
    assert stored == {"sha256": [code_hash(source)]}
    assert trust_store_path() == isolated_trust_store
    trust_code(source)  # idempotent
    assert json.loads(isolated_trust_store.read_text())["sha256"] == [code_hash(source)]
