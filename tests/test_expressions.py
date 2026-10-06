from __future__ import annotations

import numpy as np
import pytest

from anyloads import (
    ExpressionError,
    LoadError,
    check_expression,
    evaluate_expression,
    expression_names,
)


def test_a_formula_is_evaluated_for_every_point_with_functions_and_constants():
    z = np.array([0.0, 1.0, 2.0, 3.0])
    value = evaluate_expression(
        "rho * g * max(0, draft - z)", {"z": z, "rho": 1000.0, "draft": 2.0}
    )
    assert value == pytest.approx(1000.0 * 9.80665 * np.array([2.0, 1.0, 0.0, 0.0]))


def test_a_constant_formula_still_gives_one_value_per_point():
    assert evaluate_expression("5", {}, size=3).tolist() == [5.0, 5.0, 5.0]


def test_comparisons_chains_and_where():
    x = np.array([0.0, 1.0, 2.0])
    assert evaluate_expression("where(x > 0.5, 1, 0)", {"x": x}).tolist() == [0.0, 1.0, 1.0]
    assert evaluate_expression("0 < x < 2", {"x": x}).tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('x')",
        "open('f')",
        "x.real",
        "x[0]",
        "lambda: 1",
        "[1, 2]",
        "'text'",
        "True",
        "sin(x=1)",
        "x if x else 1",
        "",
        "1 +",
    ],
)
def test_anything_outside_the_vocabulary_is_refused(text):
    with pytest.raises(ExpressionError):
        evaluate_expression(text, {"x": np.zeros(2)})
    with pytest.raises(LoadError):
        check_expression(text)


def test_an_unknown_name_and_a_non_finite_result_are_errors():
    with pytest.raises(ExpressionError, match="unknown name 'draft'"):
        evaluate_expression("draft - z", {"z": np.zeros(2)})
    with pytest.raises(ExpressionError, match="non-finite"):
        evaluate_expression("1 / x", {"x": np.array([1.0, 0.0])})


def test_expression_names_lists_variables_not_functions():
    assert expression_names("rho*g*max(0, d - z)") == {"rho", "g", "d", "z"}
