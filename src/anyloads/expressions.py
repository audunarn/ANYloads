"""Safe, vectorised scalar expressions for field-valued loads.

A pressure equation such as ``rho * g * max(0, draft - z)`` is stored as text
in the project and evaluated over every element centroid at once.  The text is
parsed with :mod:`ast` and only a fixed vocabulary is admitted -- arithmetic,
comparison, a handful of numpy functions, and named numbers -- so a project
file can never execute anything.  Anything else is refused when the expression
is created, not when a solve happens to reach it.
"""

from __future__ import annotations

import ast
import math
from typing import Callable, Dict, Mapping

import numpy as np

from .errors import LoadError

__all__ = [
    "BUILTIN_CONSTANTS",
    "COORDINATE_NAMES",
    "ExpressionError",
    "RESERVED_NAMES",
    "check_expression",
    "evaluate_expression",
    "expression_names",
]

#: Per-element variables an expression may always use: the centroid position.
COORDINATE_NAMES: tuple[str, ...] = ("x", "y", "z")

BUILTIN_CONSTANTS: Mapping[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "g": 9.80665,
}

_FUNCTIONS: Mapping[str, Callable[..., np.ndarray]] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "cosh": np.cosh,
    "sinh": np.sinh,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "sign": np.sign,
}

#: Names that cannot be shadowed by a case parameter.
RESERVED_NAMES: frozenset[str] = frozenset(
    {*COORDINATE_NAMES, "t", *BUILTIN_CONSTANTS, *_FUNCTIONS}
)

_BINARY = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
    ast.FloorDiv: np.floor_divide,
}
_UNARY = {ast.UAdd: np.positive, ast.USub: np.negative}
_COMPARE = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


class ExpressionError(LoadError):
    """An expression is malformed, uses a forbidden construct or name."""


def _parse(text: str) -> ast.Expression:
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("an expression needs a non-empty formula")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"cannot parse {text!r}: {error.msg}") from None
    _validate(tree.body, text)
    return tree


def _validate(node: ast.AST, text: str) -> None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"{text!r}: only numbers are allowed as literals")
    elif isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        _validate(node.left, text)
        _validate(node.right, text)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        _validate(node.operand, text)
    elif isinstance(node, ast.Compare) and all(
        type(op) in _COMPARE for op in node.ops
    ):
        for child in (node.left, *node.comparators):
            _validate(child, text)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ExpressionError(
                f"{text!r}: only these functions are allowed: "
                f"{', '.join(sorted(_FUNCTIONS))}"
            )
        if node.keywords:
            raise ExpressionError(f"{text!r}: keyword arguments are not allowed")
        for argument in node.args:
            _validate(argument, text)
    else:
        raise ExpressionError(
            f"{text!r}: unsupported construct {type(node).__name__}"
        )


def check_expression(text: str) -> None:
    """Refuse a malformed or non-whitelisted expression."""

    _parse(text)


def expression_names(text: str) -> frozenset[str]:
    """The free variable names a formula reads (functions excluded)."""

    tree = _parse(text)
    functions = {
        id(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    return frozenset(
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and id(node) not in functions
    )


def evaluate_expression(
    text: str,
    variables: Mapping[str, "float | np.ndarray"],
    *,
    size: int | None = None,
) -> np.ndarray:
    """Evaluate ``text`` with ``variables``; returns a float array.

    ``variables`` are looked up before the built-in constants, but may not
    redefine a reserved name that is a function.  With ``size`` the result is
    broadcast to that many values, so a constant formula still yields one value
    per element.  An unknown name or a non-finite result is an error.
    """

    tree = _parse(text)
    scope: Dict[str, "float | np.ndarray"] = {**BUILTIN_CONSTANTS, **variables}

    def visit(node: ast.AST):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            try:
                return scope[node.id]
            except KeyError:
                available = sorted(set(scope) | set(_FUNCTIONS))
                raise ExpressionError(
                    f"{text!r}: unknown name {node.id!r}; known names are "
                    f"{', '.join(available)}"
                ) from None
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Compare):
            result = None
            left = visit(node.left)
            for operator, comparator in zip(node.ops, node.comparators):
                right = visit(comparator)
                part = _COMPARE[type(operator)](left, right)
                result = part if result is None else np.logical_and(result, part)
                left = right
            return result
        if isinstance(node, ast.Call):
            return _FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        raise ExpressionError(f"{text!r}: unsupported construct")  # pragma: no cover

    with np.errstate(all="ignore"):
        value = np.asarray(visit(tree.body), dtype=float)
    if size is not None:
        value = np.broadcast_to(value, (size,)).copy()
    if not np.all(np.isfinite(value)):
        raise ExpressionError(
            f"{text!r} produced a non-finite value (check divisions, "
            "square roots and logarithms)"
        )
    return value
