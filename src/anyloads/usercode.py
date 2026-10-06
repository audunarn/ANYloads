"""User-written Python for field-valued loads, and the trust that gates it.

A pressure may be defined by a Python function::

    def pressure(case, x, y, z):
        return 1025 * 9.81 * max(0.0, 2.0 - z) * (1 + 0.1 * case)

Unlike an equation (:mod:`anyloads.expressions`), this is real code with full Python
power, so a project file must never be able to run it by itself. The rules:

* Creating or loading a project only *parses* the code (``ast``); nothing runs.
* Code runs only when its SHA-256 is in the per-user trust store, a file kept
  outside the project (``platformdirs`` user config, or the file named by the
  ``ANYLOADS_TRUSTED_CODE_FILE`` environment variable). A file received from
  someone else is therefore inert until its code is explicitly trusted.
* Typing code into an application (ANYfem's add/edit commands, ``add_wet_pressure(code=...)``)
  trusts it: the author is the user. Trust is per exact text, so any edit
  needs trusting again.

The function may declare any of ``case`` (the load case number), ``x``, ``y``,
``z`` (element centroid, metres), ``t`` (case time or ``None``), ``name`` (the
case name) and ``parameters`` (the case's named numbers); only declared names
are passed. ``numpy`` is available as ``np`` and ``math`` as ``math``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import numpy as np

from .errors import LoadError

__all__ = [
    "ALLOWED_ARGUMENTS",
    "FUNCTION_NAME",
    "UntrustedCodeError",
    "UserCodeError",
    "check_code",
    "code_hash",
    "is_trusted",
    "run_pressure_code",
    "trust_code",
    "trust_store_path",
]

FUNCTION_NAME = "pressure"
ALLOWED_ARGUMENTS: tuple[str, ...] = (
    "case", "x", "y", "z", "t", "name", "parameters",
)
_FILENAME = "<wet pressure code>"


class UserCodeError(LoadError):
    """User code is malformed or failed while running."""


class UntrustedCodeError(UserCodeError):
    """User code has not been approved by this user on this machine."""


def code_hash(code: str) -> str:
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


def check_code(code: str) -> tuple[str, ...]:
    """Parse ``code`` without running it; return the arguments it declares.

    Raises :class:`UserCodeError` unless it is valid Python defining a
    top-level ``pressure`` function with only the supported arguments.
    """

    if not isinstance(code, str) or not code.strip():
        raise UserCodeError("pressure code is empty")
    try:
        tree = ast.parse(code, filename=_FILENAME)
    except SyntaxError as error:
        raise UserCodeError(
            f"pressure code, line {error.lineno}: {error.msg}"
        ) from None
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == FUNCTION_NAME
        ),
        None,
    )
    if function is None:
        raise UserCodeError(
            f"pressure code must define 'def {FUNCTION_NAME}(case, x, y, z): ...'"
        )
    arguments = function.args
    if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        raise UserCodeError(
            f"{FUNCTION_NAME}() may only take plain positional arguments"
        )
    if arguments.defaults:
        raise UserCodeError(f"{FUNCTION_NAME}() arguments cannot have defaults")
    names = tuple(argument.arg for argument in arguments.posonlyargs + arguments.args)
    unknown = [name for name in names if name not in ALLOWED_ARGUMENTS]
    if unknown:
        raise UserCodeError(
            f"{FUNCTION_NAME}() has unsupported argument(s) {unknown}; "
            f"available: {', '.join(ALLOWED_ARGUMENTS)}"
        )
    return names


# ----------------------------------------------------------------------
# trust
# ----------------------------------------------------------------------
def trust_store_path() -> Path:
    override = os.environ.get("ANYLOADS_TRUSTED_CODE_FILE")
    if override:
        return Path(override)
    from platformdirs import user_config_dir

    return Path(user_config_dir("ANYloads", appauthor=False)) / "trusted_code.json"


def _read_store() -> List[str]:
    path = trust_store_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    hashes = data.get("sha256", []) if isinstance(data, dict) else []
    return [str(item) for item in hashes if isinstance(item, str)]


def is_trusted(code: str) -> bool:
    return code_hash(code) in set(_read_store())


def trust_code(code: str) -> str:
    """Approve exactly this text on this machine; returns its hash."""

    check_code(code)
    digest = code_hash(code)
    hashes = _read_store()
    if digest not in hashes:
        hashes.append(digest)
        path = trust_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump({"sha256": hashes}, stream, indent=1)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    return digest


# ----------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------
def _failure(error: BaseException, what: str) -> UserCodeError:
    line = None
    for frame in reversed(traceback.extract_tb(error.__traceback__)):
        if frame.filename == _FILENAME:
            line = frame.lineno
            break
    where = f", line {line}" if line is not None else ""
    return UserCodeError(
        f"{what}{where}: {type(error).__name__}: {error}"
    )


def run_pressure_code(
    code: str,
    *,
    case: int,
    name: str,
    time: float | None,
    parameters: Mapping[str, float],
    centroids: np.ndarray,
    vectorized: bool = False,
) -> np.ndarray:
    """Evaluate trusted user code at each centroid; returns one value each.

    By default the function is called once per element with scalars; with
    ``vectorized`` it is called once with numpy arrays ``x, y, z`` and must
    return an array (or a scalar) for all of them.
    """

    declared = check_code(code)
    if not is_trusted(code):
        raise UntrustedCodeError(
            "this pressure code has not been trusted on this machine "
            f"(sha256 {code_hash(code)[:12]}...). Review it, then approve it "
            "with anyloads.trust_code(code) or in the GUI."
        )
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 3)
    namespace: Dict[str, Any] = {
        "__name__": "anyloads_pressure_code",
        "np": np,
        "numpy": np,
        "math": math,
    }
    try:
        exec(compile(code, _FILENAME, "exec"), namespace)  # noqa: S102 - trusted
    except Exception as error:  # noqa: BLE001 - reported with its line
        raise _failure(error, "pressure code failed while loading") from None
    function: Callable[..., Any] = namespace[FUNCTION_NAME]
    fixed = {
        "case": int(case),
        "t": None if time is None else float(time),
        "name": str(name),
        "parameters": dict(parameters),
    }

    def arguments(x, y, z) -> List[Any]:
        values = {**fixed, "x": x, "y": y, "z": z}
        return [values[argument] for argument in declared]

    count = len(centroids)
    try:
        if vectorized:
            result = np.asarray(
                function(
                    *arguments(centroids[:, 0], centroids[:, 1], centroids[:, 2])
                ),
                dtype=float,
            )
            values = np.broadcast_to(result, (count,)).copy()
        else:
            values = np.empty(count)
            for index, (x, y, z) in enumerate(centroids):
                values[index] = float(function(*arguments(float(x), float(y), float(z))))
    except UserCodeError:
        raise
    except Exception as error:  # noqa: BLE001 - reported with its line
        raise _failure(error, f"{FUNCTION_NAME}() failed") from None
    if not np.all(np.isfinite(values)):
        bad = int(np.argmax(~np.isfinite(values)))
        raise UserCodeError(
            f"{FUNCTION_NAME}() returned a non-finite pressure "
            f"(first at x={centroids[bad, 0]:g}, y={centroids[bad, 1]:g}, "
            f"z={centroids[bad, 2]:g})"
        )
    return values
