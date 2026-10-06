"""Load definitions and their evaluation, independent of any solver or model.

ANYloads owns how a load is *defined* and *evaluated*: pressure as a number, an
equation, trusted Python code, or a CSV/Excel table of values at points per load
case, and which way fluid pressure pushes on a surface. It knows nothing of
geometry, meshes or solvers; applications (ANYfem, ANYworkspaceAI) attach these
definitions to their own models and hand them points. Because the evaluation is
here, every application computes the same pressure for the same definition.

Code never runs just because a file says so: see :mod:`anyloads.usercode`.
"""

from __future__ import annotations

from .case import CaseContext, check_parameter
from .errors import LoadError
from .expressions import (
    BUILTIN_CONSTANTS,
    COORDINATE_NAMES,
    RESERVED_NAMES,
    ExpressionError,
    check_expression,
    evaluate_expression,
    expression_names,
)
from .field import PressureField, table_selector
from .fluid import FLUID_SIDES, check_fluid_side, fluid_sign
from .tables import (
    PressureTable,
    PressureTableError,
    TableData,
    file_sha256,
    interpolate_table,
    read_table,
)
from .usercode import (
    ALLOWED_ARGUMENTS,
    FUNCTION_NAME,
    UntrustedCodeError,
    UserCodeError,
    check_code,
    code_hash,
    is_trusted,
    run_pressure_code,
    trust_code,
    trust_store_path,
)

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_ARGUMENTS",
    "BUILTIN_CONSTANTS",
    "COORDINATE_NAMES",
    "CaseContext",
    "ExpressionError",
    "FLUID_SIDES",
    "FUNCTION_NAME",
    "LoadError",
    "PressureField",
    "PressureTable",
    "PressureTableError",
    "RESERVED_NAMES",
    "TableData",
    "UntrustedCodeError",
    "UserCodeError",
    "check_code",
    "check_expression",
    "check_fluid_side",
    "check_parameter",
    "code_hash",
    "evaluate_expression",
    "expression_names",
    "file_sha256",
    "fluid_sign",
    "interpolate_table",
    "is_trusted",
    "read_table",
    "run_pressure_code",
    "table_selector",
    "trust_code",
    "trust_store_path",
    "__version__",
]
