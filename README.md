# ANYloads

Load definitions and their evaluation, independent of any solver, geometry or
mesh. ANYloads owns how a load is *defined* and *evaluated*; applications
(ANYfem, ANYworkspaceAI) attach definitions to their own models and hand over
points, so the same definition gives the same pressure everywhere.

```python
import numpy as np
from anyloads import CaseContext, PressureField, PressureTable, trust_code

case = CaseContext(number=2, name="storm", time=3.0, parameters={"draft": 2.0})
points = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.5], [0.0, 0.0, 2.5]])  # metres

# An equation
PressureField(expression="1025 * g * max(0, draft - z)").evaluate(case, points)

# Your own code (runs only if trusted; see below)
code = """
def pressure(case, x, y, z):
    return 1025 * 9.81 * max(0.0, 2.0 + 0.1 * case - z)
"""
trust_code(code)
PressureField(code=code).evaluate(case, points)

# A table: rows of  case, x, y, z, pressure  (CSV, TSV, text, .xlsx)
table = PressureTable.from_file("loads.csv", key="time", interpolate_time=True)
PressureField(table=table).evaluate(case, points)
```

A field returns a pressure *magnitude*; `fluid_sign(...)` turns it into a
pressure signed along a surface normal for a given fluid side.

## Equations

A formula of `x, y, z` (metres), the case time `t`, the case's named parameters
and `g`, `pi`, `e`, with `abs sqrt exp log sin cos tan tanh sinh cosh min max
clip where sign`. It is parsed, never run with `eval`.

## Code and trust

`def pressure(case, x, y, z)` may also declare `t`, `name` and `parameters`; it is
called per point, or once with arrays when `code_vectorized`. **Code is never run
because a file says so.** Parsing only checks it. It runs only when its SHA-256
is in the per-user trust list (`trusted_code.json` in the user config directory,
or the file in `ANYLOADS_TRUSTED_CODE_FILE`), which lives outside any project.
Typing code into an application trusts it; code that arrives in a file does not
until the user approves it (`trust_code`). Trust is per exact text. Trusting code
means it runs with your account's full access.

## Tables

Rows `(case,) x, y, z, pressure`, with a header or without (`[key,] x, y, z,
pressure`). `key` is `"number"`, `"time"` (optionally `interpolate_time`) or
`"name"`, or `None` for one map. Points map by `"nearest"` or `"idw"`;
`max_distance` refuses a point with nothing near it; `length_scale` converts
coordinates to metres and `PressureField.scale` converts pressure to pascals. The
path and SHA-256 are recorded and a changed file is refused. Excel needs
`pip install ANYloads[excel]`.

## Load case number

`CaseContext.number` is 1, 2, 3 ...; the application decides numbering (ANYfem
numbers cases in creation order and a series step as step + 1).

Licensed under MPL-2.0.
