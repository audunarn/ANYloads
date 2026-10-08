# Changelog

## 0.1.0 (unreleased)

First version. Moved from ANYfem, where it was written, so that ANYfem and
ANYworkspaceAI evaluate a load definition identically:

- `Envelope`: when a load acts and how strongly over a time history. A window
  (`start`, `stop`, both inclusive and optional, counted in load case numbers or in
  case time) and a factor inside it (an expression of `t`, `number`, `k`, `tau` and the
  case parameters, or piecewise-linear `(time, factor)` points held at the ends).
  `factor(case)` is 0 outside the window; `to_dict`/`from_dict` round-trip it.
  Applications attach one to any load, so one stored load can cover a 600-step record.
- `evaluate_expression`: a parsed, whitelisted formula of position, time and
  named parameters; never `eval`.
- `run_pressure_code` and the trust list: user Python `def pressure(case, x, y,
  z)` that runs only when its SHA-256 is trusted on this machine; parsing never
  runs it.
- `PressureTable`: `case, x, y, z, pressure` rows from CSV/TSV/text/.xlsx,
  selected by case number, time (optionally interpolated) or name, mapped to
  points by nearest point or inverse-distance weighting, with a SHA-256 so a
  changed file is refused.
- `PressureField` and `CaseContext`: one definition (value, expression, code or
  table) evaluated for a case at points, with plain-data (de)serialization.
- `fluid_sign`: which way fluid pressure acts for a given fluid side.
