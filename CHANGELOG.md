# Changelog

## 0.1.0 (unreleased)

First version. Moved from ANYfem, where it was written, so that ANYfem and
ANYworkspaceAI evaluate a load definition identically:

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
