from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anyloads import (
    PressureTable,
    PressureTableError,
    interpolate_table,
    read_table,
)


def grid():
    xs = np.arange(0.25, 2.0, 0.5)
    zs = np.arange(0.25, 3.0, 0.5)
    return np.array([(x, 0.0, z) for x in xs for z in zs])


def write(path, rows, header="case,x,y,z,pressure", delimiter=","):
    lines = [header.replace(",", delimiter)] if header else []
    for row in rows:
        lines.append(
            delimiter.join(v if isinstance(v, str) else f"{v:.12g}" for v in row)
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def rows_at(key, fn):
    return [(key, *point, fn(point)) for point in grid()]


def test_matching_points_are_read_back_exactly(tmp_path):
    plane = lambda p: 1000 + 100 * p[0] + 2000 * p[2]
    table = PressureTable.from_file(write(tmp_path / "p.csv", rows_at(1, plane)), key="number")
    values = interpolate_table(table, 1, grid())
    assert values == pytest.approx([plane(p) for p in grid()])


def test_each_case_takes_its_own_rows_by_number(tmp_path):
    rows = rows_at(1, lambda p: 1000.0) + rows_at(2, lambda p: 5000.0)
    table = PressureTable.from_file(write(tmp_path / "p.csv", rows), key="number")
    assert interpolate_table(table, 2, grid()) == pytest.approx(5000.0)
    with pytest.raises(PressureTableError, match="no rows for number 3"):
        interpolate_table(table, 3, grid())
    with pytest.raises(PressureTableError, match="no number to look up"):
        interpolate_table(table, None, grid())


def test_time_keyed_tables_can_interpolate_between_times(tmp_path):
    rows = rows_at(0.0, lambda p: 1000.0) + rows_at(2.0, lambda p: 3000.0)
    path = write(tmp_path / "p.csv", rows)
    exact = PressureTable.from_file(path, key="time")
    smooth = PressureTable.from_file(path, key="time", interpolate_time=True)
    assert interpolate_table(exact, 2.0, grid()) == pytest.approx(3000.0)
    assert interpolate_table(smooth, 0.5, grid()) == pytest.approx(1500.0)
    with pytest.raises(PressureTableError, match=r"no rows for time 0.5.*interpolate_time"):
        interpolate_table(exact, 0.5, grid())
    with pytest.raises(PressureTableError, match="outside the table's times"):
        interpolate_table(smooth, 9.0, grid())
    assert exact.keys() == [0.0, 2.0]


def test_name_keyed_tables_and_a_single_map(tmp_path):
    rows = rows_at("calm", lambda p: 100.0) + rows_at("storm", lambda p: 900.0)
    named = PressureTable.from_file(write(tmp_path / "n.csv", rows), key="name")
    assert interpolate_table(named, "storm", grid()) == pytest.approx(900.0)
    with pytest.raises(PressureTableError, match=r"no rows for case 'other'.*calm"):
        interpolate_table(named, "other", grid())
    plain = write(tmp_path / "one.csv", [(*p, 700.0) for p in grid()], header="x,y,z,pressure")
    assert interpolate_table(PressureTable.from_file(plain), None, grid()) == pytest.approx(700.0)


def test_headerless_semicolon_tab_files_and_odd_column_names(tmp_path):
    rows = [(1, *p, 800.0) for p in grid()]
    files = [
        write(tmp_path / "h.csv", rows, header=None),
        write(tmp_path / "s.csv", rows, delimiter=";"),
        write(tmp_path / "t.tsv", rows, delimiter="\t"),
    ]
    for path in files:
        table = PressureTable.from_file(path, key="number")
        assert interpolate_table(table, 1, grid()) == pytest.approx(800.0)
    millimetres = write(
        tmp_path / "mm.csv",
        [("A", *(p * 1000 for p in point), 800.0) for point in grid()],
        header="Run,X [mm],Y [mm],Z [mm],P [kPa]",
    )
    odd = PressureTable.from_file(
        millimetres, key="name",
        columns={"key": "Run", "x": "X [mm]", "y": "Y [mm]", "z": "Z [mm]", "pressure": "P [kPa]"},
        length_scale=1e-3,
    )
    assert interpolate_table(odd, "A", grid()) == pytest.approx(800.0)


def test_an_excel_sheet_is_read(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "notes"
    sheet = workbook.create_sheet("loads")
    sheet.append(["time", "x", "y", "z", "pressure"])
    for point in grid():
        sheet.append([1.5, *map(float, point), 2500.0])
    path = tmp_path / "p.xlsx"
    workbook.save(path)
    table = PressureTable.from_file(path, key="time", sheet="loads")
    assert interpolate_table(table, 1.5, grid()) == pytest.approx(2500.0)
    with pytest.raises(PressureTableError, match="no sheet 'nope'"):
        PressureTable.from_file(path, key="time", sheet="nope")


def test_inverse_distance_weighting_blends_neighbours_and_hits_points_exactly(tmp_path):
    a, b = (0.25, 0.0, 0.25), (1.75, 0.0, 2.75)
    path = write(tmp_path / "p.csv", [(*a, 1000.0), (*b, 5000.0)], header="x,y,z,pressure")
    table = PressureTable.from_file(path, method="idw", neighbours=2)
    values = interpolate_table(table, None, np.array([a, b, (1.0, 0.0, 1.5)]))
    assert values[0] == pytest.approx(1000.0, rel=1e-6)
    assert values[1] == pytest.approx(5000.0, rel=1e-6)
    assert 1000.0 < values[2] < 5000.0


def test_max_distance_refuses_points_far_from_the_table(tmp_path):
    path = write(tmp_path / "p.csv", [(0.25, 0.0, 0.25, 1000.0)], header="x,y,z,pressure")
    table = PressureTable.from_file(path, max_distance=0.6)
    with pytest.raises(PressureTableError, match=r"no table point within 0.6 m"):
        interpolate_table(table, None, grid())


def test_a_changed_or_missing_file_is_refused(tmp_path):
    path = write(tmp_path / "p.csv", rows_at(1, lambda p: 1000.0))
    table = PressureTable.from_file(path, key="number")
    first = read_table(table)
    assert read_table(table) is first  # cached
    write(path, rows_at(1, lambda p: 9999.0))
    with pytest.raises(PressureTableError, match="has changed since it was added"):
        read_table(table)
    path.unlink()
    with pytest.raises(PressureTableError, match="does not exist"):
        read_table(table)


def test_bad_contents_are_reported_with_the_row(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("case,x,y,z,pressure\n1,0,0,0,abc\n", encoding="utf-8")
    with pytest.raises(PressureTableError, match=r"row 2.*finite numbers"):
        PressureTable.from_file(bad, key="number")
    short = tmp_path / "short.csv"
    short.write_text("x,y,z\n1,2,3\n", encoding="utf-8")
    with pytest.raises(PressureTableError, match="cannot find column"):
        PressureTable.from_file(short)
    with pytest.raises(PressureTableError):
        PressureTable.from_file(tmp_path / "missing.csv")
    with pytest.raises(PressureTableError, match="table key"):
        PressureTable(path="x", sha256="0" * 64, key="step")
    with pytest.raises(PressureTableError, match="time-keyed"):
        PressureTable(path="x", sha256="0" * 64, key="number", interpolate_time=True)


def test_a_table_round_trips_as_plain_data_and_leaves_defaults_out(tmp_path):
    path = write(tmp_path / "p.csv", rows_at(1, lambda p: 1000.0))
    table = PressureTable.from_file(
        path, key="number", method="idw", neighbours=3, max_distance=1.5, length_scale=1.0
    )
    assert PressureTable.from_dict(table.to_dict()) == table
    assert set(PressureTable.from_file(path, key="number").to_dict()) == {"path", "sha256", "key"}
