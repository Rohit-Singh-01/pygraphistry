"""probe (not for review): cuDF / Polars / pandas column-wise string ops with a series pattern, and result dtypes."""
import os

import pandas as pd
import pytest

L = ["abc", "xyz", None, "a.c", "abcabc", 5]
R = ["a", "z", "q", ".", "bc", "a"]


def _report(tag, fn):
    try:
        out = fn()
        res = "OK %s dtype=%s %s" % (tag, getattr(out, "dtype", type(out)), list(out.to_pandas() if hasattr(out, "to_pandas") else out))
    except Exception as exc:  # noqa
        res = "ERR %s %s: %s" % (tag, type(exc).__name__, str(exc)[:100].replace("\n", " "))
    return res


def test_cudf_starts_with_series():
    cudf = pytest.importorskip("cudf")
    if os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cudf")
    left = cudf.Series(["abc", "xyz", None, "a.c", "abcabc"])
    right = cudf.Series(["a", "z", "q", ".", "bc"])
    outs = [
        _report("startswith(Series)", lambda: left.str.startswith(right)),
        _report("endswith(Series)", lambda: left.str.endswith(right)),
        _report("contains(Series,regex=False)", lambda: left.str.contains(right, regex=False)),
        _report("find(Series)", lambda: left.str.find(right)),
        _report("startswith(list)", lambda: left.str.startswith(right.to_pandas().tolist())),
        _report("count_re", lambda: left.str.count(right)),
    ]
    assert False, "CUDFPROBE|" + " || ".join(outs)


def test_cudf_literal_path_dtype():
    cudf = pytest.importorskip("cudf")
    if os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cudf")
    import graphistry
    nodes = pd.DataFrame({"id": [1, 2, 3], "name": ["abc", "xyz", None], "pre": ["a", "z", "q"]})
    res = []
    for eng, mk in (("pandas", lambda d: d), ("cudf", cudf.from_pandas)):
        g = graphistry.nodes(mk(nodes), "id")
        for q in ("MATCH (n) RETURN n.id AS i, n.name STARTS WITH 'a' AS v ORDER BY i", "MATCH (n) RETURN n.id AS i, n.name STARTS WITH n.pre AS v ORDER BY i"):
            try:
                out = g.gfql(q, engine=eng)._nodes
                res.append("%s|%s|dtype=%s|%s" % (eng, q[-40:], out["v"].dtype, list(out.to_pandas()["v"] if hasattr(out, "to_pandas") else out["v"])))
            except Exception as exc:  # noqa
                res.append("%s|%s|ERR %s: %s" % (eng, q[-40:], type(exc).__name__, str(exc)[:80]))
    assert False, "DTYPEPROBE|" + " || ".join(res)


def test_polars_expr_pattern():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": ["abc", "xyz", None, "a.c", "abcabc"], "b": ["a", "z", "q", ".", "bc"], "n": [1, 2, 3, 4, 5]})
    outs = []
    for tag, fn in (
        ("starts_with(expr)", lambda: df.select(pl.col("a").str.starts_with(pl.col("b")))),
        ("ends_with(expr)", lambda: df.select(pl.col("a").str.ends_with(pl.col("b")))),
        ("contains(expr,literal)", lambda: df.select(pl.col("a").str.contains(pl.col("b"), literal=True))),
        ("starts_with(int col)", lambda: df.select(pl.col("a").str.starts_with(pl.col("n")))),
    ):
        try:
            out = fn()
            outs.append("OK %s %s" % (tag, out.to_series().to_list()))
        except Exception as exc:  # noqa
            outs.append("ERR %s %s: %s" % (tag, type(exc).__name__, str(exc)[:80].replace("\n", " ")))
    assert False, "POLARSPROBE|polars=%s|" % pl.__version__ + " || ".join(outs)
