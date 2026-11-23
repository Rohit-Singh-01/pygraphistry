"""probe 2 (not for review): cuDF column-wise prefix/suffix via slice_from with series bounds."""
import os

import pandas as pd
import pytest


def test_cudf_slice_from_prefix_suffix():
    cudf = pytest.importorskip("cudf")
    if os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cudf")
    left = cudf.Series(["abc", "xyz", None, "a.c", "abcabc", "", "ab", "b"])
    right = cudf.Series(["a", "z", "q", ".", "bc", "", "abc", None])
    outs = []
    try:
        ll, lr = left.str.len().fillna(0), right.str.len().fillna(0)
        zeros = ll * 0
        prefix = left.str.slice_from(starts=zeros, stops=lr)
        pre = (prefix == right) & (lr <= ll)
        outs.append("PREFIX dtype=%s %s" % (pre.dtype, pre.to_pandas().tolist()))
        starts = (ll - lr).clip(lower=0)
        suffix = left.str.slice_from(starts=starts, stops=ll)
        suf = (suffix == right) & (lr <= ll)
        outs.append("SUFFIX dtype=%s %s" % (suf.dtype, suf.to_pandas().tolist()))
    except Exception as exc:  # noqa
        outs.append("ERR slice_from %s: %s" % (type(exc).__name__, str(exc)[:160].replace("\n", " ")))
    try:
        c = left.str.contains(right, regex=False)
        outs.append("CONTAINS %s %s" % (c.dtype, c.to_pandas().tolist()))
    except Exception as exc:  # noqa
        outs.append("ERR contains %s" % str(exc)[:100])
    assert False, "SLICEPROBE|" + " || ".join(outs)
