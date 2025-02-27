"""The set of problems (identified by api + commit hash) to be included in the collected benchmark"""

PANDAS_PROBLEMS = [
    # --- v0.0.1 ---
    ("pandas-index._getitem_slice", "9b4cffc"),
    ("pandas-merge_ordered", "061c2e9"),
    ("pandas-dataframegroupby.skew", "233bd83"),
    ("pandas-dataframegroupby.idxmin", "ccca5df"),
    # --- v0.0.2 ---
    ("pandas-period.strftime", "2cdca01"),
    ("pandas-groupby.quantile", "e8961f1"),
    ("pandas-seriesgroupby.ffill", "84aca21"),
    ("pandas-merge_asof", "2f4c93e"),
    ("pandas-pandas.concat", "b661313"),  # NOTE: easy to beat commit
    ("pandas-datetimeindex.isocalendar", "1ae00c6"),  # NOTE: easy to beat commit
    # --- v0.0.3 ---
    ("pandas-datetimelikearraymixin._add_datetimelike_scalar", "a6c0ae4"),
    ("pandas-rangeindex.dtype", "1e03419"),
    ("pandas-pandas.isna", "9097263"),
    ("pandas-block.make_block", "15fd7d7"),
    ("pandas-ensure_string_array", "2a08b05"),
    ("pandas-maybe_sequence_to_range", "bfaf917"),
    ("pandas-index.infer_dtype", "12faa2e"),
]


TEST_PROBLEMS = PANDAS_PROBLEMS
