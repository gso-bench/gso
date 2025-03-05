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

NUMPY_PROBLEMS = [
    # --- string manipulation ---
    ("numpy-np.char.rfind", "22ab9aa"),
    ("numpy-numpy.char.rstrip", "728fedc"),
    ("numpy-numpy.strings.ljust", "cb0d7cd"),
    ("numpy-numpy.char.startswith", "ee75c87"),
    ("numpy-numpy.char.multiply", "567b57d"),
    ("numpy-numpy.char.endswith", "09db9c7"),  # NOTE: sub commit for ee75c87?
    ("numpy-numpy.char.isnumeric", "893db31"),
    # --- math operations ---
    ("numpy-numpy.equal", "1134be2"),
    ("numpy-numpy.add.at", "ba89ef9"),
    ("numpy-numpy.core.umath.log", "2dfd21e"),
    ("numpy-numpy.arctan2", "5f94eb8"),
    ("numpy-numpy.exp", "8dd6761"),
    ("numpy-np.ma.cov", "9b1a9c5"),
    ("numpy-numpy.subtract", "be52f19"),
    ("numpy-numpy.power", "e0194de"),
    ("numpy-numpy.sum", "330057f"),
    ("numpy-numpy.kron", "54605f3"),
    # --- array manipulation ---
    ("numpy-np.minimum.at", "11a7e2d"),
    ("numpy-np.partition", "ac5c664"),
    ("numpy-maskedarray.clip", "6d77c59"),
    ("numpy-np.divide.at", "28706af"),
    ("numpy-numpy.repeat", "905d37e"),
    ("numpy-numpy.lib.recfunctions.structured_to_unstructured", "2540554"),
    ("numpy-numpy.can_cast", "8535df6"),
    ("numpy-numpy.ndarray.flat", "ec52363"),
    ("numpy-np.in1d", "91e753c"),  # NOTE: simple?
    ("numpy-numpy.where", "780799b"),  # NOTE: simple?
    ("numpy-numpy.choose", "68eead8"),  # NOTE: simple?
    ("numpy-array_equal", "7ff7ec7"),
]


TEST_PROBLEMS = {
    "pandas": PANDAS_PROBLEMS,
    "numpy": NUMPY_PROBLEMS,
}
