"""The set of problems (identified by api + commit hash) to be included in the collected benchmark"""

PANDAS_PROBLEMS = [
    # --- v0.0.1 ---
    ("pandas-index._getitem_slice", "9b4cffc"),
    ("pandas-merge_ordered", "061c2e9"),
    ("pandas-dataframegroupby.skew", "233bd83"),
    ("pandas-dataframegroupby.idxmin", "ccca5df"),
    ("pandas-period.strftime", "2cdca01"),
    ("pandas-groupby.quantile", "e8961f1"),
    ("pandas-seriesgroupby.ffill", "84aca21"),
    ("pandas-merge_asof", "2f4c93e"),
    ("pandas-pandas.concat", "b661313"),  # NOTE: commit < base?: remove
    ("pandas-datetimeindex.isocalendar", "1ae00c6"),  # NOTE: commit < base?: remove
    # --- v0.0.2 ---
    ("pandas-datetimelikearraymixin._add_datetimelike_scalar", "a6c0ae4"),
    ("pandas-rangeindex.dtype", "1e03419"),
    ("pandas-pandas.isna", "9097263"),
    ("pandas-block.make_block", "15fd7d7"),
    ("pandas-ensure_string_array", "2a08b05"),  # NOTE: simple?
    ("pandas-maybe_sequence_to_range", "bfaf917"),
    ("pandas-index.infer_dtype", "12faa2e"),
    ("pandas-multiindex.argsort", "609c3b7"),
    ("pandas-dataframe.__setitem__", "e7e3676"),
    ("pandas-rangeindex.take", "fd43d4b"),  # NOTE: use (i)loc instead?
    ("pandas-dataframe.apply", "5555c51"),
    ("pandas-multiindex.get_locs", "2278923"),
    ("pandas-series.loc", "6466fc6"),
    ("pandas-dataframe.transpose", "f1211e7"),
    ("pandas-dataframe.round", "f298507"),
    ("pandas-multiindex.intersection", "438b957"),
    ("pandas-arrays.integerarray.dtype", "37e9e06"),
    ("pandas-dataframegroupby.nunique", "d377cc9"),
    ("pandas-concat", "4583a04"),
    ("pandas-multiindex.equals", "7281475"),
    ("pandas-basemaskedarray._validate_setitem_value", "71c94af"),
    ("pandas-basemaskedarray.reshape", "fa29f09"),
    ("pandas-merge_asof", "ad3f3f7"),
    ("pandas-dataframe", "9a6c8f0"),
]

NUMPY_PROBLEMS = [
    # --- string manipulation ---
    ("numpy-np.char.rfind", "22ab9aa"),
    ("numpy-numpy.char.rstrip", "728fedc"),
    ("numpy-numpy.strings.ljust", "cb0d7cd"),
    ("numpy-numpy.char.startswith", "ee75c87"),
    ("numpy-numpy.char.multiply", "567b57d"),
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
