"""The set of problems (identified by api + commit hash) to be included in the collected benchmark"""

PANDAS_PROBLEMS = [
    # --- v0.0.1 ---
    ("pandas-merge_ordered", "061c2e9"),
    ("pandas-dataframegroupby.skew", "233bd83"),
    ("pandas-dataframegroupby.idxmin", "ccca5df"),
    ("pandas-period.strftime", "2cdca01"),
    ("pandas-groupby.quantile", "e8961f1"),
    ("pandas-seriesgroupby.ffill", "84aca21"),
    ("pandas-merge_asof", "2f4c93e"),
    ("pandas-rangeindex.dtype", "1e03419"),
    ("pandas-pandas.isna", "9097263"),
    ("pandas-block.make_block", "15fd7d7"),
    ("pandas-ensure_string_array", "2a08b05"),
    ("pandas-maybe_sequence_to_range", "bfaf917"),
    ("pandas-index.infer_dtype", "12faa2e"),
    ("pandas-multiindex.argsort", "609c3b7"),
    ("pandas-dataframe.__setitem__", "e7e3676"),
    ("pandas-rangeindex.take", "fd43d4b"),
    ("pandas-dataframe.apply", "5555c51"),
    ("pandas-multiindex.get_locs", "2278923"),
    ("pandas-series.loc", "6466fc6"),
    ("pandas-dataframe.transpose", "f1211e7"),
    ("pandas-dataframe.round", "f298507"),
    ("pandas-multiindex.intersection", "438b957"),
    ("pandas-arrays.integerarray.dtype", "37e9e06"),
    ("pandas-dataframegroupby.nunique", "d377cc9"),
    ("pandas-basemaskedarray._validate_setitem_value", "71c94af"),
    ("pandas-basemaskedarray.reshape", "fa29f09"),
    ("pandas-dataframe", "9a6c8f0"),  # weak test?
    ("pandas-index.union", "c34da50"),
    ("pandas-pandas.testing.assert_frame_equal", "32ebcfc"),
    ("pandas-multiindex.symmetric_difference", "c6cf37a"),
    # --- v0.0.3 ---
    ("pandas-merge_asof", "ad3f3f7"),
    ("pandas-to_datetime", "2421931"),
    ("pandas-pandas.testing.assert_frame_equal", "00d88e9"),
    ("pandas-datetimeindex.tz_localize", "b731518"),  # weak test?
    ("pandas-indexengine.get_indexer_non_unique", "5d82d8b"),  # internal test
    ("pandas-dataframe.last_valid_index", "65bca65"),  # sc, hl
    ("pandas-merge", "81b5f1d"),  # sc, hl
    ("pandas-multiindex.get_locs", "9d6d587"),  # sl
    ("pandas-dataframe.duplicated", "235113e"),
    ("pandas-dataframe.all", "28f4942"),
    ("pandas-merge", "c51c2a7"),
    ("pandas-concat", "1c2ad16"),
    ("pandas-series.__init__", "191557d"),  # single test!
    ("pandas-datetimelikearraymixin.astype", "45f0705"),  # single test!
    # --- buffer ---
    ("pandas-concat", "4583a04"),
    # ("pandas-index._getitem_slice", "9b4cffc"),
    # ("pandas-multiindex.equals", "7281475"),
    # ("pandas-datetimelikearraymixin._add_datetimelike_scalar", "a6c0ae4"),
    # ("pandas-ensure_index_from_sequences", "38086f1"),  # single test!
]

NUMPY_PROBLEMS = [
    # --- v0.0.1 ---
    ("numpy-np.char.rfind", "22ab9aa"),
    ("numpy-numpy.char.rstrip", "728fedc"),
    ("numpy-numpy.strings.ljust", "cb0d7cd"),
    ("numpy-numpy.char.startswith", "ee75c87"),
    ("numpy-numpy.char.multiply", "567b57d"),
    ("numpy-numpy.char.isnumeric", "893db31"),
    ("numpy-numpy.add.at", "ba89ef9"),
    ("numpy-numpy.core.umath.log", "2dfd21e"),
    ("numpy-numpy.arctan2", "5f94eb8"),
    ("numpy-numpy.exp", "8dd6761"),
    ("numpy-numpy.subtract", "be52f19"),
    ("numpy-numpy.power", "e0194de"),
    ("numpy-numpy.sum", "330057f"),
    ("numpy-np.minimum.at", "11a7e2d"),
    ("numpy-np.partition", "ac5c664"),
    ("numpy-np.divide.at", "28706af"),
    ("numpy-numpy.repeat", "905d37e"),
    ("numpy-numpy.lib.recfunctions.structured_to_unstructured", "2540554"),
    ("numpy-numpy.can_cast", "8535df6"),
    ("numpy-numpy.ndarray.flat", "ec52363"),
    ("numpy-numpy.where", "780799b"),
    ("numpy-numpy.choose", "68eead8"),
    ("numpy-array_equal", "7ff7ec7"),
    # --- v0.0.2 ---
    ("numpy-np.add", "b862e4f"),
    ("numpy-np.zeros", "382b3ff"),
    ("numpy-np.char.find", "83c780d"),
    ("numpy-numpy.char.strip", "cb461ba"),
    ("numpy-np.char.isalpha", "ef5e545"),
    ("numpy-np.char.add", "19bfa3f"),
    ("numpy-numpy.char.endswith", "09db9c7"),
    ("numpy-np.add.at", "7853cbc"),
    ("numpy-numpy.char.replace", "1b861a2"),
    ("numpy-flatiter.__getitem__", "f69f273"),
    ("numpy-numpy.ufunc.at", "eb21b25"),
    ("numpy-numpy.char.isdecimal", "248c60e"),
    ("numpy-numpy.char.count", "e801e7a"),
    ("numpy-numpy.finfo", "4aca866"),  # sc, hl
    ("numpy-np.isin", "cedba62"),  # related to np.in1d
    ("numpy-numpy.log", "4973914"),  # enable SVML?
    ("numpy-numpy.greater", "ec8d5db"),  # low speedup?
    ("numpy-numpy.vecdot", "1fcda82"),  # enable BLAS?
    ("numpy-np.sort", "794f474"),  # enable AVX2/SIMD?
    # --- buffer ---
    ("numpy-maskedarray.clip", "6d77c59"),  # simple? low speedup?
    # ("numpy-numpy.add.reduce", "3ea71da"), # flaky speedup
    # ("numpy-numpy.conjugate", "d352270"),  # flaky speedup
    # ("numpy-numpy.equal", "1134be2"),
    # ("numpy-np.ma.cov", "9b1a9c5"),
    # ("numpy-numpy.kron", "54605f3"),
    # ("numpy-np.in1d", "91e753c"),
]

PILLOW_PROBLEMS = [
    ("pillow-image.split", "d8af3fc"),
    ("pillow-putchunk", "4bc33d3"),
    ("pillow-imaginggetbbox", "63f398b"),
    ("pillow-imagecolor.getcolor", "99760f4"),
    ("pillow-gifimagefile.n_frames", "f854676"),
    ("pillow-image.rotate", "929c561"),
    ("pillow-imagefilter.kernel", "3b5c2c3"),
    ("pillow-tiffimagefile.is_animated", "fd8ee84"),
    # --- buffer ---
    ("pillow-imagefilter.color3dlut", "2b09e7f"),  # simple? low speedup?
    ("pillow-image.new", "c82f9fe"),  # single test!
]

DATASETS_PROBLEMS = [
    ("datasets-load_dataset_builder", "ef3b5dd"),
    ("datasets-dataset.select_columns", "32b206d"),
    ("datasets-iterabledataset.skip", "c5464b3"),
    ("datasets-iterabledataset.filter", "ef2fb35"),
    ("datasets-dataset._select_contiguous", "5994036"),  # single test
    ("datasets-datasetbuilder.download_and_prepare", "2878019"),  # single test
    ("datasets-image.cast_storage", "d9a8d8a"),  # single test
]

TORNADO_PROBLEMS = [
    ("tornado-parse_cookie", "0a39ba8"),  # long running!
    ("tornado-tcpserver.__init__", "bf1b21a"),
    ("tornado-tornado.websocket.websocketclientconnection.write_message", "9a18f6c"),
    ("tornado-baseiostream.write", "1b464c4"),
    ("tornado-addthreadselectoreventloop.remove_reader", "5cfe2fc"),
    ("tornado-http1serverconnection.close", "715ef05"),
    ("tornado-future.set_exception", "4d4c1e0"),
    ("tornado-future.done", "ac13ee5"),  # common PR w/ set_exception
    # --- buffer ---
    ("tornado-oauth2mixin.oauth2_request", "0ab8263"),  # simple? low speedup?
    # ("tornado-resolver.resolve", "bc74d7b"), # flaky speedup
]

PYDANTIC_PROBLEMS = [
    ("pydantic-basemodel.__setattr__", "addf1f9"),
    ("pydantic-genericmodel.__concrete_name__", "4a09447"),
    ("pydantic-basemodel._iter", "7d9614e"),  # long running!
    ("pydantic-typeadapter.validate_strings", "c2647ab"),
    ("pydantic-typeadapter.validate_python", "ac9e6ee"),
    ("pydantic-typeadapter.__init__", "c5dce37"),
]


TEST_PROBLEMS = {
    "pandas": PANDAS_PROBLEMS,
    "numpy": NUMPY_PROBLEMS,
    "pillow": PILLOW_PROBLEMS,
    "datasets": DATASETS_PROBLEMS,
    "tornado": TORNADO_PROBLEMS,
    "pydantic": PYDANTIC_PROBLEMS,
}


# add problems here that have long runtimes to reduce number of tests
LONG_RUNNING_PROBLEMS = [
    ("pandas-dataframe.__setitem__", "e7e3676"),
    ("datasets-load_dataset_builder", "ef3b5dd"),
    ("datasets-dataset.select_columns", "32b206d"),
]
