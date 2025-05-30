"""The set of problems (identified by api + commit hash) to be included in the collected benchmark"""

PANDAS_PROBLEMS = [
    ("pandas-merge_ordered", "061c2e9"),
    ("pandas-dataframegroupby.skew", "233bd83"),
    ("pandas-dataframegroupby.idxmin", "ccca5df"),
    ("pandas-period.strftime", "2cdca01"),
    ("pandas-groupby.quantile", "e8961f1"),
    ("pandas-seriesgroupby.ffill", "84aca21"),
    ("pandas-merge_asof", "2f4c93e"),
    ("pandas-pandas.isna", "9097263"),
    ("pandas-ensure_string_array", "2a08b05"),
    ("pandas-maybe_sequence_to_range", "bfaf917"),
    ("pandas-multiindex.argsort", "609c3b7"),
    ("pandas-dataframe.__setitem__", "e7e3676"),
    ("pandas-rangeindex.take", "fd43d4b"),
    ("pandas-multiindex.get_locs", "2278923"),
    ("pandas-dataframe.transpose", "f1211e7"),
    ("pandas-dataframe.round", "f298507"),
    ("pandas-multiindex.intersection", "438b957"),
    ("pandas-arrays.integerarray.dtype", "37e9e06"),
    ("pandas-dataframegroupby.nunique", "d377cc9"),
    ("pandas-basemaskedarray._validate_setitem_value", "71c94af"),
    ("pandas-dataframe", "9a6c8f0"),  # weak test?
    ("pandas-index.union", "c34da50"),
    ("pandas-multiindex.symmetric_difference", "c6cf37a"),
    ("pandas-merge_asof", "ad3f3f7"),
    ("pandas-to_datetime", "2421931"),
    ("pandas-indexengine.get_indexer_non_unique", "5d82d8b"),  # internal test
    ("pandas-dataframe.last_valid_index", "65bca65"),  # sc, hl
    ("pandas-merge", "81b5f1d"),  # sc, hl
    ("pandas-multiindex.get_locs", "9d6d587"),  # sl
    ("pandas-dataframe.duplicated", "235113e"),
    ("pandas-merge", "c51c2a7"),
    ("pandas-concat", "1c2ad16"),
    ("pandas-series.__init__", "191557d"),  # single test!
    ("pandas-datetimelikearraymixin.astype", "45f0705"),  # single test!
    # --- buffer ---
]

NUMPY_PROBLEMS = [
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
    ("numpy-numpy.sum", "330057f"),
    ("numpy-np.minimum.at", "11a7e2d"),
    ("numpy-np.partition", "ac5c664"),
    ("numpy-np.divide.at", "28706af"),
    ("numpy-numpy.repeat", "905d37e"),
    ("numpy-numpy.lib.recfunctions.structured_to_unstructured", "2540554"),
    ("numpy-numpy.ndarray.flat", "ec52363"),
    ("numpy-numpy.where", "780799b"),
    ("numpy-numpy.choose", "68eead8"),
    ("numpy-array_equal", "7ff7ec7"),
    ("numpy-np.add", "b862e4f"),
    ("numpy-np.zeros", "382b3ff"),
    ("numpy-np.char.find", "83c780d"),
    ("numpy-numpy.char.strip", "cb461ba"),
    ("numpy-np.char.isalpha", "ef5e545"),
    ("numpy-np.char.add", "19bfa3f"),
    ("numpy-numpy.char.endswith", "09db9c7"),
    ("numpy-np.add.at", "7853cbc"),
    ("numpy-numpy.char.replace", "1b861a2"),
    ("numpy-numpy.ufunc.at", "eb21b25"),
    ("numpy-numpy.char.isdecimal", "248c60e"),
    ("numpy-numpy.char.count", "e801e7a"),
    ("numpy-np.isin", "cedba62"),
    ("numpy-numpy.vecdot", "1fcda82"),
    ("numpy-np.sort", "794f474"),
    # --- buffer ---
]

PILLOW_PROBLEMS = [
    ("pillow-image.split", "d8af3fc"),
    ("pillow-imaginggetbbox", "63f398b"),
    ("pillow-gifimagefile.n_frames", "f854676"),
    ("pillow-tiffimagefile.is_animated", "fd8ee84"),
    # --- buffer ---
]

DATASETS_PROBLEMS = [
    ("datasets-load_dataset_builder", "ef3b5dd"),
    ("datasets-iterabledataset.skip", "c5464b3"),  # simple?
    ("datasets-dataset._select_contiguous", "5994036"),
    # --- buffer ---
]

TORNADO_PROBLEMS = [
    ("tornado-tornado.websocket.websocketclientconnection.write_message", "9a18f6c"),
    ("tornado-baseiostream.write", "1b464c4"),
    ("tornado-future.set_exception", "4d4c1e0"),
    ("tornado-future.done", "ac13ee5"),
    # --- buffer ---
]

PYDANTIC_PROBLEMS = [
    ("pydantic-basemodel.__setattr__", "addf1f9"),
    ("pydantic-genericmodel.__concrete_name__", "4a09447"),
    ("pydantic-typeadapter.validate_strings", "c2647ab"),
    ("pydantic-typeadapter.validate_python", "ac9e6ee"),
    # --- buffer ---THA
]

PILLOW_SIMD_PROBLEMS = [
    ("pillow-simd-imagingfilter", "2818b90"),
    ("pillow-simd-image.resize", "b4045cf"),
    ("pillow-simd-color3dlut.generate", "6eacce9"),
    ("pillow-simd-image.gaussian_blur", "9e60023"),
    ("pillow-simd-imagingalphacomposite", "0514e20"),
    ("pillow-simd-image.reduce", "7511039"),
    ("pillow-simd-image.reduce", "d970a39"),
    # -- buffer ---
]

TOKENIZERS_PROBLEMS = [
    ("tokenizers-normalizedstring.replace", "c893204"),
    ("tokenizers-tokenizer.encode", "076319d"),
    ("tokenizers-tokenizer.encode_batch_fast", "bfd9cde"),
    ("tokenizers-tokenizers.trainers.unigramtrainer.train", "fc76ad4"),  # long running!
    # --- buffer ---
]

TRANSFORMERS_PROBLEMS = [
    ("transformers-gptneoxdynamicntkscalingrotaryembedding", "253f9a3"),
    ("transformers-nobadwordslogitsprocessor.__call__", "63b90a5"),
    ("transformers-whispertokenizer._preprocess_token_ids", "211f93a"),
    # --- buffer ---
    ("transformers-xlnetlmheadmodel.forward", "d51b589"),  # simple, but exploratory
]

LLAMA_CPP_PROBLEMS = [
    ("llama-cpp-python-llama_cpp.gen_a", "218d361"),
    ("llama-cpp-python-llama_cpp.gen_b", "2bc1d97"),
]


TEST_PROBLEMS = {
    "pandas": PANDAS_PROBLEMS,
    "numpy": NUMPY_PROBLEMS,
    "pillow": PILLOW_PROBLEMS,
    "datasets": DATASETS_PROBLEMS,
    "tornado": TORNADO_PROBLEMS,
    "pydantic": PYDANTIC_PROBLEMS,
    "pillow-simd": PILLOW_SIMD_PROBLEMS,
    "tokenizers": TOKENIZERS_PROBLEMS,
    "transformers": TRANSFORMERS_PROBLEMS,
    "llama-cpp-python": LLAMA_CPP_PROBLEMS,
}


# add problems here that have long runtimes to reduce number of tests
LONG_RUNNING_PROBLEMS = [
    ("pandas-dataframe.__setitem__", "e7e3676", 5),
    ("datasets-load_dataset_builder", "ef3b5dd", 5),
    ("tokenizers-tokenizers.trainers.unigramtrainer.train", "fc76ad4", 5),
    ("pillow-simd-image.gaussian_blur", "9e60023", 5),
]
