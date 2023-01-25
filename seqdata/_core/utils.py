import numpy as np
import pandas as pd
from functools import singledispatch
from pandas.api.types import is_string_dtype

@singledispatch
def _gen_dataframe(anno, length, index_names):
    if anno is None or len(anno) == 0:
        return pd.DataFrame(index=pd.RangeIndex(0, length, name=None).astype(str))
    for index_name in index_names:
        if index_name in anno:
            return pd.DataFrame(
                anno,
                index=anno[index_name],
                columns=[k for k in anno.keys() if k != index_name],
            )
    return pd.DataFrame(anno, index=pd.RangeIndex(0, length, name=None).astype(str))


@_gen_dataframe.register(pd.DataFrame)
def _(anno, length, index_names):
    anno = anno.copy(deep=False)
    if not is_string_dtype(anno.index):
        # warnings.warn("Transforming to str index.", ImplicitModificationWarning)
        anno.index = anno.index.astype(str)
    return anno


@_gen_dataframe.register(pd.Series)
@_gen_dataframe.register(pd.Index)
def _(anno, length, index_names):
    raise ValueError(f"Cannot convert {type(anno)} to DataFrame")


@singledispatch
def convert_to_dict(obj) -> dict:
    return dict(obj)


@convert_to_dict.register(dict)
def convert_to_dict_dict(obj: dict):
    return obj


@convert_to_dict.register(np.ndarray)
def convert_to_dict_ndarray(obj: np.ndarray):
    if obj.dtype.fields is None:
        raise TypeError(
            "Can only convert np.ndarray with compound dtypes to dict, "
            f"passed array had “{obj.dtype}”."
        )
    return {k: obj[k] for k in obj.dtype.fields.keys()}


@convert_to_dict.register(type(None))
def convert_to_dict_nonetype(obj: None):
    return dict()