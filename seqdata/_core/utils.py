from itertools import accumulate, chain, repeat
from typing import Sequence, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def _filter_by_exact_dims(ds: xr.Dataset, dims: Union[str, Tuple[str, ...]]):
    if isinstance(dims, str):
        dims = (dims,)
    else:
        dims = tuple(dims)
    selector = []
    for name, arr in ds.data_vars.items():
        if arr.dims == dims:
            selector.append(name)
    return ds[selector]


def _filter_layers(ds: xr.Dataset):
    selector = []
    for name, arr in ds.data_vars.items():
        if (
            len(arr.dims) > 1
            and arr.dims[0] == ds.attrs["sequence_dim"]
            and arr.dims[1] != ds.attrs["sequence_dim"]
        ):
            selector.append(name)
    return ds[selector]


def _filter_uns(ds: xr.Dataset):
    selector = []
    for name, arr in ds.data_vars.items():
        if ds.attrs["sequence_dim"] not in arr.dims:
            selector.append(name)
    return ds[selector]


def _cartesian_product(arrays: Sequence[NDArray]) -> NDArray:
    """Get the cartesian product of multiple arrays such that each entry corresponds to
    a unique combination of the input arrays' values.
    """
    # https://stackoverflow.com/a/49445693
    la = len(arrays)
    shape = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(shape, dtype=dtype)
    arrs = (*accumulate(chain((arr,), repeat(0, la - 1)), np.ndarray.__getitem__),)
    idx = slice(None), *repeat(None, la - 1)
    for i in range(la - 1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[: la - i]]
        arrs[i - 1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)
