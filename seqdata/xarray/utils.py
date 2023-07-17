from typing import Tuple, Union

import xarray as xr


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
