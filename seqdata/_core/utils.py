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
        if arr.dims[:2] == ("sequence", "length"):
            selector.append(name)
    return ds[selector]


def _filter_obsm(ds: xr.Dataset):
    selector = []
    for name, arr in ds.data_vars.items():
        if len(arr.dims) > 1 and arr.dims[0] == "sequence" and "length" not in arr.dims:
            selector.append(name)
    return ds[selector]


def _filter_varm(ds: xr.Dataset):
    selector = []
    for name, arr in ds.data_vars.items():
        if len(arr.dims) > 1 and arr.dims[0] == "length" and "sequence" not in arr.dims:
            selector.append(name)
    return ds[selector]


def _filter_uns(ds: xr.Dataset):
    selector = []
    for name, arr in ds.data_vars.items():
        if np.isin(arr.dims, ["sequence", "length"], invert=True).all():  # type: ignore
            selector.append(name)
    return ds[selector]