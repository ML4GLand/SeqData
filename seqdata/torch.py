from itertools import accumulate, chain, repeat
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import dask.config
import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Sampler


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


@overload
def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transforms: Optional[
        Dict[
            Union[str, Tuple[str]], Callable[[Union[NDArray, Tuple[NDArray]]], NDArray]
        ]
    ] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples: Literal[False],
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[Sequence]", Iterable[Sequence]]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Dict[str, torch.Tensor]]":
    ...


@overload
def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transforms: Optional[
        Dict[
            Union[str, Tuple[str]], Callable[[Union[NDArray, Tuple[NDArray]]], NDArray]
        ]
    ] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples: Literal[True],
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[Sequence]", Iterable[Sequence]]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Tuple[torch.Tensor, ...]]":
    ...


@overload
def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transforms: Optional[
        Dict[
            Union[str, Tuple[str]], Callable[[Union[NDArray, Tuple[NDArray]]], NDArray]
        ]
    ] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples=False,
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[Sequence]", Iterable[Sequence]]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]]":
    ...


def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transforms: Optional[
        Dict[
            Union[str, Tuple[str]], Callable[[Union[NDArray, Tuple[NDArray]]], NDArray]
        ]
    ] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples=False,
    batch_size: Optional[int] = 1,
    shuffle=False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[Sequence]", Iterable[Sequence]]] = None,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    timeout=0.0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]]":
    """Get a PyTorch DataLoader for this SeqData.

    Parameters
    ----------
    sample_dims : str or list[str]
        Sample dimensions that will be indexed over when fetching batches. For
        example, if `sample_dims = ['_sequence', 'sample']` for a variable with
        dimensions `['_sequence', 'length', 'sample']` then a batch of data will
        have dimensions `['batch', 'length']`.
    variables : list[str]
        Which variables to sample from.
    transforms : Dict[str | tuple[str], (ndarray | tuple[ndarray]) -> ndarray], optional
        Transforms to apply to each variable. Will be applied in order and keys that are
        tuples of strings will pass the corresponding variables to the transform in the
        order that the variable names appear. See examples for details.
    dtypes : torch.dtype, Dict[str, torch.dtype]
        Data type to convert each variable to after applying all transforms.

    For other parameters, see documentation for [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Returns
    -------
    DataLoader that returns dictionaries or tuples of tensors.
    """

    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if isinstance(variables, str):
        variables = [variables]

    variables_not_in_ds = set(variables) - set(sdata.data_vars.keys())
    if variables_not_in_ds:
        raise ValueError(
            f"Got variables that are not in the SeqData: {variables_not_in_ds}"
        )

    if transforms is None:
        _transforms: Dict[
            Union[str, Tuple[str]], Callable[[Union[NDArray, Tuple[NDArray]]], NDArray]
        ] = {}
    else:
        _transforms = transforms

    vars_with_transforms = set()
    for k in _transforms:
        if isinstance(k, tuple):
            vars_with_transforms.update(k)
        else:
            vars_with_transforms.add(k)
    transform_vars_not_in_vars = vars_with_transforms - set(variables)
    if transform_vars_not_in_vars:
        raise ValueError(
            f"""Got transforms for variables that are not requested: 
            {transform_vars_not_in_vars}"""
        )

    transform_vars_not_in_ds = vars_with_transforms - set(sdata.data_vars.keys())
    if transform_vars_not_in_ds:
        raise ValueError(
            f"""Got transforms for variables that are not in the dataset: 
            {transform_vars_not_in_ds}"""
        )

    if isinstance(dtypes, torch.dtype):
        dtypes = {k: dtypes for k in variables}
    dim_sizes = [sdata.dims[d] for d in sample_dims]
    dataset = _cartesian_product([np.arange(d, dtype="uintp") for d in dim_sizes])

    def collate_fn(indices: List[NDArray]):
        idx = np.vstack(indices)

        # avoid Dask PerformanceWarning caused by using an unsorted 1-d indexer
        if idx.shape[-1] == 1:
            idx = np.sort(idx, axis=None).reshape(-1, 1)

        # make a selector to grab the batch
        selector = {
            d: xr.DataArray(idx[:, i], dims="batch") for i, d in enumerate(sample_dims)
        }

        # select data and convert to numpy
        out: Union[Tuple[torch.Tensor], Dict[str, NDArray], Dict[str, torch.Tensor]]
        with dask.config.set({"array.slicing.split_large_chunks": False}):
            out = {
                k: arr.isel(selector, missing_dims="ignore").to_numpy()
                for k, arr in sdata[variables].data_vars.items()
            }
            out = cast(Dict[str, NDArray], out)

        # apply transforms
        for k, fn in _transforms.items():
            if isinstance(k, tuple):
                _arrs = tuple(out[var] for var in k)
                out.update(dict(zip(k, fn(_arrs))))
            else:
                out[k] = fn(out[k])

        # convert to torch
        out = cast(Dict[str, torch.Tensor], out)
        for k in out:
            out[k] = torch.as_tensor(out[k], dtype=dtypes[k])

        # convert to a tuple if desired
        if return_tuples:
            out = tuple(out.values())

        return out

    return DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        prefetch_factor=prefetch_factor,  # type: ignore
        persistent_workers=persistent_workers,
    )
