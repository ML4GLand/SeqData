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
    overload,
)

import numpy as np
import xarray as xr
from numpy.typing import NDArray

try:
    import torch
    from torch.utils.data import DataLoader, Sampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    transforms: Optional[Dict[str, Callable[[NDArray], "torch.Tensor"]]] = None,
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
    transforms: Optional[Dict[str, Callable[[NDArray], "torch.Tensor"]]] = None,
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
    transforms: Optional[Dict[str, Callable[[NDArray], "torch.Tensor"]]] = None,
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
    transforms: Optional[Dict[str, Callable[[NDArray], "torch.Tensor"]]] = None,
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
    transforms : Dict[str, (ndarray) -> Tensor], optional
        Transforms to apply to each variable. Defaults to simply transforming arrays
        to Tensor.

    For other parameters, see documentation for [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Returns
    -------
    DataLoader that returns dictionaries or tuples of tensors.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Install PyTorch to get a DataLoader from SeqData.")

    variables_not_in_ds = set(variables) - set(sdata.data_vars.keys())
    if variables_not_in_ds:
        raise ValueError(
            f"Got variables that are not in the SeqData: {variables_not_in_ds}"
        )

    if transforms is None:
        _transforms: Dict[str, Callable[[NDArray], "torch.Tensor"]] = {}
    else:
        _transforms = transforms

    transform_vars_not_in_vars = set(_transforms.keys()) - set(variables)
    if transform_vars_not_in_vars:
        raise ValueError(
            f"""Got transforms for variables that are not requested: 
            {transform_vars_not_in_vars}"""
        )

    transform_vars_not_in_ds = set(_transforms.keys()) - set(sdata.data_vars.keys())
    if transform_vars_not_in_ds:
        raise ValueError(
            f"""Got transforms for variables that are not in the SeqData: 
            {transform_vars_not_in_ds}"""
        )

    if isinstance(sample_dims, str):
        sample_dims = [sample_dims]
    if isinstance(variables, str):
        variables = [variables]

    dim_sizes = [sdata.dims[d] for d in sample_dims]
    dataset = _cartesian_product([np.arange(d) for d in dim_sizes])

    def collate_fn(indices: List[NDArray]):
        idx = np.vstack(indices)
        # avoid Dask PerformanceWarning using an unsorted 1-d indexer
        if idx.shape[-1] == 1:
            idx = np.sort(idx, axis=None).reshape(-1, 1)
        selector = {
            d: xr.DataArray(idx[:, i], dims="batch") for i, d in enumerate(sample_dims)
        }
        out = {
            k: arr.isel(selector, missing_dims="ignore").to_numpy()
            for k, arr in sdata[variables].data_vars.items()
        }
        out_tensors = {
            k: _transforms.get(k, lambda x: torch.as_tensor(x))(arr)
            for k, arr in out.items()
        }
        if return_tuples:
            out_tensors = tuple(out_tensors.values())
        return out_tensors

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
