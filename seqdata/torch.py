import warnings
from itertools import accumulate, chain, repeat
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)

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
    transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples: Literal[False],
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[List]", Iterable[List]]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Dict[str, torch.Tensor]]": ...


@overload
def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples: Literal[True],
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[List]", Iterable[List]]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Tuple[torch.Tensor, ...]]": ...


@overload
def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples=False,
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[List]", Iterable[List]]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
) -> "DataLoader[Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]]": ...


def get_torch_dataloader(
    sdata: xr.Dataset,
    sample_dims: Union[str, List[str]],
    variables: Union[str, List[str]],
    transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
    *,
    return_tuples=False,
    batch_size: Optional[int] = 1,
    shuffle=False,
    sampler: Optional[Union["Sampler", Iterable]] = None,
    batch_sampler: Optional[Union["Sampler[List]", Iterable[List]]] = None,
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

    if isinstance(dtypes, torch.dtype):
        dtypes = {k: dtypes for k in variables}
    dim_sizes = [sdata.sizes[d] for d in sample_dims]
    ravel_indices = cast(
        NDArray[np.intp],
        np.arange(np.prod(dim_sizes, dtype=int), dtype=np.intp),  # type: ignore
    )
    data: Dict[Hashable, NDArray] = {
        var: arr.to_numpy()
        for var, arr in sdata[variables].transpose(*sample_dims, ...).items()
    }

    def collate_fn(indices: List[np.intp]):
        # improve performance by sorted indexing
        # note: assumes order within batch is irrelevant (true for ML)
        indices.sort()
        _idxs = np.unravel_index(indices, dim_sizes)

        # select data
        out = {
            var: dat[
                tuple(_idxs[i] for i, d in enumerate(sample_dims) if d in arr.dims)
            ]
            for dat, (var, arr) in zip(
                data.values(), sdata[variables].data_vars.items()
            )
        }
        out = cast(Dict[str, NDArray], out)

        # apply transforms
        if transform is not None:
            out = transform(out)

        # convert to torch
        for k in out:
            out[k] = torch.as_tensor(out[k], dtype=dtypes[k])  # type: ignore
        out = cast(Dict[str, torch.Tensor], out)

        # convert to a tuple if desired
        if return_tuples:
            out = tuple(out.values())

        return out

    return DataLoader(
        ravel_indices,  # type: ignore
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
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


# TODO: allow in-memory sdata
# TODO: add parameters for `sampler`, `pin_memory`, `drop_last`
class XArrayDataLoader:
    def __init__(
        self,
        sdata: xr.Dataset,
        sample_dims: Union[str, List[str]],
        variables: Union[str, List[str]],
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
        dtypes: Union[torch.dtype, Dict[str, torch.dtype]] = torch.float32,
        batch_size: int = 1,
        prefetch_factor: int = 2,
        shuffle: bool = False,
        seed: Optional[int] = None,
        return_tuples: bool = False,
    ) -> None:
        """Get an XArray DataLoader that supports substantially faster out-of-core
        dataloading from chunked storage formats than a PyTorch DataLoader. Note the
        absence of concurrency parameters. This is intentional: concurrent I/O is
        enabled by instantiating a `dask.distributed.Client` before iteration.

        Parameters
        ----------
        sdata : xr.Dataset
        sample_dims : Union[str, List[str]]
            Dimensions to sample over (i.e. what dimensions would you index over to get
            a single instance?)
        variables : Union[str, List[str]]
            What variables to load data from.
        transform : Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]]
            A function to transform batches after loading them into memory. Should take
            a dictionary of numpy arrays, each corresponding to a `variable`, transform
            them, and return the result as a dictionary with the same keys. By default
            no transforms will be applied.
        dtypes : Union[torch.dtype, Dict[str, torch.dtype]], optional
            What dtype to convert each batch array to. Either a single dtype to convert
            all variables or a dictionary mapping variables to dtypes. By default
            `torch.float32`.
        batch_size : int, optional
            How many instances per batch, by default 1
        prefetch_factor : int, optional
            What multiple of chunks to prefetch, by default 2. Tune this and the Zarr
            chunk sizes appropriately to control peak memory usage and balance speed and
            memory usage. A higher prefetch factor improves speed but uses more memory.
        shuffle : bool, optional
            Whether to randomly shuffle the dataset on each epoch, by default False
        seed : Optional[int], optional
            Seed for random shuffling, by default None
        return_tuples : bool, optional
            Whether to yield tuples (or dictionaries). By default False.

        Raises
        ------
        ValueError
            When `variables` specifies variables that aren't in the Dataset.
        ValueError
            When variables have different chunk sizes in any of the sample dimensions.

        Notes
        -----
        **Data flow**

        1. Load contiguous chunks of data from the dataset into buffers that are larger
        than the batch size.
        2. Yield batches from the buffer until the buffer is empty, then repeat.

        **Random shuffling**

        We implement random shuffling by prefetching random chunks and then randomly
        sampling data from within those chunks. It is possible (although unlikely) that
        the data may have structure that isn't randomized due to the lack of fully
        random sampling.
        """
        if isinstance(sample_dims, str):
            sample_dims = [sample_dims]
        if isinstance(variables, str):
            variables = [variables]

        variables_not_in_ds = set(variables) - set(sdata.data_vars.keys())
        if variables_not_in_ds:
            raise ValueError(
                f"Got variables that are not in the dataset: {variables_not_in_ds}"
            )

        if isinstance(dtypes, torch.dtype):
            self.dtypes = {k: dtypes for k in variables}
        else:
            self.dtypes = dtypes

        self.sdata = sdata
        self.variables = variables
        # mapping from dimension name to chunksize
        self.chunksizes = self.get_chunksizes(sdata, sample_dims, variables)
        self.sample_dims = sample_dims

        self.instances_per_chunk = np.prod(list(self.chunksizes.values()), dtype=int)
        chunks_per_batch = -(-batch_size // self.instances_per_chunk)
        self.n_prefetch_chunks = prefetch_factor * chunks_per_batch
        self.n_instances = np.prod([sdata.sizes[d] for d in sample_dims], dtype=int)
        if batch_size > self.n_instances:
            warnings.warn(
                f"""Batch size {batch_size} is larger than the number of instances in 
                the dataset {self.n_instances}. Reducing batch size to maximum number of
                instances."""
            )
            self.batch_size = self.n_instances
        else:
            self.batch_size = batch_size
        self.max_batches = -(-self.n_instances // self.batch_size)

        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.transform = transform
        self.return_tuples = return_tuples

        chunk_start_idx: Dict[str, NDArray[np.int64]] = {}
        for dim in self.chunksizes:
            length = sdata.sizes[dim]
            chunksize = self.chunksizes[dim]
            chunk_start_idx[dim] = np.arange(0, length, chunksize, dtype=np.int64)
        self.chunk_idxs = _cartesian_product(list(chunk_start_idx.values()))

    def get_chunksizes(
        self, sdata: xr.Dataset, sample_dims: List[str], variables: List[str]
    ):
        chunksizes: Dict[str, Set[int]] = {}
        for dim in sample_dims:
            dim_chunk_sizes = set()
            for v in sdata[variables].data_vars.values():
                if dim in v.dims:
                    dim_chunk_sizes.add(v.data.chunksize[v.get_axis_num(dim)])
            chunksizes[dim] = dim_chunk_sizes
        discrepant_chunk_sizes = {k: v for k, v in chunksizes.items() if len(v) > 1}
        if len(discrepant_chunk_sizes) > 1:
            raise ValueError(
                f"""Variables have different chunksizes in the sample dimensions.\n
                Dimensions with discrepant chunksizes: {list(discrepant_chunk_sizes.keys())}.\n
                Rechunk the variables in the sample dimensions so they are the same.
                """
            )
        return {k: v.pop() for k, v in chunksizes.items()}

    def __len__(self):
        return self.max_batches

    def __iter__(self):
        # which slice of chunks is going into the buffer
        self.chunk_slice = slice(0, self.n_prefetch_chunks)
        # which slice of the buffer is going into the batch
        self.buffer_slice = slice(0, self.batch_size)
        # which slice of the batch is getting pulled & processed
        # i.e. batch[self.batch_slice] = self.buffer[self.buffer_slice]
        self.batch_slice = slice(0, self.batch_size)
        self.current_batch = 0
        if self.shuffle:
            self.chunk_idxs = self.rng.permutation(self.chunk_idxs, axis=0)
        self._flush_and_fill_buffers()
        return self

    def _flush_and_fill_buffers(self):
        """Flush buffers and fill them with new data."""
        # Each buffer in buffers will have shape (self.buffer_size, ...)
        self.buffers: Dict[str, NDArray] = {}
        shuffler = None
        # (n_chunks, n_dim)
        chunk_idx = self.chunk_idxs[self.chunk_slice]
        self.chunk_slice = slice(
            self.chunk_slice.start, self.chunk_slice.start + self.n_prefetch_chunks
        )
        for var in self.variables:
            var_dims = [d for d in self.sdata[var].dims if d in self.sample_dims]
            buffer = []
            for chunk in chunk_idx:
                selector = {
                    d: slice(start, start + self.chunksizes[d])
                    for start, d in zip(chunk, self.sample_dims)
                }
                buffer.append(
                    self.sdata[var]
                    .isel(selector, missing_dims="ignore")
                    .stack(batch=var_dims)
                    .transpose("batch", ...)
                    .to_numpy()
                )
            buffer = np.concatenate(buffer)
            if shuffler is None:
                shuffler = self.rng.permutation(len(buffer))
            if self.shuffle:
                buffer = buffer[shuffler]
            self.buffers[var] = buffer

    def __next__(self):
        if self.current_batch == self.max_batches:
            raise StopIteration

        # init empty batch arrays
        batch: Dict[str, NDArray] = {
            k: np.empty_like(v.data, shape=(self.batch_size, *v.shape[1:]))
            for k, v in self.buffers.items()
        }

        overshoot = self.buffer_slice.stop - len(self.buffers[self.variables[0]])

        # buffers don't have enough data to fill the batch
        if overshoot > 0:
            # grab what they do have
            self.batch_slice = slice(0, self.batch_size - overshoot)
            for var, buffer in self.buffers.items():
                batch[var][self.batch_slice] = buffer[self.buffer_slice]

            # fetch more data
            self._flush_and_fill_buffers()

            # setup to fill the rest of the batch
            self.buffer_slice = slice(0, overshoot)
            self.batch_slice = slice(self.batch_slice.stop, self.batch_size)

        for var, buffer in self.buffers.items():
            batch[var][self.batch_slice] = buffer[self.buffer_slice]

        # setup for next batch
        self.buffer_slice = slice(
            self.buffer_slice.stop, self.buffer_slice.stop + self.batch_size
        )
        self.batch_slice = slice(0, self.batch_size)
        self.current_batch += 1

        # apply transforms, if any
        if self.transform is not None:
            batch = self.transform(batch)

        out = self._apply_dtypes(batch)

        if self.return_tuples:
            return tuple(out.values())

        return out

    def _apply_dtypes(self, batch: Dict[str, NDArray]):
        out = {
            k: torch.as_tensor(v, dtype=dtype)
            for (k, dtype), v in zip(self.dtypes.items(), batch.values())
        }
        return out
