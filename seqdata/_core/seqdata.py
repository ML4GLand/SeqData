import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import zarr
from numcodecs import Blosc
from numpy.typing import NDArray

from seqdata._io.bed_ops import (
    _expand_regions,
    _set_uniform_length_around_center,
    read_bedlike,
)
from seqdata._io.utils import _polars_df_to_xr_zarr
from seqdata.types import FlatReader, PathType, RegionReader

from .utils import (
    _cartesian_product,
    _filter_by_exact_dims,
    _filter_layers,
    _filter_uns,
)

try:
    import torch
    from torch.utils.data import DataLoader, Sampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def open_zarr(
    store: PathType,
    group: Optional[str] = None,
    synchronizer=None,
    chunks: Optional[
        Union[Literal["auto"], int, Mapping[str, int], Tuple[int, ...]]
    ] = "auto",
    decode_cf=True,
    mask_and_scale=False,
    decode_times=True,
    concat_characters=False,
    decode_coords=True,
    drop_variables: Optional[Union[str, Iterable[str]]] = None,
    consolidated: Optional[bool] = None,
    overwrite_encoded_chunks=False,
    chunk_store: Optional[Union[MutableMapping, PathType]] = None,
    storage_options: Optional[Dict[str, str]] = None,
    decode_timedelta: Optional[bool] = None,
    use_cftime: Optional[bool] = None,
    zarr_version: Optional[int] = None,
    **kwargs,
):
    ds = xr.open_zarr(
        store=store,
        group=group,
        synchronizer=synchronizer,
        chunks=chunks,  # type: ignore
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        consolidated=consolidated,
        overwrite_encoded_chunks=overwrite_encoded_chunks,
        chunk_store=chunk_store,
        storage_options=storage_options,
        decode_timedelta=decode_timedelta,
        use_cftime=use_cftime,
        zarr_version=zarr_version,
        **kwargs,
    )
    return ds


def to_zarr(
    sdata: xr.Dataset,
    store: PathType,
    chunk_store: Optional[Union[MutableMapping, PathType]] = None,
    mode: Optional[Literal["w", "w-", "a", "r+"]] = None,
    synchronizer: Optional[Any] = None,
    group: Optional[str] = None,
    encoding: Optional[Dict] = None,
    compute=True,
    consolidated: Optional[bool] = None,
    append_dim: Optional[Hashable] = None,
    region: Optional[Dict] = None,
    safe_chunks=True,
    storage_options: Optional[Dict] = None,
    zarr_version: Optional[int] = None,
):
    sdata.reset_encoding()

    # rechunk non-uniform chunking
    # Use chunk size that is:
    # 1. most frequent
    # 2. to break ties, largest
    for arr in sdata.data_vars.values():
        if arr.chunksizes is not None:
            new_chunks = {}
            chunk: Tuple[int, ...]
            for dim, chunk in arr.chunksizes:
                if len(chunk) > 1 and (
                    (len(set(chunk[:-1])) > 1 or chunk[-2] > chunk[-1])
                ):
                    chunks, counts = np.unique(chunk, return_counts=True)
                    chunk_size = chunks[counts == counts.max()].max()
                    new_chunks[dim] = chunk_size
                else:
                    new_chunks[dim] = chunk
            arr.chunk(new_chunks)

    sdata.to_zarr(
        store=store,
        chunk_store=chunk_store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        encoding=encoding,
        compute=compute,  # type: ignore
        consolidated=consolidated,
        append_dim=append_dim,
        region=region,
        safe_chunks=safe_chunks,
        storage_options=storage_options,
        zarr_version=zarr_version,
    )


def from_flat_files(
    *readers: FlatReader,
    path: PathType,
    fixed_length: bool,
    sequence_dim: Optional[str] = None,
    length_dim: Optional[str] = None,
    overwrite=False,
) -> xr.Dataset:
    """Save a SeqData to disk and open it (without loading it into memory).

    Parameters
    ----------
    path : str, Path
        Path to save this SeqData to.
    fixed_length : bool
        `True`: assume the all sequences have the same length and will infer it
        from the first sequence.

        `False`: write variable length sequences.
    overwrite : bool, optional
        Whether to overwrite existing arrays of the SeqData at `path`, by default False

    Returns
    -------
    xr.Dataset
    """
    sequence_dim = "_sequence" if sequence_dim is None else sequence_dim
    if not fixed_length and length_dim is not None:
        warnings.warn("Treating sequences as variable length, ignoring `length_dim`.")
    elif fixed_length:
        length_dim = "_length" if length_dim is None else length_dim

    for reader in readers:
        reader._write(
            out=path,
            fixed_length=fixed_length,
            overwrite=overwrite,
            sequence_dim=sequence_dim,
            length_dim=length_dim,
        )

    zarr.consolidate_metadata(path)  # type: ignore

    ds = open_zarr(path)
    return ds


def from_region_files(
    *readers: RegionReader,
    path: PathType,
    fixed_length: Union[int, bool],
    bed: Union[PathType, pd.DataFrame],
    max_jitter=0,
    sequence_dim: Optional[str] = None,
    length_dim: Optional[str] = None,
    splice=False,
    overwrite=False,
) -> xr.Dataset:
    """Save a SeqData to disk and open it (without loading it into memory).

    Parameters
    ----------
    path : str, Path
        Path to save this SeqData to.
    fixed_length : int, bool, optional
        `int`: use regions of this length centered around those in the BED file.

        `True`: assume the all sequences have the same length and will try to infer it
        from the data.

        `False`: write variable length sequences
    bed : str, Path, pd.DataFrame, optional
        BED file or DataFrame matching the BED3+ specification describing what regions
        to write.
    max_jitter : int, optional
        How much jitter to allow for the SeqData object by writing additional
        flanking sequences, by default 0
    sequence_dim : str, optional
        Name of sequence dimension. Defaults to "_sequence".
    length_dim : str, optional
        Name of length dimension. Defaults to "_length".
    splice : bool, optional
        Whether to splice together regions that have the same `name` in the BED file, by
        default False
    overwrite : bool, optional
        Whether to overwrite existing arrays of the SeqData at `path`, by default False

    Returns
    -------
    xr.Dataset
    """
    sequence_dim = "_sequence" if sequence_dim is None else sequence_dim
    if not fixed_length and length_dim is not None:
        warnings.warn("Treating sequences as variable length, ignoring `length_dim`.")
    elif fixed_length:
        length_dim = "_length" if length_dim is None else length_dim

    root = zarr.open_group(path)
    root.attrs["max_jitter"] = max_jitter
    root.attrs["sequence_dim"] = sequence_dim
    root.attrs["length_dim"] = length_dim

    if isinstance(bed, (str, Path)):
        _bed = read_bedlike(bed)
    else:
        _bed = bed

    if not splice:
        if fixed_length is False:
            _expand_regions(_bed, max_jitter)
        else:
            if fixed_length is True:
                fixed_length = cast(
                    int,
                    _bed.loc[0, "chromEnd"] - _bed.loc[0, "chromStart"],  # type: ignore
                )
            fixed_length += 2 * max_jitter
            _set_uniform_length_around_center(_bed, fixed_length)
        _polars_df_to_xr_zarr(
            pl.from_pandas(_bed),
            root,
            sequence_dim,
            compressor=Blosc("zstd", clevel=7, shuffle=-1),
            overwrite=overwrite,
        )
    else:
        _bed = pl.from_pandas(_bed)
        if max_jitter > 0:
            _bed = _bed.with_columns(
                pl.when(pl.col("chromStart") == pl.col("chromStart").min().over("name"))
                .then(pl.col("chromStart").min().over("name") - max_jitter)
                .otherwise(pl.col("chromStart"))
                .alias("chromStart"),
                pl.when(pl.col("chromEnd") == pl.col("chromEnd").max().over("name"))
                .then(pl.col("chromEnd").max().over("name") + max_jitter)
                .otherwise(pl.col("chromEnd"))
                .alias("chromEnd"),
            )
        bed_to_write = _bed.groupby("name").agg(
            pl.col(pl.Utf8).first(), pl.exclude(pl.Utf8)
        )
        _polars_df_to_xr_zarr(
            bed_to_write,
            root,
            sequence_dim,
            compressor=Blosc("zstd", clevel=7, shuffle=-1),
            overwrite=overwrite,
        )
        _bed = _bed.to_pandas()

    for reader in readers:
        reader._write(
            out=path,
            bed=_bed,
            fixed_length=fixed_length,
            sequence_dim=sequence_dim,
            length_dim=length_dim,
            overwrite=overwrite,
            splice=splice,
        )

    zarr.consolidate_metadata(path)  # type: ignore

    ds = open_zarr(path)
    return ds


@xr.register_dataset_accessor("sd")
class SeqDataAccessor:
    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    @property
    def obs(self):
        return _filter_by_exact_dims(self._ds, self._ds.attrs["sequence_dim"])

    @property
    def layers(self):
        return _filter_layers(self._ds)

    @property
    def obsp(self):
        return _filter_by_exact_dims(
            self._ds, (self._ds.attrs["sequence_dim"], self._ds.attrs["sequence_dim"])
        )

    @property
    def uns(self):
        return _filter_uns(self._ds)

    def __repr__(self) -> str:
        return "SeqData accessor."


def add_layers_from_files(
    sdata: xr.Dataset,
    *readers: Union[FlatReader, RegionReader],
    path: PathType,
    overwrite=False,
):
    raise NotImplementedError
    if any(map(lambda r: isinstance(r, RegionReader), readers)):
        bed = sdata[["chrom", "chromStart", "chromEnd", "strand"]].to_dataframe()

    for reader in readers:
        if isinstance(reader, FlatReader):
            if reader.n_seqs is not None and reader.n_seqs != sdata.sizes["_sequence"]:
                raise ValueError(
                    f"""Reader "{reader.name}" has a different number of sequences 
                    than this SeqData."""
                )
            _fixed_length = fixed_length is not False
            reader._write(out=path, fixed_length=_fixed_length, overwrite=overwrite)
        elif isinstance(reader, RegionReader):
            reader._write(
                out=path,
                bed=bed,  # type: ignore
                overwrite=overwrite,
            )

    ds = xr.open_zarr(path, mask_and_scale=False, concat_characters=False)
    return ds


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
