import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import zarr
from numcodecs import Blosc

from seqdata._io.bed_ops import (
    _bed_to_zarr,
    _expand_regions,
    _set_uniform_length_around_center,
    read_bedlike,
)
from seqdata.types import FlatReader, PathType, RegionReader

from .utils import _filter_by_exact_dims, _filter_layers, _filter_uns


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
    load_first = False,
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
    sdata = sdata.reset_encoding()

    for arr in sdata.data_vars.values():
        if "_FillValue" in arr.attrs:
            del arr.attrs["_FillValue"]

        # rechunk non-uniform chunking
        # Use chunk size that is:
        # 1. most frequent
        # 2. to break ties, largest
        if arr.chunksizes is not None:
            new_chunks = {}
            for dim, chunk in arr.chunksizes.items():
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

    if "strand" not in _bed:
        _bed["strand"] = "+"

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
        _bed_to_zarr(
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
        _bed_to_zarr(
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


def merge_obs(
    sdata: xr.Dataset,
    obs: Union[xr.Dataset, pd.DataFrame],
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: Literal["inner", "left", "right", "outer", "exact"] = "inner",
):
    if on is None and (left_on is None or right_on is None):
        raise ValueError
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError

    if on is None:
        assert left_on is not None
        assert right_on is not None
    else:
        left_on = on
        right_on = on

    if left_on not in sdata.data_vars:
        sdata = sdata.assign({left_on: np.arange(sdata.sizes[left_on])})
    if left_on not in sdata.xindexes:
        sdata = sdata.set_coords(left_on).set_xindex(left_on)

    if isinstance(obs, pd.DataFrame):
        if obs.index.name != right_on:
            obs = obs.set_index(right_on)
            obs.index.name = left_on
            obs = obs.to_xarray()
            sdata_dim = sdata[left_on].dims[0]
            obs_dim = obs[left_on].dims[0]
            if sdata_dim != obs_dim:
                obs[left_on].rename({obs_dim: sdata_dim})
        sdata = sdata.merge(obs, join=how)  # type: ignore
    elif isinstance(obs, xr.Dataset):
        if right_on not in obs.data_vars:
            obs = obs.assign({right_on: np.arange(sdata.sizes[right_on])})
        if right_on not in obs.xindexes:
            obs = (
                obs.rename({right_on: left_on}).set_coords(left_on).set_xindex(left_on)
            )
        sdata = sdata.merge(obs, join=how)

    return sdata


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
