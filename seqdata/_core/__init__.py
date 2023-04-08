from typing import (
    Dict,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import xarray as xr

from seqdata.types import PathType

from .seqdata import SeqData


def concat(*sdatas: SeqData):
    ds = xr.concat([sdata.ds for sdata in sdatas], dim="sequence")
    return SeqData(ds)


def open_zarr(
    store: PathType,
    group: Optional[str] = None,
    synchronizer=None,
    chunks: Optional[
        Union[Literal["auto"], int, Mapping[str, int], Tuple[int, ...]]
    ] = "auto",
    decode_cf=True,
    mask_and_scale=True,
    decode_times=True,
    concat_characters=True,
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
    return SeqData.open_zarr(
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
