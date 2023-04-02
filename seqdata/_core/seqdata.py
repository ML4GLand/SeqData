from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc
from typing_extensions import Self

from seqdata._io.utils import (
    _df_to_xr_zarr,
    _read_bedlike,
    _set_uniform_length_around_center,
)
from seqdata.types import FlatReader, PathType, RegionReader

from .utils import (
    _filter_by_exact_dims,
    _filter_layers,
    _filter_obsm,
    _filter_uns,
    _filter_varm,
)


def concat(*sdatas: "SeqData"):
    ds = xr.concat([sdata.ds for sdata in sdatas], dim="sequence")
    return SeqData(ds)


class SeqData:
    path: Optional[Path]
    max_jitter: Optional[int]

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds.copy()

    def sel(self, **kwargs):
        return SeqData(self.ds.sel(**kwargs))

    def isel(self, **kwargs):
        return SeqData(self.ds.isel(**kwargs))

    @property
    def layers(self):
        return _filter_layers(self.ds)

    @property
    def obs(self):
        return _filter_by_exact_dims(self.ds, "sequence")

    @property
    def var(self):
        return _filter_by_exact_dims(self.ds, "length")

    @property
    def obsm(self):
        return _filter_obsm(self.ds)

    @property
    def varm(self):
        return _filter_varm(self.ds)

    @property
    def obsp(self):
        return _filter_by_exact_dims(self.ds, ("sequence", "sequence"))

    @property
    def varp(self):
        return _filter_by_exact_dims(self.ds, ("length", "length"))

    @property
    def uns(self):
        return _filter_uns(self.ds)

    def __repr__(self) -> str:
        msg = "SeqData holding an Xarray dataset.\n"
        msg += repr(self.ds)
        return msg

    def _repr_html_(self) -> str:
        msg = "SeqData holding an Xarray dataset.\n"
        msg += self.ds._repr_html_()
        return msg

    def load(self, **kwargs):
        """Load this SeqData into memory, if it isn't already."""
        self.ds = self.ds.load(**kwargs)

    def to_zarr(
        self,
        store=None,
        chunk_store=None,
        mode=None,
        synchronizer=None,
        group=None,
        encoding=None,
        compute=True,
        append_dim=None,
        region=None,
        safe_chunks=True,
        storage_options=None,
        zarr_version=None,
    ):
        self.ds.to_zarr(
            store=store,
            chunk_store=chunk_store,
            mode=mode,
            synchronizer=synchronizer,
            group=group,
            encoding=encoding,
            compute=compute,  # type: ignore
            consolidated=True,
            append_dim=append_dim,
            region=region,
            safe_chunks=safe_chunks,
            storage_options=storage_options,
            zarr_version=zarr_version,
        )

    @classmethod
    def open_zarr(
        cls,
        store,
        group=None,
        synchronizer=None,
        chunks="auto",
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
        decode_coords=True,
        drop_variables=None,
        consolidated=None,
        overwrite_encoded_chunks=False,
        chunk_store=None,
        storage_options=None,
        decode_timedelta=None,
        use_cftime=None,
        zarr_version=None,
        **kwargs,
    ):
        ds = xr.open_zarr(
            store=store,
            group=group,
            synchronizer=synchronizer,
            chunks=chunks,
            decode_cf=decode_cf,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=False,
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
        self = cls(ds)
        self.path = Path(store)
        self.max_jitter = self.ds.attrs["max_jitter"]
        return self

    @classmethod
    def from_files(
        cls,
        *readers: Union[FlatReader, RegionReader],
        path: PathType,
        length: Optional[int] = None,
        bed: Optional[Union[PathType, pd.DataFrame]] = None,
        max_jitter: Optional[int] = None,
        overwrite=False,
    ) -> Self:
        max_jitter = 0 if max_jitter is None else max_jitter

        z = zarr.open_group(path)
        z.attrs["max_jitter"] = max_jitter

        if bed is not None and length is not None:
            length += 2 * max_jitter
            if isinstance(bed, (str, Path)):
                _bed = _read_bedlike(bed)
            else:
                _bed = bed
            _set_uniform_length_around_center(_bed, length)
            _df_to_xr_zarr(
                _bed,
                path,
                ["sequence"],
                compressor=Blosc("zstd", clevel=7, shuffle=-1),
                overwrite=overwrite,
            )
        elif bed is not None or length is not None:
            raise ValueError("bed and length are mutually inclusive arguments.")
        else:
            if any(map(lambda r: isinstance(r, RegionReader), readers)):
                raise ValueError(
                    "Got readers that need bed and length but didn't get bed and length."
                )

        for reader in readers:
            if isinstance(reader, FlatReader):
                reader._write(path, overwrite)
            elif isinstance(reader, RegionReader):
                reader._write(
                    path,
                    length,  # type: ignore
                    _bed,  # type: ignore
                    overwrite,
                )

        zarr.consolidate_metadata(path)  # type: ignore

        self = cls.open_zarr(path)
        return self

    def add_layers_from_files(
        self,
        *readers: Union[FlatReader, RegionReader],
        overwrite=False,
    ):
        if self.path is None:
            raise ValueError("SeqData must be backed on disk to add layers from files.")

        if any(map(lambda r: isinstance(r, RegionReader), readers)):
            length = self.ds.sizes["length"]
            bed = self.ds[["chrom", "chromStart", "chromEnd", "strand"]].to_dataframe()

        for reader in readers:
            if isinstance(reader, FlatReader):
                if (
                    reader.n_seqs is not None
                    and reader.n_seqs != self.ds.sizes["sequence"]
                ):
                    raise ValueError(
                        f'Reader "{reader.name}" has a different number of sequences than this SeqData.'
                    )
                reader._write(self.path, overwrite)
            elif isinstance(reader, RegionReader):
                reader._write(
                    self.path,
                    length,  # type: ignore
                    bed,  # type: ignore
                    overwrite,
                )

        self = SeqData.open_zarr(self.path)
        return self
