import xarray as xr

from .utils import (
    _filter_by_exact_dims,
    _filter_layers,
    _filter_obsm,
    _filter_uns,
    _filter_varm,
)


class SeqData:
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
        **kwargs
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
            **kwargs
        )
        return cls(ds)

    def __repr__(self) -> str:
        msg = "SeqData holding an Xarray dataset:\n"
        msg += repr(self.ds)
        return msg

    def _repr_html_(self) -> str:
        msg = "SeqData holding an Xarray dataset\n"
        msg += self.ds._repr_html_()
        return msg
