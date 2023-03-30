import xarray as xr

from .utils import _filter_by_exact_dims, _filter_layers, _filter_obsm, _filter_uns, _filter_varm


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

    def to_zarr(self, *args, **kwargs):
        self.ds.to_zarr(*args, **kwargs, consolidated=True)

    @classmethod
    def open_zarr(cls, *args, **kwargs):
        ds = xr.open_zarr(*args, **kwargs)
        return cls(ds)

    def __repr__(self) -> str:
        msg = "SeqData holding an Xarray dataset:\n"
        msg += repr(self.ds)
        return msg
