import xarray as xr

from .seqdata import SeqData


def concat(*sdatas: SeqData):
    ds = xr.concat([sdata.ds for sdata in sdatas], dim="sequence")
    return SeqData(ds)
