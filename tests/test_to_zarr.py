import tempfile

import numpy as np
import xarray as xr
from pytest_cases import fixture

import seqdata as sd


@fixture
def sdata():
    """Dummy dataset with AA, CC, GG, TT sequences."""
    seqs = np.array([[b"A", "A"], [b"C", "C"], [b"G", "G"], [b"T", b"T"]])
    return xr.Dataset(
        {
            "seqs": xr.DataArray(seqs, dims=["_sequence", "_length"]),
        }
    )


def test_to_zarr_non_uniform_chunks(sdata: xr.Dataset):
    # set chunks to violate write requirements
    # - uniform except last
    # - last <= in size than the rest
    sdata = sdata.chunk({"_sequence": (1, 3), "_length": -1})

    with tempfile.TemporaryDirectory() as tmpdir:
        sd.to_zarr(sdata, tmpdir, mode="w")

        after = sd.open_zarr(tmpdir)
        # chunks should satisfy write requirements
        assert after.chunksizes == {"_sequence": (3, 1), "_length": (2,)}
