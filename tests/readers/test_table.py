from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import polars as pl
import seqpro as sp
import xarray as xr
from pytest_cases import fixture

import seqdata as sd


@fixture
def gen_table():
    """Dummy dataset with AA, CC, GG, TT sequences."""

    ds = xr.Dataset(
        {
            "seq": xr.DataArray(
                sp.random_seqs((2, 5), sp.DNA, 0), dims=["_sequence", "_length"]
            ),
            "target": xr.DataArray([5, 11.2], dims=["_sequence"]),
        }
    )
    return ds


def write_table(csv: str, ds: xr.Dataset):
    (
        ds.assign(
            seq=xr.DataArray(
                ds["seq"].values.view("S5").astype(str).squeeze(), dims=["_sequence"]
            )
        )
        .to_pandas()
        .reset_index(drop=True)
        .to_csv(csv, index=False)
    )


def test_write_table(gen_table):
    with NamedTemporaryFile(suffix=".csv") as csv:
        write_table(csv.name, gen_table)

        df = pl.read_csv(csv.name)
        for name, da in gen_table.items():
            df_data = df[name].to_numpy()
            if da.dtype.char == "S":
                df_data = df_data.astype("S")[..., None].view("S1")
            np.testing.assert_array_equal(da.values, df_data)


def test_table(gen_table):
    with (
        NamedTemporaryFile(suffix=".csv") as csv,
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        write_table(csv.name, gen_table)

        ds = sd.from_flat_files(
            sd.Table("seq", csv.name, "seq", batch_size=1),
            path=out,
            fixed_length=True,
        )

        for name, da in gen_table.items():
            np.testing.assert_array_equal(da, ds[name])
