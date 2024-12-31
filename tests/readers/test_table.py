from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import polars as pl
import seqpro as sp
import xarray as xr
from pytest_cases import fixture
import zarr

import seqdata as sd


@fixture  # type: ignore
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


@fixture
def variable_table():
    variable_tsv = "tests/data/variable.tsv"
    variable_table = pd.read_csv(variable_tsv, sep="\t")
    return variable_table


@fixture
def variable_reader():
    variable_tsv = "tests/data/variable.tsv"
    variable_reader = sd.Table(
        name="variable_seq",
        tables=variable_tsv,
        seq_col="seq",
        batch_size=50
    )
    return variable_reader

@fixture
def fixed_table():
    fixed_tsv = "tests/data/fixed.tsv"
    fixed_table = pd.read_csv(fixed_tsv, sep="\t")
    return fixed_table


@fixture
def fixed_reader():
    fixed_tsv = "tests/data/fixed.tsv"
    fixed_reader = sd.Table(
        name="fixed_seq",
        tables=fixed_tsv,
        seq_col="seq",
        batch_size=50
    )
    return fixed_reader


@fixture
def combo_table():
    tsvs = ["tests/data/variable.tsv", "tests/data/fixed.tsv"]
    combo_table = pd.concat([pd.read_csv(tsv, sep="\t") for tsv in tsvs])
    return combo_table


@fixture
def combo_reader():
    tsvs = ["tests/data/variable.tsv", "tests/data/fixed.tsv"]
    combo_reader = sd.Table(
        name="seq",
        tables=tsvs,
        seq_col="seq",
        batch_size=50
    )
    return combo_reader


def test_variable_write(
    variable_reader,
    variable_table,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        variable_reader._write(
            out=out,      
            fixed_length=False,
            sequence_dim="_sequence",
            overwrite=True
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = ds["variable_seq"].values.astype(str)
        targets = ds["target"].values
        np.testing.assert_array_equal(seqs, variable_table["seq"])
        np.testing.assert_almost_equal(targets, variable_table["target"])


def test_fixed_write(
    fixed_reader,
    fixed_table
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        fixed_reader._write(
            out=out,
            fixed_length=20,
            sequence_dim="_sequence",
            length_dim="_length",
            overwrite=True
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = [''.join(row.astype(str)) for row in ds["fixed_seq"].values]
        targets = ds["target"].values
        np.testing.assert_array_equal(seqs, fixed_table["seq"])
        np.testing.assert_almost_equal(targets, fixed_table["target"])


def test_combo_write(
    combo_reader,
    combo_table
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        combo_reader._write(
            out=out,
            fixed_length=False,
            sequence_dim="_sequence",
            overwrite=True
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = ds["seq"].values.astype(str)
        targets = ds["target"].values
        np.testing.assert_array_equal(seqs, combo_table["seq"])
        np.testing.assert_almost_equal(targets, combo_table["target"])


def test_from_flat_files(
        variable_reader,
        fixed_reader,
        variable_table,
        fixed_table,
    ):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.from_flat_files(
            variable_reader,
            fixed_reader,
            path=out,
            fixed_length=False,
            sequence_dim="_sequence",
            overwrite=True
        )

        variable_seqs = ds["variable_seq"].values.astype(str)
        fixed_seqs = ds["fixed_seq"].values.astype(str)
        np.testing.assert_array_equal(variable_seqs, variable_table["seq"])
        np.testing.assert_array_equal(fixed_seqs, fixed_table["seq"])
        

def test_read_table(
    combo_table,
):
    tsvs = ["tests/data/variable.tsv", "tests/data/fixed.tsv"]
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.read_table(
            tables=tsvs,
            seq_col="seq",
            name="seq",
            out=out,
            fixed_length=False,
            batch_size=50,
            overwrite=True
        )
        seqs = ds["seq"].values.astype(str)
        targets = ds["target"].values
        np.testing.assert_array_equal(seqs, combo_table["seq"])
        np.testing.assert_almost_equal(targets, combo_table["target"])
