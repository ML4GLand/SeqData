from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pandas as pd
import seqpro as sp
import xarray as xr
from pytest import fixture

import seqdata as sd
import zarr


def read_fasta(file_path):
    """
    Reads a FASTA file and returns a dictionary of sequences.

    Parameters:
        file_path (str): Path to the FASTA file.

    Returns:
        dict: A dictionary where keys are sequence IDs and values are sequences.
    """
    sequences = {}
    with open(file_path, 'r') as file:
        sequence_id = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # Save the previous sequence (if any) before starting a new one
                if sequence_id:
                    sequences[sequence_id] = ''.join(sequence_lines)
                # Start a new sequence
                sequence_id = line[1:]  # Remove the '>'
                sequence_lines = []
            else:
                # Append sequence lines
                sequence_lines.append(line)
        # Save the last sequence
        if sequence_id:
            sequences[sequence_id] = ''.join(sequence_lines)
    return sequences


@fixture
def variable_fasta():
    variable_fasta = "tests/data/variable.fa"
    return read_fasta(variable_fasta)


@fixture
def fixed_fasta():
    fixed_fasta = "tests/data/fixed.fa"
    return read_fasta(fixed_fasta)

@fixture
def variable_reader():
    variable_fasta = "tests/data/variable.fa"
    variable_reader = sd.FlatFASTA(
        name="variable_seq",
        fasta=variable_fasta,
        batch_size=50
    )
    return variable_reader


@fixture
def fixed_reader():
    fixed_fasta = "tests/data/fixed.fa"
    fixed_reader = sd.FlatFASTA(
        name="fixed_seq",
        fasta=fixed_fasta,
        batch_size=50
    )
    return fixed_reader


def test_variable_write(
    variable_reader,
    variable_fasta,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        variable_reader._write(
            out=out,
            fixed_length=False,
            sequence_dim="_sequence",
            overwrite=True,
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = ds["variable_seq"].values.astype(str)
        np.testing.assert_array_equal(seqs, list(variable_fasta.values()))


def test_fixed_write(
    fixed_reader,
    fixed_fasta,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        fixed_reader._write(
            out=out,
            fixed_length=20,
            sequence_dim="_sequence",
            length_dim="_length",
            overwrite=True,
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = [''.join(row.astype(str)) for row in ds["fixed_seq"].values]
        np.testing.assert_array_equal(seqs, list(fixed_fasta.values()))


def test_from_flat_files(
        variable_reader,
        fixed_reader,
        variable_fasta,
        fixed_fasta,
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
        np.testing.assert_array_equal(variable_seqs, list(variable_fasta.values()))
        np.testing.assert_array_equal(fixed_seqs, list(fixed_fasta.values()))


def test_read_flat_fasta(
    variable_fasta,
):
    variable_fasta = "tests/data/variable.fa"
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.read_flat_fasta(
            fasta=variable_fasta,
            out=out,
            name="seq",
            fixed_length=False,
            batch_size=50,
            overwrite=True
        )
        seqs = ds["seq"].values.astype(str)
        np.testing.assert_array_equal(seqs, list(read_fasta(variable_fasta).values()))
        