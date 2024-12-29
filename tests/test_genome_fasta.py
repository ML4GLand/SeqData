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
def fasta():
    fasta = "tests/data/variable.fa"
    return read_fasta(fasta)


@fixture
def fasta_reader():
    fasta = "tests/data/variable.fa"
    reader = sd.GenomeFASTA(
        name="seq",
        fasta=fasta,
        batch_size=50
    )
    return reader


@fixture
def variable_bed():
    variable_bed = "tests/data/variable.bed"
    variable_bed = pd.read_csv(variable_bed, sep="\t", header=None)
    return variable_bed


@fixture
def fixed_bed():
    fixed_bed = "tests/data/fixed.bed"
    fixed_bed = pd.read_csv(fixed_bed, sep="\t", header=None)
    return fixed_bed


def test_variable_write(
    fasta_reader,
    fasta,
    variable_bed,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        true_seqs = [fasta[chrom][start:end] for chrom, start, end in variable_bed.values]
        variable_bed["strand"] = "+"
        fasta_reader._write(
            out=out,
            bed=variable_bed,
            fixed_length=False,
            sequence_dim="_sequence",
            overwrite=True
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = ds["seq"].values.astype(str)
        np.testing.assert_array_equal(seqs, true_seqs)


def test_fixed_write(
    fasta_reader,
    fasta,
    fixed_bed,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        true_seqs = [fasta[chrom][start:end] for chrom, start, end in fixed_bed.values]
        fixed_bed["strand"] = "+"
        fasta_reader._write(
            out=out,
            bed=fixed_bed,
            fixed_length=20,
            sequence_dim="_sequence",
            length_dim="_length",
            overwrite=True
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        seqs = [''.join(row.astype(str)) for row in ds["seq"].values]
        np.testing.assert_array_equal(seqs, true_seqs)


def test_from_region_files(
    fasta_reader,
    fasta,
):
    variable_bed = "tests/data/variable.bed"
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.from_region_files(
            fasta_reader,
            bed=variable_bed,
            path=out,
            fixed_length=False,
            sequence_dim="_sequence",
            overwrite=True
        )
        bed = pd.read_csv(variable_bed, sep="\t", header=None)
        true_seqs = [fasta[chrom][start:end] for chrom, start, end in bed.values]
        seqs = ds["seq"].values.astype(str)
        np.testing.assert_array_equal(seqs, true_seqs)


def test_read_genome_fasta():
    fasta = "tests/data/variable.fa"
    variable_bed = "tests/data/variable.bed"
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.read_genome_fasta(
            name="seq",
            out=out,
            fasta=fasta,
            bed=variable_bed,
            fixed_length=False,
            batch_size=50,
            overwrite=True
        )

        bed = pd.read_csv(variable_bed, sep="\t", header=None)
        true_seqs = [read_fasta(fasta)[chrom][start:end] for chrom, start, end in bed.values]
        seqs = ds["seq"].values.astype(str)
        np.testing.assert_array_equal(seqs, true_seqs)
