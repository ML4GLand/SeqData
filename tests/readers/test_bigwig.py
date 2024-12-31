from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pandas as pd
import seqpro as sp
import xarray as xr
from pytest import fixture
import pickle

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

@fixture
def variable_coverage():
    variable_coverage = "tests/data/variable.bedcov.pkl"
    variable_coverage = pickle.load(open(variable_coverage, 'rb'))
    variable_coverage = {k.replace(".bam", ".bw"): v for k, v in variable_coverage.items()}
    return variable_coverage


@fixture
def fixed_coverage(variable_coverage, fixed_bed):
    fixed_coverage = {}
    bws = variable_coverage.keys()
    for bw in bws:
        fixed_coverage[bw] = {}
        for i, (region, coverage) in enumerate(variable_coverage[bw].items()):
            coverage_interval = region.split(":")[1]
            coverage_start, coverage_end = map(int, coverage_interval.split("-"))
            start_offset = coverage_start - fixed_bed[1].values[i]
            end_offset = fixed_bed[2].values[i] - coverage_end
            new_region = f"{fixed_bed[0].values[i]}:{fixed_bed[1].values[i]}-{fixed_bed[2].values[i]}"
            if end_offset == 0:
                fixed_coverage[bw][new_region] = coverage[start_offset:]
            else:
                fixed_coverage[bw][new_region] = coverage[start_offset:end_offset]
    return fixed_coverage
    

@fixture
def single_reader():
    bw = "tests/data/simulated1.bw"
    single_reader = sd.BigWig(
        name="cov",
        bigwigs=[bw],
        samples=["simulated1.bw"],
        batch_size=50
    )
    return single_reader


@fixture
def multi_reader():
    bws = [f"tests/data/simulated{i}.bw" for i in range(1, 6)]
    multi_reader = sd.BigWig(
        name="cov",
        bigwigs=bws,
        samples=[f"simulated{i}.bw" for i in range(1, 6)],
        batch_size=50
    )
    return multi_reader


def test_single_bigwig_write(
    single_reader,
    fixed_bed,
    fixed_coverage,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        fixed_bed["strand"] = "+"
        single_reader._write(
            out=out,
            bed=fixed_bed,
            fixed_length=20,
            sequence_dim="_sequence",
            length_dim="_length",
            overwrite=True,
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        cov = ds.sel(cov_sample="simulated1.bw").cov.values
        for i in range(len(cov)):
            np.testing.assert_array_equal(cov[i], list(fixed_coverage["simulated1.bw"].values())[i])


def test_multi_bigwig_write(
    multi_reader,
    fixed_bed,
    fixed_coverage,
):
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        fixed_bed["strand"] = "+"
        multi_reader._write(
            out=out,
            bed=fixed_bed,
            fixed_length=20,
            sequence_dim="_sequence",
            length_dim="_length",
            overwrite=True,
        )
        zarr.consolidate_metadata(out)
        ds = sd.open_zarr(out)
        for i in range(1, 6):
            cov = ds.sel(cov_sample=f"simulated{i}.bw").cov.values
            for j in range(len(cov)):
                np.testing.assert_array_equal(cov[j], list(fixed_coverage[f"simulated{i}.bw"].values())[j])


def test_from_region_files(
    fasta_reader,
    multi_reader,
    fasta,
    fixed_coverage,
):
    fixed_bed = "tests/data/fixed.bed"
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.from_region_files(
            fasta_reader,
            multi_reader,
            path=out,
            bed=fixed_bed,
            fixed_length=20,
            sequence_dim="_sequence",
            overwrite=True
        )
        bed = pd.read_csv(fixed_bed, sep="\t", header=None)
        zarr.consolidate_metadata(out)
        seqs = [''.join(row.astype(str)) for row in ds["seq"].values]
        true_seqs = [fasta[chrom][start:end] for chrom, start, end in bed.values]
        np.testing.assert_array_equal(seqs, true_seqs)
        for i in range(1, 6):
            cov = ds.sel(cov_sample=f"simulated{i}.bw").cov.values
            for j in range(len(cov)):
                np.testing.assert_array_equal(cov[j], list(fixed_coverage[f"simulated{i}.bw"].values())[j])


def test_read_bigwig(fixed_coverage):
    fasta = "tests/data/variable.fa"
    fixed_bed = "tests/data/fixed.bed"
    bws = [f"tests/data/simulated{i}.bw" for i in range(1, 6)]
    with (
        TemporaryDirectory(suffix=".zarr") as out,
    ):
        ds = sd.read_bigwig(
            seq_name="seq",
            cov_name="cov",
            out=out,
            fasta=fasta,
            bigwigs=bws,
            samples=[f"simulated{i}.bw" for i in range(1, 6)],
            bed=fixed_bed,
            batch_size=50,
            fixed_length=20,
            overwrite=True
        )
        bed = pd.read_csv(fixed_bed, sep="\t", header=None)
        zarr.consolidate_metadata(out)
        seqs = [''.join(row.astype(str)) for row in ds["seq"].values]
        fasta = read_fasta(fasta)
        true_seqs = [fasta[chrom][start:end] for chrom, start, end in bed.values]
        np.testing.assert_array_equal(seqs, true_seqs)
        for i in range(1, 6):
            cov = ds.sel(cov_sample=f"simulated{i}.bw").cov.values
            for j in range(len(cov)):
                np.testing.assert_array_equal(cov[j], list(fixed_coverage[f"simulated{i}.bw"].values())[j])
        