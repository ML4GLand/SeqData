import os
import pytest
import pandas as pd
import numpy as np
import zarr
from seqdata import Table, FlatFASTA, GenomeFASTA, BigWig, BAM

# Temporary directory for test outputs
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


# Test instantiation of each reader class
def test_table_instantiation():
    table_reader = Table(
        name="seq",
        tables=["tests/data/sample100.tsv"],
        seq_col="seq",
        batch_size=10,
    )
    assert isinstance(table_reader, Table), "Table reader instantiation failed."


def test_flat_fasta_instantiation():
    fasta_reader = FlatFASTA(
        name="seq",
        fasta="tests/data/tangermeme.fa",
        batch_size=5,
        n_threads=2,
    )
    assert isinstance(fasta_reader, FlatFASTA), "FlatFASTA reader instantiation failed."


def test_genome_fasta_instantiation():
    genome_reader = GenomeFASTA(
        name="seq",
        fasta="tests/data/tangermeme.fa",
        batch_size=10,
        n_threads=2,
    )
    assert isinstance(genome_reader, GenomeFASTA), "GenomeFASTA reader instantiation failed."


def test_bigwig_instantiation():
    bigwig_reader = BigWig(
        name="coverage",
        bigwigs=["tests/data/tangermeme.bw"],
        samples=["sample1"],
        batch_size=10,
        n_jobs=1,
        threads_per_job=1,
    )
    assert isinstance(bigwig_reader, BigWig), "BigWig reader instantiation failed."


def test_bam_instantiation():
    bam_reader = BAM(
        name="coverage",
        bams=["tests/data/sample.sorted.bam"],
        samples=["sample1"],
        batch_size=10,
        n_jobs=1,
        threads_per_job=1,
    )
    assert isinstance(bam_reader, BAM), "BAM reader instantiation failed."


# Test _reader functionality (example for Table)
def test_table_reader(temp_dir):
    table_reader = Table(
        name="seq",
        tables=["tests/data/sample100.tsv"],
        seq_col="seq",
        batch_size=10,
    )
    reader = table_reader._get_reader(table_reader.tables[0])
    first_batch = next(reader)
    assert "seq" in first_batch.columns, "Sequence column missing in first batch."
    assert len(first_batch) <= 10, "Batch size exceeds limit."


# Test _write functionality (example for Table)
def test_table_write(temp_dir):
    table_reader = Table(
        name="seq",
        tables=["tests/data/sample100.tsv"],
        seq_col="seq",
        batch_size=10,
    )
    output_path = os.path.join(temp_dir, "test_table.zarr")
    table_reader._write(out=output_path, overwrite=True, fixed_length=False, sequence_dim="_sequence")

    z = zarr.open_group(output_path, mode="r")
    assert "seq" in z, "Sequence data missing in Zarr output."
    assert z["seq"].shape[0] > 0, "No sequences written to Zarr output."


# Similar _reader and _write tests can be written for other classes

def test_flat_fasta_reader(temp_dir):
    # Implement a test for FlatFASTA._reader functionality
    pass

def test_genome_fasta_reader(temp_dir):
    # Implement a test for GenomeFASTA._reader functionality
    pass

def test_bigwig_reader(temp_dir):
    # Implement a test for BigWig._reader functionality
    pass

def test_bam_reader(temp_dir):
    # Implement a test for BAM._reader functionality
    pass

# Additional tests for edge cases or specific behaviors as identified
