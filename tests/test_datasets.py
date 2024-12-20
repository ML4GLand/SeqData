import os
from pathlib import Path
import pytest
import shutil

from seqdata.datasets import K562_ATAC_seq, K562_CTCF_ChIP_seq

# Helper function to clean up extracted files
def cleanup_test_files(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def test_K562_ATAC_seq_seqdata():
    """Test fetching and extracting K562 ATAC-seq as SeqData."""
    seqdata_path = K562_ATAC_seq(type="seqdata")
    assert seqdata_path.exists(), "Extracted SeqData path should exist"
    assert seqdata_path.is_dir(), "SeqData path should be a directory"
    cleanup_test_files(seqdata_path)  # Clean up after test

def test_K562_ATAC_seq_bigwig():
    """Test fetching K562 ATAC-seq signal BigWig file."""
    bigwig_path = K562_ATAC_seq(type="bigwig")
    assert bigwig_path.exists(), "BigWig path should exist"
    assert bigwig_path.is_file(), "BigWig path should be a file"

def test_K562_CTCF_ChIP_seq_seqdata():
    """Test fetching and extracting K562 CTCF ChIP-seq as SeqData."""
    seqdata_path = K562_CTCF_ChIP_seq(type="seqdata")
    assert seqdata_path.exists(), "Extracted SeqData path should exist"
    assert seqdata_path.is_dir(), "SeqData path should be a directory"
    cleanup_test_files(seqdata_path)  # Clean up after test

def test_K562_CTCF_ChIP_seq_bigwig():
    """Test fetching K562 CTCF ChIP-seq signal BigWig files."""
    plus_bw, minus_bw = K562_CTCF_ChIP_seq(type="bigwig")
    assert plus_bw.exists(), "Plus strand BigWig file should exist"
    assert minus_bw.exists(), "Minus strand BigWig file should exist"
    assert plus_bw.is_file(), "Plus strand BigWig path should be a file"
    assert minus_bw.is_file(), "Minus strand BigWig path should be a file"

def test_datasets_cleanup():
    """Ensure datasets cleanup after fetching and extraction."""
    seqdata_path = K562_ATAC_seq(type="seqdata")
    tar_path = seqdata_path.parent / "K562_ATAC-seq.zarr.tar.gz"

    # Ensure tar file is removed after extraction
    assert not tar_path.exists(), "Tar.gz file should be removed after extraction"

    # Cleanup extracted data
    cleanup_test_files(seqdata_path)
