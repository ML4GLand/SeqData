import os
import pytest
import seqdata as sd
from xarray import Dataset
from numpy.testing import assert_array_almost_equal

# Temporary directory for test outputs
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_read_table(temp_dir):
    """
    Test the read_table function.
    """
    # Input and output paths
    input_file = "tests/data/sample100.tsv"  # Replace with the actual table file path
    output_path = os.path.join(temp_dir, "test_table.zarr")

    # Parameters
    seq_col = "seq"  # Adjust based on the actual column name in your data
    batch_size = 10
    fixed_length = False

    # Call the function
    result = sd.read_table(
        name="seq",
        out=output_path,
        tables=input_file,
        seq_col=seq_col,
        batch_size=batch_size,
        fixed_length=fixed_length,
        overwrite=True,
    )

    # Assertions to validate the result
    assert isinstance(result, Dataset), "Result should be an XArray dataset"
    assert "seq" in result, "Missing 'seq' variable in the result"
    assert "_sequence" in result.dims, "Missing '_sequence' dimension in the result"
    assert result.dims["_sequence"] > 0, "No sequences found in the result"
    assert os.path.exists(output_path), "Output Zarr store was not created"

    # Check sequence and target integrity
    expected_sequence = "TGCATTTTTTTCACATCTCAAGCCGCGACACTGCTTTTGCTCCTTGGTGCAGTTCAGAGCTTCGGTTTAGATCGCGGTTGCTCTTATTTTAGAGGTAGGTTACGGCTGTT"
    assert str(result.seq[0].values.astype("U")) == expected_sequence, "First sequence does not match expected"
    assert_array_almost_equal(result.target[0].values, 11, decimal=6)


def test_read_flat_fasta(temp_dir):
    """
    Test the read_flat_fasta function.
    """
    # Input and output paths
    input_file = "tests/data/tangermeme.fa"  # Replace with your actual FASTA file path
    output_path = os.path.join(temp_dir, "test_flat_fasta.zarr")

    # Parameters
    name = "seq"
    batch_size = 5
    fixed_length = False
    n_threads = 2

    # Call the function
    result = sd.read_flat_fasta(
        name=name,
        out=output_path,
        fasta=input_file,
        batch_size=batch_size,
        fixed_length=fixed_length,
        n_threads=n_threads,
        overwrite=True,
    )

    # Assertions to validate the result
    assert isinstance(result, Dataset), "Result should be an XArray dataset"
    assert name in result, f"Missing '{name}' variable in the result"
    assert "_sequence" in result.dims, "Missing '_sequence' dimension in the result"
    assert result.dims["_sequence"] > 0, "No sequences found in the result"
    assert os.path.exists(output_path), "Output Zarr store was not created"

    # Check a specific sequence value for integrity
    expected_sequence = (
        "CGACTACTACCGACTAACTGACTGATGATGATGCATGCTGATGCTGAACTGACTAGCACTGCATGACTGATGACTGACTG"
        "TACTCCTACCATGACTATCCTAGTGCTGACCTGACTGATGCTGACTGACTGCATATGCACTGACTGACTCTACATGACTG"
        "ACTCACTCATCTGACATATCCATGCTGCATACTCATGATCATGCATGCATCATACTCATGCATGACTGACTCATGATGCA"
        "CATACTACTGCAGTCTGCATCATGCATGCATGCATGCACATCAT"
    )
    assert str(result.seq[0].values.astype("U")) == expected_sequence, "First sequence does not match expected"


def test_read_genome_fasta(temp_dir):
    """
    Test the read_genome_fasta function.
    """
    # Input and output paths
    fasta_file = "tests/data/tangermeme.fa"  # Replace with your actual genome FASTA file path
    bed_file = "tests/data/tangermeme.bed"  # Replace with your actual BED file path
    output_path = os.path.join(temp_dir, "test_genome_fasta.zarr")

    # Parameters
    name = "seq"
    batch_size = 10
    fixed_length = False
    n_threads = 2

    # Call the function
    result = sd.read_genome_fasta(
        name=name,
        out=output_path,
        fasta=fasta_file,
        bed=bed_file,
        batch_size=batch_size,
        fixed_length=fixed_length,
        n_threads=n_threads,
        overwrite=True,
    )

    # Assertions to validate the result
    assert isinstance(result, Dataset), "Result should be an XArray dataset"
    assert name in result, f"Missing '{name}' variable in the result"
    assert "_sequence" in result.dims, "Missing '_sequence' dimension in the result"
    assert result.dims["_sequence"] == 5, "Unexpected number of sequences extracted"  # Update based on your BED file
    assert os.path.exists(output_path), "Output Zarr store was not created"

    # Validate extracted sequences against expected values
    expected_sequences = [
        "CGACTACTACCGACTAACTG",  # Sequence for region 10-30
        "TACTGACTGTACTCCTACCA",  # Sequence for region 80-100
        "TACCTGACTGACTGCATATG",  # Sequence for region 140-160
    ]
    for i, expected in enumerate(expected_sequences):
        extracted_seq = str(result.seq[i].values.astype("U"))
        assert extracted_seq == expected, f"Sequence mismatch for region {i+1}"


def test_read_genome_fasta_fixed_length(temp_dir):
    """
    Test the read_genome_fasta function with fixed-length regions of length 10.
    Validates that regions are extended equally around their midpoints.
    """
    # Input and output paths
    fasta_file = "tests/data/tangermeme.fa"  # Replace with your actual genome FASTA file path
    bed_file = "tests/data/tangermeme.bed"  # Replace with your actual BED file path
    output_path = os.path.join(temp_dir, "test_genome_fasta_fixed_length.zarr")

    # Parameters
    name = "seq"
    batch_size = 10
    fixed_length = 10  # Fixed length of 10
    n_threads = 2

    # Call the function
    result = sd.read_genome_fasta(
        name=name,
        out=output_path,
        fasta=fasta_file,
        bed=bed_file,
        batch_size=batch_size,
        fixed_length=fixed_length,
        n_threads=n_threads,
        overwrite=True,
    )

    # Assertions to validate the result
    assert isinstance(result, Dataset), "Result should be an XArray dataset"
    assert name in result, f"Missing '{name}' variable in the result"
    assert "_sequence" in result.dims, "Missing '_sequence' dimension in the result"
    assert result.dims["_sequence"] > 0, "No sequences found in the result"
    assert os.path.exists(output_path), "Output Zarr store was not created"

    # Validate fixed-length sequences against expected values
    expected_sequences = [
        "ACTAACTGAC",  # chr1: 10-30, midpoint adjusted to length 10
        "CCTACCATGA",  # chr1: 80-100, midpoint adjusted to length 10
        "ACTGACTCTA",  # chr1: 140-160, midpoint adjusted to length 10
    ]
    for i, expected in enumerate(expected_sequences):
        extracted_seq = str(result.seq[i].values.astype("U10"))
        assert extracted_seq == expected, f"Mismatch for fixed-length sequence {i+1}: {extracted_seq}"


def test_read_bam(temp_dir):
    """
    Test the read_bam function.
    """
    # Input and output paths
    fasta_file = "tests/data/tangermeme.fa"  # Replace with your actual genome FASTA file path
    bam_file = "tests/data/sample.sorted.bam"  # Replace with your actual BAM file path
    bed_file = "tests/data/tangermeme.bed"  # Replace with your actual BED file path
    output_path = os.path.join(temp_dir, "test_bam.zarr")

    # Parameters
    seq_name = "seq"
    cov_name = "coverage"
    bams = [bam_file]
    samples = ["sample1"]
    batch_size = 10
    fixed_length = 100  # Example: fixed length of 100
    n_jobs = 1
    threads_per_job = 1
    dtype = "uint16"

    # Call the function
    result = sd.read_bam(
        seq_name=seq_name,
        cov_name=cov_name,
        out=output_path,
        fasta=fasta_file,
        bams=bams,
        samples=samples,
        bed=bed_file,
        batch_size=batch_size,
        fixed_length=fixed_length,
        n_jobs=n_jobs,
        threads_per_job=threads_per_job,
        dtype=dtype,
        overwrite=True,
    )

    # Assertions to validate the result
    assert isinstance(result, Dataset), "Result should be an XArray dataset"
    assert seq_name in result, f"Missing '{seq_name}' variable in the result"
    assert cov_name in result, f"Missing '{cov_name}' variable in the result"
    assert "_sequence" in result.dims, "Missing '_sequence' dimension in the result"
    assert "_length" in result.dims, "Missing '_position' dimension in the result"
    assert result.dims["_sequence"] > 0, "No sequences found in the result"
    assert os.path.exists(output_path), "Output Zarr store was not created"


def test_read_bigwig(temp_dir):
    """
    Test the read_bigwig function.
    """
    # Input and output paths
    fasta_file = "tests/data/tangermeme.fa"  # Replace with your actual genome FASTA file path
    bigwig_file = "tests/data/tangermeme.bw"  # Replace with your actual bigWig file path
    bed_file = "tests/data/tangermeme.bed"  # Replace with your actual BED file path
    output_path = os.path.join(temp_dir, "test_bigwig.zarr")

    # Parameters
    seq_name = "seq"
    cov_name = "coverage"
    bigwigs = [bigwig_file]
    samples = ["sample1"]
    batch_size = 10
    fixed_length = 100  # Example: fixed length of 100
    n_jobs = 1
    threads_per_job = 1

    # Call the function
    result = sd.read_bigwig(
        seq_name=seq_name,
        cov_name=cov_name,
        out=output_path,
        fasta=fasta_file,
        bigwigs=bigwigs,
        samples=samples,
        bed=bed_file,
        batch_size=batch_size,
        fixed_length=fixed_length,
        n_jobs=n_jobs,
        threads_per_job=threads_per_job,
        overwrite=True,
    )

    # Assertions to validate the result
    assert isinstance(result, Dataset), "Result should be an XArray dataset"
    assert seq_name in result, f"Missing '{seq_name}' variable in the result"
    assert cov_name in result, f"Missing '{cov_name}' variable in the result"
    assert "_sequence" in result.dims, "Missing '_sequence' dimension in the result"
    assert "_length" in result.dims, "Missing '_length' dimension in the result"
    assert result.dims["_sequence"] > 0, "No sequences found in the result"
    assert os.path.exists(output_path), "Output Zarr store was not created"
