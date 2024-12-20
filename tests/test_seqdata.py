import os
import pytest
import xarray as xr
import numpy as np
import zarr
from seqdata import open_zarr, to_zarr, from_flat_files, from_region_files
from seqdata import FlatFASTA, GenomeFASTA, BigWig, BAM


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mock_xarray_dataset():
    """Mock an xarray dataset."""
    data = np.random.rand(10, 5)
    coords = {"sequence": np.arange(10), "length": np.arange(5)}
    dataset = xr.Dataset({"example_var": (["sequence", "length"], data)}, coords=coords)
    return dataset


def test_open_zarr(temp_dir, mock_xarray_dataset):
    """Test opening a Zarr store."""
    store_path = os.path.join(temp_dir, "test_open.zarr")
    mock_xarray_dataset.to_zarr(store_path)

    opened_ds = open_zarr(store=store_path)
    assert isinstance(opened_ds, xr.Dataset), "open_zarr did not return a Dataset"
    assert "example_var" in opened_ds.data_vars, "Variable missing in opened Dataset"
    assert "_ARRAY_DIMENSIONS" in opened_ds.attrs, "Attributes not set correctly"


def test_to_zarr(temp_dir, mock_xarray_dataset):
    """Test writing a Dataset to Zarr."""
    store_path = os.path.join(temp_dir, "test_to.zarr")
    to_zarr(mock_xarray_dataset, store=store_path, mode="w", consolidated=True)

    # Validate Zarr store content
    zarr_group = zarr.open_group(store_path)
    assert "example_var" in zarr_group, "Variable missing in Zarr store"
    assert "_ARRAY_DIMENSIONS" in zarr_group.attrs, "Attributes missing in Zarr store"


def test_from_flat_files(temp_dir):
    """Test from_flat_files functionality."""
    output_path = os.path.join(temp_dir, "test_flat_files.zarr")

    # Mock FlatReader
    fasta_reader = FlatFASTA(
        name="seq",
        fasta="tests/data/tangermeme.fa",
        batch_size=5,
        n_threads=2,
    )

    ds = from_flat_files(
        fasta_reader,
        path=output_path,
        fixed_length=True,
        overwrite=True,
    )

    assert isinstance(ds, xr.Dataset), "from_flat_files did not return a Dataset"
    assert "seq" in ds, "Expected variable missing in the Dataset"


def test_from_region_files(temp_dir):
    """Test from_region_files functionality."""
    output_path = os.path.join(temp_dir, "test_region_files.zarr")

    # Mock RegionReader
    genome_reader = GenomeFASTA(
        name="seq",
        fasta="tests/data/tangermeme.fa",
        batch_size=10,
        n_threads=2,
    )

    # Use a BED file or mock dataframe
    bed_path = "tests/data/tangermeme.bed"

    ds = from_region_files(
        genome_reader,
        path=output_path,
        fixed_length=100,
        bed=bed_path,
        overwrite=True,
    )

    assert isinstance(ds, xr.Dataset), "from_region_files did not return a Dataset"
    assert "seq" in ds, "Expected variable missing in the Dataset"
    assert ds.dims["_sequence"] > 0, "No sequences found in the Dataset"


# Edge Cases and Additional Tests

def test_open_zarr_empty_store(temp_dir):
    """Test opening an empty Zarr store."""
    store_path = os.path.join(temp_dir, "empty.zarr")
    os.makedirs(store_path)
    with pytest.raises(ValueError):
        open_zarr(store=store_path)


def test_to_zarr_overwrite(temp_dir, mock_xarray_dataset):
    """Test overwriting an existing Zarr store."""
    store_path = os.path.join(temp_dir, "overwrite.zarr")
    mock_xarray_dataset.to_zarr(store_path)
    mock_xarray_dataset["example_var"] *= 2  # Modify the data

    to_zarr(mock_xarray_dataset, store=store_path, mode="w", consolidated=True)

    opened_ds = open_zarr(store=store_path)
    np.testing.assert_array_equal(
        opened_ds["example_var"].values, mock_xarray_dataset["example_var"].values
    ), "Overwrite did not correctly update the Zarr store"


def test_from_region_files_with_jitter(temp_dir):
    """Test from_region_files with max_jitter."""
    output_path = os.path.join(temp_dir, "region_files_with_jitter.zarr")
    genome_reader = GenomeFASTA(
        name="seq",
        fasta="tests/data/tangermeme.fa",
        batch_size=10,
        n_threads=2,
    )
    bed_path = "tests/data/tangermeme.bed"

    ds = from_region_files(
        genome_reader,
        path=output_path,
        fixed_length=100,
        bed=bed_path,
        max_jitter=10,
        overwrite=True,
    )

    assert "seq" in ds, "Expected variable missing in the Dataset"
    assert ds.attrs["max_jitter"] == 10, "max_jitter attribute not set correctly"
