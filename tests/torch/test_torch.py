from typing import Dict

import numpy as np
import pytest
import seqpro as sp
import torch
import xarray as xr
from numpy.typing import NDArray
from pytest_cases import fixture

import seqdata as sd


@fixture
def dummy_dataset():
    """Dummy dataset with 3 sequences of length 5."""
    seqs = sp.random_seqs((3, 5), sp.DNA, 0)
    return xr.Dataset(
        {
            "seqs": xr.DataArray(seqs, dims=["_sequence", "_length"]),
        }
    )


def test_no_transforms(dummy_dataset: xr.Dataset):
    dl = sd.get_torch_dataloader(
        dummy_dataset, sample_dims="_sequence", variables=["seqs"]
    )
    # this should raise a TypeError: can't convert np.bytes_ to Tensor
    with pytest.raises(TypeError):
        next(iter(dl))


def test_ohe_transform(dummy_dataset: xr.Dataset):
    def transform(batch: Dict[str, NDArray]):
        batch["seqs"] = sp.DNA.ohe(batch["seqs"])
        return batch

    dl = sd.get_torch_dataloader(
        dummy_dataset,
        sample_dims="_sequence",
        variables=["seqs"],
        transform=transform,
        batch_size=2,
    )
    batch: Dict[str, torch.Tensor] = next(iter(dl))
    seqs: NDArray = batch["seqs"].numpy()
    ds_seqs = sp.DNA.ohe(dummy_dataset["seqs"].values)
    np.testing.assert_array_equal(seqs, ds_seqs[:2])
