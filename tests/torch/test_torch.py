from typing import Dict

import numpy as np
import pytest
import seqpro as sp
import torch
import xarray as xr
from numpy.typing import NDArray
from pytest import fixture

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


@fixture
def multi_var_dataset():
    """Dataset with multiple variables."""
    seqs = sp.random_seqs((3, 5), sp.DNA, 0)
    scores = np.random.rand(3, 5)
    return xr.Dataset(
        {
            "seqs": xr.DataArray(seqs, dims=["_sequence", "_length"]),
            "scores": xr.DataArray(scores, dims=["_sequence", "_length"]),
        }
    )


def test_multi_variable(multi_var_dataset: xr.Dataset):
    dl = sd.get_torch_dataloader(
        multi_var_dataset,
        sample_dims="_sequence",
        variables=["seqs", "scores"],
        batch_size=2,
    )
    batch = next(iter(dl))
    assert "seqs" in batch, "seqs variable missing in batch"
    assert "scores" in batch, "scores variable missing in batch"
    assert batch["seqs"].shape == (2, 5), "Shape mismatch for seqs"
    assert batch["scores"].shape == (2, 5), "Shape mismatch for scores"


def test_shuffling(dummy_dataset: xr.Dataset):
    dl_shuffled = sd.get_torch_dataloader(
        dummy_dataset,
        sample_dims="_sequence",
        variables=["seqs"],
        shuffle=True,
        batch_size=3,
        seed=42,  # Ensure deterministic shuffling
    )
    dl_unshuffled = sd.get_torch_dataloader(
        dummy_dataset,
        sample_dims="_sequence",
        variables=["seqs"],
        shuffle=False,
        batch_size=3,
    )

    batch_shuffled = next(iter(dl_shuffled))
    batch_unshuffled = next(iter(dl_unshuffled))
    assert not np.array_equal(
        batch_shuffled["seqs"].numpy(), batch_unshuffled["seqs"].numpy()
    ), "Shuffled data should not match unshuffled data"


def test_return_tuples(dummy_dataset: xr.Dataset):
    dl = sd.get_torch_dataloader(
        dummy_dataset,
        sample_dims="_sequence",
        variables=["seqs"],
        return_tuples=True,
        batch_size=2,
    )
    batch = next(iter(dl))
    assert isinstance(batch, tuple), "Batch should be a tuple"
    assert len(batch) == 1, "Tuple should contain one item"
    assert isinstance(batch[0], torch.Tensor), "Tuple item should be a tensor"
