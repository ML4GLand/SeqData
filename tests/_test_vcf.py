<<<<<<<< HEAD:tests/experimental/test_vcf.py
import pytest  # noqa
from pytest import fixture, parametrize_with_cases
========
from pytest_cases import fixture, parametrize_with_cases
>>>>>>>> 508ebc4088fba492eaeec2bf59624d7aec2d68f0:tests/_test_vcf.py

from seqdata import read_vcf


@fixture
def vcf():
    raise NotImplementedError


@fixture
def reference():
    raise NotImplementedError


def consensus(sample):
    raise NotImplementedError


def bed_no_variants():
    raise NotImplementedError


def bed_variants():
    raise NotImplementedError


def length_variable():
    return None


def length_600():
    return 600


def samples_one():
    raise NotImplementedError


def samples_two():
    raise NotImplementedError


@parametrize_with_cases("bed", cases=".", prefix="bed_")
@parametrize_with_cases("samples", cases=".", prefix="samples_")
@parametrize_with_cases("length", cases=".", prefix="length_")
def test_fixed_length(vcf, reference, samples, bed, length):
    sdata = read_vcf(
        "vcf", "foo", vcf, reference, samples, bed, 1024, length, overwrite=True
    )  # noqa
    for region in sdata.obs[["contig", "start", "end"]].itertuples():
        for i, sample in enumerate(sdata.ds.coords["vcf_samples"]):
            pass
    # consensus_path =
    raise NotImplementedError


def test_variable_length():
    raise NotImplementedError


def test_spliced_fixed_length():
    raise NotImplementedError


def test_spliced_variable_length():
    raise NotImplementedError
