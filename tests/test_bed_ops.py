from tempfile import NamedTemporaryFile

import polars as pl
import polars.testing as pl_testing
from pytest_cases import parametrize_with_cases

import seqdata as sd


def bed_bed3():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1, 2, 3],
            "chromEnd": [2, 3, 4],
        }
    )
    return bed


def bed_bed4():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1, 2, 3],
            "chromEnd": [2, 3, 4],
            "name": ["a", "b", "c"],
        }
    )
    return bed


def bed_bed5():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1, 2, 3],
            "chromEnd": [2, 3, 4],
            "name": ["a", "b", "c"],
            "score": [1.1, 2, 3],
        }
    )
    return bed


def narrowpeak_simple():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1, 2, 3],
            "chromEnd": [2, 3, 4],
            "name": ["a", "b", "c"],
            "score": [1.1, 2, 3],
            "strand": ["+", "-", None],
            "signalValue": [1.1, 2, 3],
            "pValue": [1.1, 2, 3],
            "qValue": [1.1, 2, 3],
            "peak": [1, 2, 3],
        }
    )
    return bed


def broadpeak_simple():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1, 2, 3],
            "chromEnd": [2, 3, 4],
            "name": ["a", "b", "c"],
            "score": [1.1, 2, 3],
            "strand": ["+", "-", None],
            "signalValue": [1.1, 2, 3],
            "pValue": [1.1, 2, 3],
            "qValue": [1.1, 2, 3],
        }
    )
    return bed


@parametrize_with_cases("bed", cases=".", prefix="bed_")
def test_read_bed(bed: pl.DataFrame):
    with NamedTemporaryFile(suffix=".bed") as f:
        bed.write_csv(f.name, include_header=False, separator="\t", null_value=".")
        bed2 = sd.read_bedlike(f.name)
        pl_testing.assert_frame_equal(bed, bed2)


@parametrize_with_cases("narrowpeak", cases=".", prefix="narrowpeak_")
def test_read_narrowpeak(narrowpeak: pl.DataFrame):
    with NamedTemporaryFile(suffix=".narrowPeak") as f:
        narrowpeak.write_csv(
            f.name, include_header=False, separator="\t", null_value="."
        )
        narrowpeak2 = sd.read_bedlike(f.name)
        pl_testing.assert_frame_equal(narrowpeak, narrowpeak2)


@parametrize_with_cases("broadpeak", cases=".", prefix="broadpeak_")
def test_read_broadpeak(broadpeak: pl.DataFrame):
    with NamedTemporaryFile(suffix=".broadPeak") as f:
        broadpeak.write_csv(
            f.name, include_header=False, separator="\t", null_value="."
        )
        broadpeak2 = sd.read_bedlike(f.name)
        pl_testing.assert_frame_equal(broadpeak, broadpeak2)
