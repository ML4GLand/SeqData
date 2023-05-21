from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat

from seqdata.types import PathType


def _set_uniform_length_around_center(bed: pd.DataFrame, length: int):
    if "peak" in bed:
        center = bed["chromStart"] + bed["peak"]
    else:
        center = (bed["chromStart"] + bed["chromEnd"]) / 2
    bed["chromStart"] = (center - length / 2).round().astype(np.uint64)
    bed["chromEnd"] = bed["chromStart"] + length


def _expand_regions(bed: pd.DataFrame, expansion_length: int):
    bed["chromStart"] = bed["chromStart"] - expansion_length
    bed["chromEnd"] = bed["chromEnd"] + expansion_length


def _read_bedlike(path: PathType):
    """Reads a bed-like file as a pandas DataFrame. The file type is inferred from the
    file extension.

    Parameters
    ----------
    path : PathType
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    path = Path(path)
    if ".bed" in path.suffixes:
        return _read_bed(path)
    elif ".narrowPeak" in path.suffixes:
        return _read_narrowpeak(path)
    elif ".broadPeak" in path.suffixes:
        return _read_broadpeak(path)
    else:
        raise ValueError(
            f"""Unrecognized file extension: {''.join(path.suffixes)}. Expected one of 
            .bed, .narrowPeak, or .broadPeak"""
        )


class BEDSchema(pa.DataFrameModel):
    chrom: pat.Series[pa.Category]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: Optional[pat.Series[str]]
    score: Optional[pat.Series[float]]
    strand: Optional[pat.Series[pa.Category]] = pa.Field(isin=["+", "-", "."])
    thickStart: Optional[pat.Series[int]]
    thickEnd: Optional[pat.Series[int]]
    itemRgb: Optional[pat.Series[str]]
    blockCount: Optional[pat.Series[pa.UInt]]
    blockSizes: Optional[pat.Series[str]]
    blockStarts: Optional[pat.Series[str]]

    class Config:
        coerce = True


def _read_bed(bed_path: PathType):
    with open(bed_path) as f:
        while (line := f.readline()).startswith(("track", "browser")):
            continue
    n_cols = line.count("\t") + 1
    bed_cols = [
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "score",
        "strand",
        "thickStart",
        "thickEnd",
        "itemRgb",
        "blockCount",
        "blockSizes",
        "blockStarts",
    ]
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        skiprows=lambda x: x in ["track", "browser"],
        names=bed_cols[:n_cols],
        dtype={"chrom": str, "name": str},
    )
    if "strand" not in bed:
        bed["strand"] = "+"
    bed = BEDSchema.to_schema()(bed)
    return bed


class NarrowPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[pa.Category]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str]
    score: pat.Series[float]
    strand: pat.Series[pa.Category] = pa.Field(isin=["+", "-", "."])
    signalValue: pat.Series[float]
    pValue: pat.Series[float]
    qValue: pat.Series[float]
    peak: pat.Series[int]

    class Config:
        coerce = True


def _read_narrowpeak(narrowpeak_path: PathType) -> pd.DataFrame:
    narrowpeaks = pd.read_csv(
        narrowpeak_path,
        sep="\t",
        header=None,
        skiprows=lambda x: x in ["track", "browser"],
        names=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
            "peak",
        ],
        dtype={"chrom": str, "name": str},
    )
    narrowpeaks = NarrowPeakSchema.to_schema()(narrowpeaks)
    return narrowpeaks


class BroadPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[pa.Category]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str]
    score: pat.Series[float]
    strand: pat.Series[pa.Category] = pa.Field(isin=["+", "-", "."])
    signalValue: pat.Series[float]
    pValue: pat.Series[float]
    qValue: pat.Series[float]

    class Config:
        coerce = True


def _read_broadpeak(broadpeak_path: PathType):
    broadpeaks = pd.read_csv(
        broadpeak_path,
        sep="\t",
        header=None,
        skiprows=lambda x: x in ["track", "browser"],
        names=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
        ],
        dtype={"chrom": str, "name": str},
    )
    broadpeaks = BroadPeakSchema.to_schema()(broadpeaks)
    return broadpeaks
