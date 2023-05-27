from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd
import pandera as pa
import pandera.typing as pat
import polars as pl
import xarray as xr
from pybedtools import BedTool

from seqdata.types import PathType


def _set_uniform_length_around_center(bed: pd.DataFrame, length: int):
    if "peak" in bed:
        center = bed["chromStart"] + bed["peak"]
    else:
        center = (bed["chromStart"] + bed["chromEnd"]) / 2
    bed["chromStart"] = (center - length / 2).round().astype(int)
    bed["chromEnd"] = bed["chromStart"] + length


def _expand_regions(bed: pd.DataFrame, expansion_length: int):
    bed["chromStart"] = bed["chromStart"] - expansion_length
    bed["chromEnd"] = bed["chromEnd"] + expansion_length


def add_bed_to_sdata(
    sdata: xr.Dataset,
    bed: pd.DataFrame,
    col_prefix: Optional[str] = None,
    sequence_dim: Optional[str] = None,
):
    if col_prefix is not None:
        bed.columns = [col_prefix + c for c in bed.columns]
    if sequence_dim is not None:
        bed.index.name = sequence_dim
    return sdata.merge(bed.to_xarray())


def mark_sequences_for_classification(
    sdata: xr.Dataset,
    targets: Union[pd.DataFrame, List[str]],
    mode: Literal["binary", "multitask"],
):
    """Mark sequences for binary or multitask classification based on whether they
    intersect with another set of regions.

    Parameters
    ----------
    sdata : xr.Dataset
    targets : Union[pd.DataFrame, List[str]]
        Either a DataFrame with at least columns ['chrom', 'chromStart', 'chromEnd',
        'name'], or a list of variable names in `sdata` to use that correspond to the
        ['chrom', 'chromStart', 'chromEnd', 'name'] columns, in that order. This is
        useful if, for example, another set of regions is already in the `sdata` object
        under a different set of column names.
    mode : Literal["binary", "multitask"]
        Whether to mark regions for binary (intersects with any of the target regions)
        or multitask classification (which target region does it intersect with?).
    """
    bed1 = BedTool.from_dataframe(
        sdata[["chrom", "chromStart", "chromEnd", "strand"]].to_dataframe()
    )

    if isinstance(targets, pd.DataFrame):
        bed2 = BedTool.from_dataframe(targets)
    elif isinstance(targets, list):
        bed2 = BedTool.from_dataframe(sdata[targets].to_dataframe())

    if mode == "binary":
        labels = (
            pl.read_csv(
                bed1.intersect(bed2, c=True).fn,  # type: ignore
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 3],
                new_columns=["chrom", "chromStart", "chromEnd", "label"],
            )
            .with_columns((pl.col("label") > 0).cast(pl.UInt8))["label"]
            .to_numpy()
        )
        sdata["label"] = xr.DataArray(labels, dims=sdata.attrs["sequence_dim"])
    elif mode == "multitask":
        labels = (
            pl.read_csv(
                bed1.intersect(bed2, loj=True).fn,  # type: ignore
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 6],
                new_columns=["chrom", "start", "end", "label"],
            )
            .to_dummies("label")
            .drop("label_.")
            .groupby("chrom", "start", "end", maintain_order=True)
            .agg(pl.exclude(r"^label.*$").first(), pl.col(r"^label.*$").max())
            .select(r"^label.*$")  # (sequences labels)
        )
        label_names = xr.DataArray(
            [c.split("_", 1)[1] for c in labels.columns], dims="_label"
        )
        sdata["label"] = xr.DataArray(
            labels.to_numpy(),
            coords=[label_names],
            dims=[sdata.attrs["sequence_dim"], "_label"],
        )


def read_bedlike(path: PathType) -> pd.DataFrame:
    """Reads a bed-like (BED3+) file as a pandas DataFrame. The file type is inferred
    from the file extension.

    Parameters
    ----------
    path : PathType

    Returns
    -------
    pandas.DataFrame
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
