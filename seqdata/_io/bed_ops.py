import warnings
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
    label_dim: Optional[str] = None,
) -> xr.DataArray:
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
    label_dim : str, optional
        Name of the label dimension. Only needed for multitask classification.
    """
    bed1 = BedTool.from_dataframe(
        sdata[["chrom", "chromStart", "chromEnd", "strand"]].to_dataframe()
    )

    if isinstance(targets, pd.DataFrame):
        bed2 = BedTool.from_dataframe(targets)
    elif isinstance(targets, list):
        bed2 = BedTool.from_dataframe(sdata[targets].to_dataframe())

    if mode == "binary":
        if label_dim is not None:
            warnings.warn("Ignoring `label_dim` for binary classification.")
        res = bed1.intersect(bed2, c=True)  # type: ignore
        with open(res.fn) as f:
            n_cols = len(f.readline().split("\t"))
        labels = (
            pl.read_csv(
                res.fn,
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, n_cols - 1],
                new_columns=["chrom", "chromStart", "chromEnd", "label"],
            )
            .with_columns((pl.col("label") > 0).cast(pl.UInt8))["label"]
            .to_numpy()
        )
        return xr.DataArray(labels, dims=sdata.attrs["sequence_dim"])
    elif mode == "multitask":
        if label_dim is None:
            raise ValueError(
                """Need a name for the label dimension when generating labels for 
                multitask classification."""
            )
        res = bed1.intersect(bed2, loj=True)  # type: ignore
        labels = (
            pl.read_csv(
                res.fn,
                separator="\t",
                has_header=False,
                columns=[0, 1, 2, 7],
                new_columns=["chrom", "chromStart", "chromEnd", "label"],
            )
            .to_dummies("label")
            .select(pl.exclude(r"^label_\.$"))
            .groupby("chrom", "chromStart", "chromEnd", maintain_order=True)
            .agg(pl.exclude(r"^label.*$").first(), pl.col(r"^label.*$").max())
            .select(r"^label.*$")  # (sequences labels)
        )
        label_names = xr.DataArray(
            [c.split("_", 1)[1] for c in labels.columns], dims=label_dim
        )
        return xr.DataArray(
            labels.to_numpy(),
            coords={label_dim: label_names},
            dims=[sdata.attrs["sequence_dim"], label_dim],
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
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: Optional[pat.Series[str]] = pa.Field(nullable=True)
    score: Optional[pat.Series[float]] = pa.Field(nullable=True)
    strand: Optional[pat.Series[str]] = pa.Field(isin=["+", "-", "."], nullable=True)
    thickStart: Optional[pat.Series[int]] = pa.Field(nullable=True)
    thickEnd: Optional[pat.Series[int]] = pa.Field(nullable=True)
    itemRgb: Optional[pat.Series[str]] = pa.Field(nullable=True)
    blockCount: Optional[pat.Series[pa.UInt]] = pa.Field(nullable=True)
    blockSizes: Optional[pat.Series[str]] = pa.Field(nullable=True)
    blockStarts: Optional[pat.Series[str]] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_bed(bed_path: PathType):
    with open(bed_path) as f:
        skip_rows = 0
        while (line := f.readline()).startswith(("track", "browser")):
            skip_rows += 1
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
    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=bed_cols[:n_cols],
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    if "strand" not in bed:
        bed["strand"] = "+"
    bed = BEDSchema.to_schema()(bed)
    return bed


class NarrowPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str] = pa.Field(nullable=True)
    score: pat.Series[float] = pa.Field(nullable=True)
    strand: pat.Series[str] = pa.Field(isin=["+", "-", "."], nullable=True)
    signalValue: pat.Series[float] = pa.Field(nullable=True)
    pValue: pat.Series[float] = pa.Field(nullable=True)
    qValue: pat.Series[float] = pa.Field(nullable=True)
    peak: pat.Series[int] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_narrowpeak(narrowpeak_path: PathType) -> pd.DataFrame:
    with open(narrowpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    narrowpeaks = pl.read_csv(
        narrowpeak_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=[
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
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
    ).to_pandas()
    narrowpeaks = NarrowPeakSchema.to_schema()(narrowpeaks)
    return narrowpeaks


class BroadPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str] = pa.Field(nullable=True)
    score: pat.Series[float] = pa.Field(nullable=True)
    strand: pat.Series[str] = pa.Field(isin=["+", "-", "."], nullable=True)
    signalValue: pat.Series[float] = pa.Field(nullable=True)
    pValue: pat.Series[float] = pa.Field(nullable=True)
    qValue: pat.Series[float] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_broadpeak(broadpeak_path: PathType):
    with open(broadpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    broadpeaks = pl.read_csv(
        broadpeak_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=[
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
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
    ).to_pandas()
    broadpeaks = BroadPeakSchema.to_schema()(broadpeaks)
    return broadpeaks
