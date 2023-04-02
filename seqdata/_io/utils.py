from pathlib import Path
from typing import Callable, Generator, List

import numpy as np
import pandas as pd
import pandera as pa
import zarr
from numpy.typing import NDArray

from seqdata.types import DTYPE, PathType, T


def _read_bedlike(path: PathType):
    path = Path(path)
    if path.suffix == ".bed":
        return _read_bed(path)
    elif path.suffix == ".narrowPeak":
        return _read_narrowpeak(path)
    elif path.suffix == ".broadPeak":
        return _read_broadpeak(path)
    else:
        raise ValueError(
            f"Unrecognized file extension: {path.suffix}. Expected one of .bed, .narrowPeak, or .broadPeak"
        )


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
    return bed


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
    return narrowpeaks


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
    return broadpeaks


def _set_uniform_length_around_center(bed: pd.DataFrame, length: int):
    if "peak" in bed:
        center = bed["chromStart"] + bed["peak"]
    else:
        center = (bed["chromStart"] + bed["chromEnd"]) / 2
    bed["chromStart"] = (center - length / 2).round().astype(np.uint64)
    bed["chromEnd"] = bed["chromStart"] + length


def _df_to_xr_zarr(df: pd.DataFrame, seqdata_path: PathType, dims: List[str], **kwargs):
    z = zarr.open_group(seqdata_path)
    for name, series in df.items():
        arr = z.array(name, series.to_numpy(), **kwargs)
        arr.attrs["_ARRAY_DIMENSIONS"] = dims


def _batch_io(
    sink: zarr.Array,
    batch: NDArray[DTYPE],
    reader: Generator[T, None, None],
    write_row_to_batch: Callable[[NDArray[DTYPE], T], None],
    write_batch_to_sink: Callable[[zarr.Array, NDArray[DTYPE], int], None],
):
    batch_size = len(batch)
    start_idx = 0
    idx = 0
    for row in reader:
        write_row_to_batch(batch[idx], row)
        idx += 1
        if idx == batch_size:
            write_batch_to_sink(sink, batch, start_idx)
            start_idx += batch_size
            idx = 0
    if idx != batch_size:
        write_batch_to_sink(sink, batch[:idx], start_idx)
