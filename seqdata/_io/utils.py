from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import zarr

PathType = Union[str, Path]


def _read_and_concat_dataframes(
    file_names: Union[PathType, List[PathType]],
    col_names: Optional[Union[str, List[str]]] = None,
    sep: str = "\t",
    low_memory: bool = False,
    compression: str = "infer",
    **kwargs,
) -> pd.DataFrame:
    """Reads a list of files and concatenates them into a single dataframe.

    Parameters
    ----------
    file_names : str or list
        Path to file or list of paths to files.
    col_names : str or list, optional
        Column names to use for the dataframe. If not provided, the column names will be the first row of the file.
    sep : str, optional
        Separator to use for the dataframe. Defaults to "\t".
    low_memory : bool, optional
        If True, the dataframe will be stored in memory. If False, the dataframe will be stored on disk. Defaults to False.
    compression : str, optional
        Compression to use for the dataframe. Defaults to "infer".
    **kwargs
        Additional arguments to pass to pd.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(file_names, list):
        file_names = [file_names]
    if not isinstance(col_names, list) and col_names is not None:
        col_names = [col_names]
    dfs = []
    for file_name in file_names:
        x = pd.read_csv(
            file_name,
            sep=sep,
            low_memory=low_memory,
            names=col_names,
            compression=compression,  # type: ignore
            header=0,
            **kwargs,
        )
        dfs.append(x)
    dataframe = pd.concat(dfs, ignore_index=True)
    return dataframe


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
