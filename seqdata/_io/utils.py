from pathlib import Path
from typing import Callable, Generator, List, Optional

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import zarr
from numcodecs import VLenUTF8
from numpy.typing import NDArray

from seqdata.alphabets import SequenceAlphabet
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


def _set_uniform_length_around_center(bed: pd.DataFrame, length: int):
    if "peak" in bed:
        center = bed["chromStart"] + bed["peak"]
    else:
        center = (bed["chromStart"] + bed["chromEnd"]) / 2
    bed["chromStart"] = (center - length / 2).round().astype(np.uint64)
    bed["chromEnd"] = bed["chromStart"] + length


def _df_to_xr_zarr(df: pd.DataFrame, root: zarr.Group, dims: List[str], **kwargs):
    for name, series in df.items():
        data = series.to_numpy()
        if data.dtype.type == np.object_:
            object_codec = VLenUTF8()
        else:
            object_codec = None
        arr = root.array(name, data, object_codec=object_codec, **kwargs)
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


def _batch_io_bed(
    sink: zarr.Array,
    batch: NDArray[DTYPE],
    reader: Generator[T, None, None],
    write_row_to_batch: Callable[[NDArray[DTYPE], T], None],
    write_batch_to_sink: Callable[[zarr.Array, NDArray[DTYPE], int], None],
    to_rc: NDArray[np.bool_],
    alphabet: Optional[SequenceAlphabet] = None,
):
    batch_size = len(batch)
    start_idx = 0
    idx = 0
    for row in reader:
        write_row_to_batch(batch[idx], row)
        idx += 1
        if idx == batch_size:
            batch_to_rc = to_rc[start_idx : start_idx + batch_size]
            if batch.dtype.type == np.bytes_:
                batch[batch_to_rc] = complement_bytes(batch[batch_to_rc], alphabet)  # type: ignore
            batch[batch_to_rc] = np.flip(batch[batch_to_rc], 1)
            write_batch_to_sink(sink, batch, start_idx)
            start_idx += batch_size
            idx = 0
    if idx != 0:
        batch = batch[:idx]
        batch_to_rc = to_rc[start_idx:]
        if batch.dtype.type == np.bytes_:
            batch[batch_to_rc] = complement_bytes(batch[batch_to_rc], alphabet)  # type: ignore
        batch[batch_to_rc] = np.flip(batch[batch_to_rc], 1)
        write_batch_to_sink(sink, batch, start_idx)


def bytes_to_ohe(
    arr: NDArray[np.bytes_], alphabet: SequenceAlphabet
) -> NDArray[np.uint8]:
    idx = alphabet.sorter[np.searchsorted(alphabet.array[alphabet.sorter], arr)]
    ohe = np.eye(len(alphabet.array), dtype=np.uint8)[idx]
    return ohe


def ohe_to_bytes(
    ohe_arr: NDArray[np.uint8], alphabet: SequenceAlphabet, ohe_axis=-1
) -> NDArray[np.bytes_]:
    # ohe_arr shape: (... alphabet)
    idx = ohe_arr.nonzero()[-1]
    if ohe_axis < 0:
        ohe_axis_idx = len(ohe_arr.shape) + ohe_axis
    else:
        ohe_axis_idx = ohe_axis
    shape = tuple(dim for i, dim in enumerate(ohe_arr.shape) if i != ohe_axis_idx)
    # (regs samples ploidy length)
    return alphabet.array[idx].reshape(shape)


def complement_bytes(
    byte_arr: NDArray[np.bytes_], alphabet: SequenceAlphabet
) -> NDArray[np.bytes_]:
    """Get reverse complement of byte (string) array.

    Parameters
    ----------
    byte_arr : ndarray[bytes]
        Array of shape (regions [samples] [ploidy] length) to complement.
    complement_map : dict[bytes, bytes]
        Dictionary mapping nucleotides to their complements.
    """
    # NOTE: a vectorized implementation using np.unique is not faster even for IUPAC DNA/RNA.
    out = np.empty_like(byte_arr)
    for nuc, comp in alphabet.complement_map_bytes.items():
        if nuc == b"N":
            continue
        out[byte_arr == nuc] = comp
    return out


def rev_comp_byte(
    byte_arr: NDArray[np.bytes_], alphabet: SequenceAlphabet
) -> NDArray[np.bytes_]:
    """Get reverse complement of byte (string) array.

    Parameters
    ----------
    byte_arr : ndarray[bytes]
        Array of shape (regions [samples] [ploidy] length) to complement.
    complement_map : dict[bytes, bytes]
        Dictionary mapping nucleotides to their complements.
    """
    out = complement_bytes(byte_arr, alphabet)
    return out[..., ::-1]


def rev_comp_ohe(ohe_arr: NDArray[np.uint8], has_N: bool) -> NDArray[np.uint8]:
    if has_N:
        np.concatenate(
            [np.flip(ohe_arr[..., :-1], -1), ohe_arr[..., -1][..., None]],
            axis=-1,
            out=ohe_arr,
        )
    else:
        ohe_arr = np.flip(ohe_arr, -1)
    return np.flip(ohe_arr, -2)
