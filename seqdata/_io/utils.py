from typing import Callable, Generator, List, Optional

import numpy as np
import pandas as pd
import zarr
from numcodecs import VLenUTF8
from numpy.typing import NDArray

from seqdata.alphabets import SequenceAlphabet
from seqdata.types import DTYPE, T


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
