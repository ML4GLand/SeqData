import logging
from itertools import count, cycle
from subprocess import CalledProcessError, run
from textwrap import dedent
from typing import Generator, List, Literal, Tuple

import numpy as np
import pandas as pd
import zarr
from more_itertools import mark_ends, repeat_each
from numcodecs import VLenArray, VLenBytes, VLenUTF8
from numpy.typing import NDArray

from seqdata.alphabets import SequenceAlphabet
from seqdata.types import T


def _df_to_xr_zarr(df: pd.DataFrame, root: zarr.Group, dims: List[str], **kwargs):
    for name, series in df.items():
        data = series.to_numpy()
        if data.dtype.type == np.object_:
            if isinstance(data[0], np.ndarray):
                object_codec = VLenArray(data[0].dtype)
            elif isinstance(data[0], str):
                object_codec = VLenUTF8()
            elif isinstance(data[0], bytes):
                object_codec = VLenBytes()
            else:
                raise ValueError("Got column in dataframe that isn't serializable.")
        else:
            object_codec = None
        arr = root.array(name, data, object_codec=object_codec, **kwargs)
        arr.attrs["_ARRAY_DIMENSIONS"] = dims


def _get_row_batcher(
    reader: Generator[T, None, None], batch_size: int
) -> Generator[Tuple[bool, bool, T, int, int], None, None]:
    batch_idxs = cycle(mark_ends(range(batch_size)))
    start_idxs = repeat_each(count(0, batch_size), batch_size)
    for row_info, batch_info, start_idx in zip(
        mark_ends(reader), batch_idxs, start_idxs
    ):
        first_row, last_row, row = row_info
        first_in_batch, last_in_batch, batch_idx = batch_info
        yield last_row, last_in_batch, row, batch_idx, start_idx


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
        Array of shape `(..., length)` to complement. In other words, elements of the
        array should be single characters.
    complement_map : dict[bytes, bytes]
        Dictionary mapping nucleotides to their complements.
    """
    # NOTE: a vectorized implementation using np.unique is NOT faster even for longer
    # alphabets like IUPAC DNA/RNA. Another micro-optimization to try would be using
    # vectorized bit manipulations.
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


def rev_comp_string(string: str, alphabet: SequenceAlphabet):
    comp = string.translate(alphabet.str_comp_table)
    return comp[::-1]


def rev_comp_bstring(bstring: bytes, alphabet: SequenceAlphabet):
    comp = bstring.translate(alphabet.bytes_comp_table)
    return comp[::-1]


def pad_byte_str(
    bstring: bytes,
    pad_len: int,
    padding: Literal["left", "center", "right"],
    alphabet: SequenceAlphabet,
):
    raise NotImplementedError


def run_shell(cmd: str, logger: logging.Logger, **kwargs):
    try:
        status = run(dedent(cmd).strip(), check=True, shell=True, **kwargs)
    except CalledProcessError as e:
        logger.error(e.stdout)
        logger.error(e.stderr)
        raise e
    return status
