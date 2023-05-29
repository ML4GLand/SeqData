import logging
from itertools import count, cycle
from subprocess import CalledProcessError, run
from textwrap import dedent
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import polars as pl
import zarr
from more_itertools import mark_ends, repeat_each
from numcodecs import VLenArray, VLenBytes, VLenUTF8

from seqdata.types import T


def _df_to_xr_zarr(df: pd.DataFrame, root: zarr.Group, dim: str, **kwargs):
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
        arr.attrs["_ARRAY_DIMENSIONS"] = [dim]


def _polars_df_to_xr_zarr(df: pl.DataFrame, root: zarr.Group, dim: str, **kwargs):
    for series in df.get_columns():
        data = series.to_numpy()
        if data.dtype.type == np.object_:
            if series.dtype == pl.List:
                object_codec = VLenArray(data[0].dtype)
            elif series.dtype == pl.Utf8:
                object_codec = VLenUTF8()
            elif series.dtype == pl.Binary:
                object_codec = VLenBytes()
            else:
                raise ValueError(
                    f'Got column "{series.name}" in dataframe that isn\'t serializable.'
                )
        else:
            object_codec = None
        arr = root.array(series.name, data, object_codec=object_codec, **kwargs)
        arr.attrs["_ARRAY_DIMENSIONS"] = [dim]


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


def run_shell(cmd: str, logger: logging.Logger, **kwargs):
    try:
        status = run(dedent(cmd).strip(), check=True, shell=True, **kwargs)
    except CalledProcessError as e:
        logger.error(e.stdout)
        logger.error(e.stderr)
        raise e
    return status
