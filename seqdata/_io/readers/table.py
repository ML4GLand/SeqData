from pathlib import Path
from textwrap import dedent
from typing import List, Union, cast

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc, VLenBytes

from seqdata._io.utils import _df_to_xr_zarr
from seqdata.types import FlatReader, PathType


class Table(FlatReader):
    def __init__(
        self,
        name: str,
        tables: Union[PathType, List[PathType]],
        seq_col: str,
        batch_size: int,
    ) -> None:
        self.name = name
        if not isinstance(tables, list):
            tables = [tables]
        self.tables = list(map(Path, tables))
        self.seq_col = seq_col
        self.batch_size = batch_size

    def _get_reader(self, table: Path):
        if ".csv" in table.suffixes:
            sep = ","
        elif ".tsv" in table.suffixes or ".txt" in table.suffixes:
            sep = "\t"
        else:
            raise ValueError("Unknown file extension.")
        return pd.read_csv(table, sep=sep, chunksize=self.batch_size)

    def _write_first_batch(
        self, batch: pd.DataFrame, root: zarr.Group, compressor, overwrite
    ):
        seqs = batch[self.seq_col].str.encode("ascii").to_numpy()
        obs = batch.drop(columns=self.seq_col)
        arr = root.array(
            self.name,
            data=seqs,
            chunks=self.batch_size,
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenBytes(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]
        _df_to_xr_zarr(
            obs,
            root,
            ["_sequence"],
            chunks=self.batch_size,
            compressor=compressor,
            overwrite=overwrite,
        )
        first_cols = obs.columns.to_list()
        return first_cols

    def _write_batch(
        self, batch: pd.DataFrame, root: zarr.Group, first_cols: List, table: Path
    ):
        seqs = batch[self.seq_col].str.encode("ascii").to_numpy()
        obs = batch.drop(columns=self.seq_col)
        if (
            np.isin(obs.columns, first_cols, invert=True).any()
            or np.isin(first_cols, obs.columns, invert=True).any()
        ):
            raise RuntimeError(
                dedent(
                    f"""Mismatching columns.
                First table {self.tables[0]} has columns {first_cols}
                Mismatched table {table} has columns {obs.columns}
                """
                ).strip()
            )
        root[self.name].append(seqs)  # type: ignore
        for name, series in obs.items():
            root[name].append(series.to_numpy())  # type: ignore  # type: ignore

    def _write(self, out: PathType, overwrite=False) -> None:
        compressor = Blosc("zstd", clevel=7, shuffle=-1)
        z = zarr.open_group(out)

        first_batch = True
        for table in self.tables:
            with self._get_reader(table) as reader:
                for batch in reader:
                    batch = cast(pd.DataFrame, batch)
                    if first_batch:
                        first_cols = self._write_first_batch(
                            batch, z, compressor, overwrite
                        )
                        first_batch = False
                    else:
                        self._write_batch(batch, z, first_cols, table)  # type: ignore
