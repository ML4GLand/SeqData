from functools import partial
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Union

import numpy as np
import polars as pl
import zarr
from numcodecs import Blosc, VLenBytes
from tqdm import tqdm

from seqdata._io.utils import _df_to_xr_zarr
from seqdata.types import FlatReader, ListPathType, PathType


class Table(FlatReader):
    def __init__(
        self,
        name: str,
        tables: Union[PathType, ListPathType],
        seq_col: str,
        batch_size: int,
        **kwargs,
    ) -> None:
        """Reader for tabular data.

        Parameters
        ----------
        name : str
            Name of the data.
        tables : str, Path, List[str], List[Path]
            Path to table or list of paths to tables.
        seq_col : str
            Name of the column containing the sequences.
        batch_size : int
            Number of rows to read at a time.
        **kwargs
            Additional keyword arguments to pass to `pd.read_csv`.

        Returns
        -------
        None
        """
        self.name = name
        if not isinstance(tables, list):
            tables = [Path(tables)]
        self.tables = list(map(Path, tables))
        self.seq_col = seq_col
        self.batch_size = batch_size
        self.kwargs = kwargs

    def _get_reader(self, table: Path):
        if ".csv" in table.suffixes:
            sep = ","
        elif ".tsv" in table.suffixes:
            sep = "\t"
        elif ".txt" in table.suffixes:
            sep = "\t"
        else:
            sep = None
        if sep is None:
            return pl.read_csv_batched(table, batch_size=self.batch_size, **self.kwargs)
        else:
            return pl.read_csv_batched(
                table, separator=sep, batch_size=self.batch_size, **self.kwargs
            )

    def _write_first_variable_length(
        self,
        batch: pl.DataFrame,
        root: zarr.Group,
        compressor,
        sequence_dim: str,
        overwrite: bool,
    ):
        seqs = batch[self.seq_col].cast(pl.Binary).to_numpy()
        obs = batch.drop(self.seq_col)
        arr = root.array(
            self.name,
            data=seqs,
            chunks=self.batch_size,
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenBytes(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [sequence_dim]
        _df_to_xr_zarr(
            obs,
            root,
            sequence_dim,
            chunks=self.batch_size,
            compressor=compressor,
            overwrite=overwrite,
        )
        first_cols = obs.columns
        return first_cols

    def _write_variable_length(
        self, batch: pl.DataFrame, root: zarr.Group, first_cols: List, table: Path
    ):
        seqs = batch[self.seq_col].cast(pl.Binary).to_numpy()
        obs = batch.drop(self.seq_col)
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
        for series in obs:
            root[series.name].append(series.to_numpy())  # type: ignore

    def _write_first_fixed_length(
        self,
        batch: pl.DataFrame,
        root: zarr.Group,
        compressor,
        sequence_dim: str,
        length_dim: str,
        overwrite: bool,
    ):
        seqs = (
            batch[self.seq_col]
            .cast(pl.Binary)
            .to_numpy()
            .astype("S")[..., None]
            .view("S1")
        )
        obs = batch.drop(self.seq_col)
        arr = root.array(
            self.name,
            data=seqs,
            chunks=(self.batch_size, None),
            compressor=compressor,
            overwrite=overwrite,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [sequence_dim, length_dim]
        _df_to_xr_zarr(
            obs,
            root,
            sequence_dim,
            chunks=self.batch_size,
            compressor=compressor,
            overwrite=overwrite,
        )
        first_cols = obs.columns
        return first_cols

    def _write_fixed_length(
        self, batch: pl.DataFrame, root: zarr.Group, first_cols: List, table: Path
    ):
        seqs = (
            batch[self.seq_col]
            .cast(pl.Binary)
            .to_numpy()
            .astype("S")[..., None]
            .view("S1")
        )
        obs = batch.drop(self.seq_col)
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
        for series in obs:
            root[series.name].append(series.to_numpy())  # type: ignore

    def _write(
        self,
        out: PathType,
        fixed_length: bool,
        sequence_dim: str,
        length_dim: Optional[str] = None,
        overwrite=False,
    ) -> None:
        compressor = Blosc("zstd", clevel=7, shuffle=-1)
        root = zarr.open_group(out)

        if fixed_length:
            assert length_dim is not None
            write_first = partial(
                self._write_first_fixed_length,
                sequence_dim=sequence_dim,
                length_dim=length_dim,
            )
            write_batch = self._write_fixed_length
        else:
            write_first = partial(
                self._write_first_variable_length, sequence_dim=sequence_dim
            )
            write_batch = self._write_variable_length

        pbar = tqdm()
        first_batch = True
        for table in self.tables:
            reader = self._get_reader(table)
            while batch := reader.next_batches(1):
                batch = batch[0]
                if first_batch:
                    first_cols = write_first(
                        batch=batch,
                        root=root,
                        compressor=compressor,
                        overwrite=overwrite,
                    )
                    first_batch = False
                else:
                    write_batch(
                        batch,
                        root,
                        first_cols,  # type: ignore guaranteed to be set during first batch
                        table,
                    )
                pbar.update(len(batch))
