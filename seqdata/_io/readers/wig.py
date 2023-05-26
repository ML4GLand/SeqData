from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, Type, Union, cast

import joblib
import numpy as np
import pandas as pd
import pyBigWig
import zarr
from more_itertools import split_when
from numcodecs import Blosc, Delta, VLenArray, VLenUTF8, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata._io.utils import _get_row_batcher
from seqdata.types import DTYPE, ListPathType, PathType, RegionReader


class BigWig(RegionReader, Generic[DTYPE]):
    def __init__(
        self,
        name: str,
        bigwigs: ListPathType,
        samples: List[str],
        batch_size: int,
        n_jobs=1,
        threads_per_job=1,
        dtype: Union[str, Type[np.number]] = np.uint16,
        sample_dim: Optional[str] = None,
    ) -> None:
        self.name = name
        self.bigwigs = list(map(Path, bigwigs))
        self.samples = samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.threads_per_job = threads_per_job
        self.dtype = np.dtype(dtype)
        self.sample_dim = f"{name}_sample" if sample_dim is None else sample_dim

    def _write(
        self,
        out: PathType,
        bed: pd.DataFrame,
        fixed_length: Union[int, Literal[False]],
        sequence_dim: str,
        length_dim: Optional[str] = None,
        overwrite=False,
        splice=False,
    ) -> None:
        if self.name in (sequence_dim, self.sample_dim, length_dim):
            raise ValueError(
                "Name cannot be equal to sequence_dim, sample_dim, or length_dim."
            )
        if fixed_length is False:
            self._write_variable_length(
                out=out,
                bed=bed,
                sequence_dim=sequence_dim,
                overwrite=overwrite,
                splice=splice,
            )
        else:
            assert length_dim is not None
            self._write_fixed_length(
                out=out,
                bed=bed,
                fixed_length=fixed_length,
                sequence_dim=sequence_dim,
                length_dim=length_dim,
                overwrite=overwrite,
                splice=splice,
            )

    def _write_fixed_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        fixed_length: int,
        sequence_dim: str,
        length_dim: str,
        overwrite: bool,
        splice: bool,
    ):
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = bed["name"].nunique() if splice else len(bed)
        batch_size = min(n_seqs, self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            self.sample_dim,
            data=np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenUTF8(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        coverage = z.zeros(
            self.name,
            shape=(n_seqs, len(self.samples), fixed_length),
            dtype=self.dtype,
            chunks=(batch_size, 1, None),
            overwrite=overwrite,
            compressor=compressor,
            filters=[Delta(self.dtype)],
        )
        coverage.attrs["_ARRAY_DIMENSIONS"] = [
            sequence_dim,
            self.sample_dim,
            length_dim,
        ]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(self._read_bigwig_fixed_length)(
                coverage,
                bigwig,
                bed,
                batch_size,
                sample_idx,
                self.threads_per_job,
                fixed_length=fixed_length,
                splice=splice,
            )
            for bigwig, sample_idx in zip(self.bigwigs, sample_idxs)
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)

    def _write_variable_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        sequence_dim: str,
        overwrite: bool,
        splice: bool,
    ):
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = bed["name"].nunique() if splice else len(bed)
        batch_size = min(n_seqs, self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            self.sample_dim,
            data=np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenUTF8(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        coverage = z.empty(
            self.name,
            shape=(n_seqs, len(self.samples)),
            dtype=object,
            chunks=(batch_size, 1),
            overwrite=overwrite,
            compressor=compressor,
            filters=[Delta(self.dtype)],
            object_codec=VLenArray(self.dtype),
        )
        coverage.attrs["_ARRAY_DIMENSIONS"] = [
            sequence_dim,
            self.sample_dim,
        ]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(self._read_bigwig_variable_length)(
                coverage,
                bigwig,
                bed,
                batch_size,
                sample_idx,
                self.threads_per_job,
                splice=splice,
            )
            for bigwig, sample_idx in zip(self.bigwigs, sample_idxs)
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)

    def _reader(self, bed: pd.DataFrame, f, contig_lengths: Dict[str, int]):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            values = cast(
                NDArray,
                f.values(
                    contig, max(0, start), min(contig_lengths[contig], end), numpy=True
                ),
            )
            values[values == np.nan] = 0
            values = values.astype(self.dtype)
            pad_left = max(-start, 0)
            pad_right = max(end - contig_lengths[contig], 0)
            values = np.concatenate(
                [
                    np.zeros(pad_left, self.dtype),
                    values,
                    np.zeros(pad_right, self.dtype),
                ]
            )
            yield values

    def _spliced_reader(self, bed: pd.DataFrame, f, contig_lengths: Dict[str, int]):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.itertuples(index=False), lambda x, y: x.name != y.name
        ):
            unspliced: List[NDArray[Any]] = []
            for row in rows:
                pbar.update()
                contig, start, end = row[:3]
                values = cast(NDArray, f.values(contig, start, end, numpy=True))
                values[values == np.nan] = 0
                values = values.astype(self.dtype)
                pad_left = max(-start, 0)
                pad_right = max(end - contig_lengths[contig], 0)
                values = np.concatenate(
                    [
                        np.zeros(pad_left, self.dtype),
                        values,
                        np.zeros(pad_right, self.dtype),
                    ]
                )
                unspliced.append(values)
            yield np.concatenate(unspliced)

    def _read_bigwig_fixed_length(
        self,
        coverage: zarr.Array,
        bigwig: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
        fixed_length: int,
        splice: bool,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.zeros((batch_size, fixed_length), self.dtype)

        with pyBigWig.open(str(bigwig)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            contig_lengths = f.chroms()
            row_batcher = _get_row_batcher(reader(bed, f, contig_lengths), batch_size)
            for is_last_row, is_last_in_batch, values, idx, start in row_batcher:
                batch[idx] = values
                if is_last_row or is_last_in_batch:
                    _batch = batch[: idx + 1]
                    to_rc_mask = to_rc[start : start + idx + 1]
                    _batch[to_rc_mask] = _batch[to_rc_mask, ::-1]
                    coverage[start : start + idx + 1, sample_idx] = _batch

    def _read_bigwig_variable_length(
        self,
        coverage: zarr.Array,
        bigwig: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
        splice: bool,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.empty(batch_size, object)

        with pyBigWig.open(str(bigwig)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            contig_lengths = f.chroms()
            row_batcher = _get_row_batcher(reader(bed, f, contig_lengths), batch_size)
            for is_last_row, is_last_in_batch, values, idx, start in row_batcher:
                if to_rc[idx]:
                    batch[idx] = values[::-1]
                else:
                    batch[idx] = values
                if is_last_in_batch or is_last_row:
                    coverage[start : start + idx + 1, sample_idx] = batch[: idx + 1]
