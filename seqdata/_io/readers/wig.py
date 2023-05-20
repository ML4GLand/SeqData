from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type, Union, cast

import joblib
import numpy as np
import pandas as pd
import pyBigWig
import zarr
from numcodecs import Blosc, Delta, VLenArray, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata._io.utils import _get_row_batcher
from seqdata.types import DTYPE, PathType, RegionReader


class BigWig(RegionReader, Generic[DTYPE]):
    def __init__(
        self,
        name: str,
        bigwigs: List[PathType],
        samples: List[str],
        batch_size: int,
        n_jobs=1,
        threads_per_job=1,
        samples_per_chunk=10,
        dtype: Union[str, Type[np.number]] = np.uint16,
    ) -> None:
        self.name = name
        self.bigwigs = list(map(Path, bigwigs))
        self.samples = samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.threads_per_job = threads_per_job
        self.samples_per_chunk = samples_per_chunk
        self.dtype = np.dtype(dtype)

    def _reader(self, bed: pd.DataFrame, f):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            intervals = cast(
                List[Tuple[int, int, Union[int, float]]],
                f.intervals(contig, start, end),
            )
            start = cast(int, start)
            end = cast(int, end)
            yield intervals, start, end

    def _read_bigwig_fixed_length(
        self,
        coverage: zarr.Array,
        bigwig: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
        length: int,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.zeros((batch_size, length), self.dtype)

        with pyBigWig.open(bigwig) as f:
            row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                intervals, start, end = out
                if intervals is not None:
                    for interval in intervals:
                        rel_start = interval[0] - start
                        rel_end = interval[1] - start
                        value = interval[2]
                        batch[idx, rel_start:rel_end] = value
                if is_last_row or is_last_in_batch:
                    to_rc_mask = to_rc[start : start + idx + 1]
                    batch[to_rc_mask] = batch[to_rc_mask, ::-1]
                    coverage[start : start + idx + 1, sample_idx] = batch[: idx + 1]

    def _read_bigwig_variable_length(
        self,
        coverage: zarr.Array,
        bigwig: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.empty(batch_size, object)

        with pyBigWig.open(bigwig) as f:
            row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                intervals, start, end = out
                row = np.zeros((end - start), self.dtype)
                if intervals is not None:
                    for interval in intervals:
                        rel_start = interval[0] - start
                        rel_end = interval[1] - start
                        value = interval[2]
                        row[rel_start:rel_end] = value
                if to_rc[idx]:
                    batch[idx] = row[::-1]
                else:
                    batch[idx] = row
                if is_last_in_batch or is_last_row:
                    coverage[start : start + idx + 1, sample_idx] = batch[: idx + 1]

    def _write(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: Optional[int] = None,
        overwrite=False,
    ) -> None:
        if length is None:
            self._write_variable_length(out, bed, overwrite)
        else:
            self._write_fixed_length(out, bed, length, overwrite)

    def _write_fixed_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: int,
        overwrite=False,
    ):
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        batch_size = min(len(bed), self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            f"{self.name}_sample",
            data=np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [f"{self.name}_sample"]

        coverage = z.zeros(
            self.name,
            shape=(len(bed), len(self.samples), length),
            dtype=self.dtype,
            chunks=(batch_size, self.samples_per_chunk, None),
            overwrite=overwrite,
            compressor=compressor,
            filters=[Delta(self.dtype)],
        )
        coverage.attrs["_ARRAY_DIMENSIONS"] = [
            "_sequence",
            f"{self.name}_sample",
            f"{self.name}_length",
        ]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(
                self._read_bigwig_fixed_length(
                    coverage,
                    bigwig,
                    bed,
                    batch_size,
                    sample_idx,
                    self.threads_per_job,
                    length=length,
                )
                for bigwig, sample_idx in zip(self.bigwigs, sample_idxs)
            )
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)

    def _write_variable_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        overwrite=False,
    ):
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        batch_size = min(len(bed), self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            f"{self.name}_sample",
            data=np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [f"{self.name}_sample"]

        coverage = z.empty(
            self.name,
            shape=(len(bed), len(self.samples)),
            dtype=object,
            chunks=(batch_size, self.samples_per_chunk),
            overwrite=overwrite,
            compressor=compressor,
            filters=[Delta(self.dtype)],
            object_codec=VLenArray(self.dtype),
        )
        coverage.attrs["_ARRAY_DIMENSIONS"] = [
            "_sequence",
            f"{self.name}_sample",
        ]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(
                self._read_bigwig_variable_length(
                    coverage,
                    bigwig,
                    bed,
                    batch_size,
                    sample_idx,
                    self.threads_per_job,
                )
                for bigwig, sample_idx in zip(self.bigwigs, sample_idxs)
            )
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)
