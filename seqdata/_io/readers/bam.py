from enum import Enum
from typing import Any, Dict, Generic, List, Literal, Optional, Type, Union, cast

import joblib
import numpy as np
import pandas as pd
import pysam
import zarr
from more_itertools import split_when
from numcodecs import Blosc, Delta, VLenArray, VLenUTF8, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata._io.utils import _get_row_batcher
from seqdata.types import DTYPE, ListPathType, PathType, RegionReader

### pysam implementation NOTE ###

# pysam.AlignmentFile.count_coverage
# contig not found => raises KeyError
# start < 0 => raises ValueError
# end > reference length => truncates interval


class Tn5CountMethod(str, Enum):
    CUTSITE = "tn5-cutsite"
    MIDPOINT = "tn5-midpoint"
    FRAGMENT = "tn5-fragment"


class BAM(RegionReader, Generic[DTYPE]):
    def __init__(
        self,
        name: str,
        bams: ListPathType,
        samples: List[str],
        batch_size: int,
        n_jobs=1,
        threads_per_job=1,
        dtype: Union[str, Type[np.number]] = np.uint16,
        sample_dim: Optional[str] = None,
        offset_tn5=False,
        count_method: Literal[
            "depth-only", "tn5-cutsite", "tn5-midpiont", "tn5-fragment"
        ] = "depth-only",
    ) -> None:
        self.name = name
        self.bams = bams
        self.samples = samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.threads_per_job = threads_per_job
        self.dtype = np.dtype(dtype)
        self.sample_dim = f"{name}_sample" if sample_dim is None else sample_dim
        self.offset_tn5 = offset_tn5
        if count_method != "depth-only":
            _count_method = Tn5CountMethod(count_method)
        else:
            _count_method = count_method
        self.count_method = _count_method

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
        if fixed_length is False:
            self._write_variable_length(out, bed, sequence_dim, overwrite, splice)
        else:
            assert length_dim is not None
            self._write_fixed_length(
                out, bed, fixed_length, sequence_dim, length_dim, overwrite, splice
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

        batch_size = min(len(bed), self.batch_size)
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
            shape=(len(bed), len(self.samples), fixed_length),
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
            joblib.delayed(self._read_bam_fixed_length)(
                coverage,
                bam,
                bed,
                batch_size,
                sample_idx,
                self.threads_per_job,
                fixed_length=fixed_length,
                splice=splice,
            )
            for bam, sample_idx in zip(self.bams, sample_idxs)
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

        batch_size = min(len(bed), self.batch_size)
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
            shape=(len(bed), len(self.samples)),
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
            joblib.delayed(self._read_bam_variable_length)(
                coverage,
                bam,
                bed,
                batch_size,
                sample_idx,
                self.threads_per_job,
                splice=splice,
            )
            for bam, sample_idx in zip(self.bams, sample_idxs)
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)

    def _count_tn5(self, f: pysam.AlignmentFile, contig: str, start: int, end: int):
        length = end - start
        out_array = np.zeros(length, dtype=self.dtype)

        read_cache: Dict[str, Any] = {}

        for read in f.fetch(contig, max(0, start), end):
            if not read.is_proper_pair or read.is_secondary:
                continue

            if read.query_name not in read_cache:
                read_cache[read.query_name] = read
                continue

            # Forward and Reverse w/o r1 and r2
            if read.is_reverse:
                forward_read = read_cache.pop(read.query_name)
                reverse_read = read
            else:
                forward_read = read
                reverse_read = read_cache.pop(read.query_name)

            # Shift read if accounting for offset
            if self.offset_tn5:
                forward_start: int = forward_read.reference_start + 4
                # 0 based, 1 past aligned
                reverse_end: int = reverse_read.reference_end - 5
            else:
                forward_start = forward_read.reference_start
                reverse_end = reverse_read.reference_end

            # Check count method
            if self.count_method is Tn5CountMethod.CUTSITE:
                # Add cut sites to out_array
                out_array[[forward_start, (reverse_end - 1)]] += 1
            elif self.count_method is Tn5CountMethod.MIDPOINT:
                # Add midpoint to out_array
                out_array[int((forward_start + (reverse_end - 1)) / 2)] += 1
            elif self.count_method is Tn5CountMethod.FRAGMENT:
                # Add range to out array
                out_array[forward_start:reverse_end] += 1

        return out_array

    def _count_depth_only(
        self, f: pysam.AlignmentFile, contig: str, start: int, end: int
    ):
        a, c, g, t = f.count_coverage(contig, max(start, 0), end, read_callback="all")
        coverage = np.vstack([a, c, g, t]).sum(0).astype(self.dtype)
        if (pad_len := end - start - len(coverage)) > 0:
            pad_arr = np.zeros(pad_len, dtype=self.dtype)
            pad_left = start < 0
            if pad_left:
                coverage = np.concatenate([pad_arr, coverage])
            else:
                coverage = np.concatenate([coverage, pad_arr])
        return coverage

    def _reader(self, bed: pd.DataFrame, f: pysam.AlignmentFile):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            if self.count_method == "depth-only":
                coverage = self._count_depth_only(f, contig, start, end)
            else:
                coverage = self._count_tn5(f, contig, start, end)
            yield coverage

    def _spliced_reader(self, bed: pd.DataFrame, f: pysam.AlignmentFile):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.itertuples(index=False), lambda x, y: x.name != y.name
        ):
            unspliced: List[NDArray[Any]] = []
            for row in rows:
                pbar.update()
                contig, start, end = row[:3]
                if self.count_method == "depth-only":
                    coverage = self._count_depth_only(f, contig, start, end)
                else:
                    coverage = self._count_tn5(f, contig, start, end)
                unspliced.append(coverage)
            yield cast(NDArray[DTYPE], np.concatenate(coverage))  # type: ignore

    def _read_bam_fixed_length(
        self,
        coverage: zarr.Array,
        bam: PathType,
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

        with pysam.AlignmentFile(str(bam), threads=n_threads) as f:
            reader = self._spliced_reader if splice else self._reader
            row_batcher = _get_row_batcher(reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                batch[idx] = out
                if is_last_in_batch or is_last_row:
                    _batch = batch[: idx + 1]
                    to_rc_mask = to_rc[start : start + idx + 1]
                    _batch[to_rc_mask] = _batch[to_rc_mask, ::-1]
                    coverage[start : start + idx + 1, sample_idx] = _batch

    def _read_bam_variable_length(
        self,
        coverage: zarr.Array,
        bam: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
        splice: bool,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.empty(batch_size, object)

        with pysam.AlignmentFile(str(bam), threads=n_threads) as f:
            reader = self._spliced_reader if splice else self._reader
            row_batcher = _get_row_batcher(reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                if to_rc[idx]:
                    out = out[::-1]
                batch[idx] = out
                if is_last_in_batch or is_last_row:
                    coverage[start : start + idx + 1, sample_idx] = batch[: idx + 1]
