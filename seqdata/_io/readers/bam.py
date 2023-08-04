from enum import Enum
from pathlib import Path
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
from seqdata.types import DTYPE, PathType, RegionReader

### pysam implementation NOTE ###

# pysam.AlignmentFile.count_coverage
# contig not found => raises KeyError
# start < 0 => raises ValueError
# end > reference length => truncates interval


class CountMethod(str, Enum):
    DEPTH = "depth-only"
    TN5_CUTSITE = "tn5-cutsite"
    TN5_FRAGMENT = "tn5-fragment"


# TODO: write docstring
class BAM(RegionReader, Generic[DTYPE]):
    def __init__(
        self,
        name: str,
        bams: Union[str, Path, List[str], List[Path]],
        samples: Union[str, List[str]],
        batch_size: int,
        n_jobs=1,
        threads_per_job=1,
        dtype: Union[str, Type[np.number]] = np.uint16,
        sample_dim: Optional[str] = None,
        offset_tn5=False,
        count_method: Union[
            CountMethod, Literal["depth-only", "tn5-cutsite", "tn5-fragment"]
        ] = "depth-only",
    ) -> None:
        """Reader for BAM files.

        Parameters
        ----------
        name : str
            Name of the array this reader will write.
        bams : Union[str, Path, List[str], List[Path]]
            Path or a list of paths to BAM(s).
        samples : Union[str, List[str]]
            Sample names for each BAM.
        batch_size : int
            Number of sequences to write at a time. Note this also sets the chunksize
            along the sequence dimension.
        n_jobs : int, optional
            Number of BAMs to process in parallel, by default 1, which disables
            multiprocessing. Don't set this higher than the number of BAMs or number of
            cores available.
        threads_per_job : int, optional
            Threads to use per job, by default 1. Make sure the number of available
            cores is >= n_jobs * threads_per_job.
        dtype : Union[str, Type[np.number]], optional
            Data type to write the coverage as, by default np.uint16.
        sample_dim : Optional[str], optional
            Name of the sample dimension, by default None
        offset_tn5 : bool, optional
            Whether to adjust read lengths to account for Tn5 binding, by default False
        count_method : Union[CountMethod, Literal["depth-only", "tn5-cutsite", "tn5-fragment"]]
            Count method, by default "depth-only"
        """
        if isinstance(bams, str):
            bams = [bams]
        elif isinstance(bams, Path):
            bams = [bams]
        if isinstance(samples, str):
            samples = [samples]

        self.name = name
        self.total_reads_name = f"total_reads_{name}"
        self.bams = bams
        self.samples = samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.threads_per_job = threads_per_job
        self.dtype = np.dtype(dtype)
        self.sample_dim = f"{name}_sample" if sample_dim is None else sample_dim
        self.offset_tn5 = offset_tn5
        self.count_method = CountMethod(count_method)

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
        root = zarr.open_group(out)

        arr = root.array(
            self.sample_dim,
            data=np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenUTF8(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        coverage = root.zeros(
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

        total_reads = root.zeros(
            self.total_reads_name,
            shape=len(self.samples),
            dtype=np.uint64,
            chunks=None,
            overwrite=overwrite,
            compressor=compressor,
        )
        total_reads.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(self._read_bam_fixed_length)(
                root,
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
        root = zarr.open_group(out)

        arr = root.array(
            self.sample_dim,
            data=np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenUTF8(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        coverage = root.empty(
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

        total_reads = root.zeros(
            self.total_reads_name,
            shape=len(self.samples),
            dtype=np.uint64,
            chunks=None,
            overwrite=overwrite,
            compressor=compressor,
        )
        total_reads.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(self._read_bam_variable_length)(
                root,
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

    def _read_bam_fixed_length(
        self,
        root: zarr.Group,
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

            def read_cb(x: pysam.AlignedSegment):
                return x.is_proper_pair and not x.is_secondary

            total_reads = sum([f.count(c, read_callback=read_cb) for c in f.references])
            root[self.total_reads_name][sample_idx] = total_reads

            reader = self._spliced_reader if splice else self._reader
            row_batcher = _get_row_batcher(reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                batch[idx] = out
                if is_last_in_batch or is_last_row:
                    _batch = batch[: idx + 1]
                    to_rc_mask = to_rc[start : start + idx + 1]
                    _batch[to_rc_mask] = _batch[to_rc_mask, ::-1]
                    root[self.name][start : start + idx + 1, sample_idx] = _batch

    def _read_bam_variable_length(
        self,
        root: zarr.Group,
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

            def read_cb(x: pysam.AlignedSegment):
                return x.is_proper_pair and not x.is_secondary

            total_reads = sum([f.count(c, read_callback=read_cb) for c in f.references])
            root[self.total_reads_name][sample_idx] = total_reads

            reader = self._spliced_reader if splice else self._reader
            row_batcher = _get_row_batcher(reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                if to_rc[idx]:
                    out = out[::-1]
                batch[idx] = out
                if is_last_in_batch or is_last_row:
                    root[self.name][start : start + idx + 1, sample_idx] = batch[
                        : idx + 1
                    ]

    def _reader(self, bed: pd.DataFrame, f: pysam.AlignmentFile):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            if self.count_method is CountMethod.DEPTH:
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
                if self.count_method is CountMethod.DEPTH:
                    coverage = self._count_depth_only(f, contig, start, end)
                else:
                    coverage = self._count_tn5(f, contig, start, end)
                unspliced.append(coverage)
            yield cast(NDArray[DTYPE], np.concatenate(coverage))  # type: ignore

    def _count_depth_only(
        self, f: pysam.AlignmentFile, contig: str, start: int, end: int
    ):
        a, c, g, t = f.count_coverage(
            contig,
            max(start, 0),
            end,
            read_callback=lambda x: x.is_proper_pair and not x.is_secondary,
        )
        coverage = np.vstack([a, c, g, t]).sum(0).astype(self.dtype)
        if (pad_len := end - start - len(coverage)) > 0:
            pad_arr = np.zeros(pad_len, dtype=self.dtype)
            pad_left = start < 0
            if pad_left:
                coverage = np.concatenate([pad_arr, coverage])
            else:
                coverage = np.concatenate([coverage, pad_arr])
        return coverage

    def _count_tn5(self, f: pysam.AlignmentFile, contig: str, start: int, end: int):
        length = end - start
        out_array = np.zeros(length, dtype=self.dtype)

        read_cache: Dict[str, pysam.AlignedSegment] = {}

        for i, read in enumerate(f.fetch(contig, max(0, start), end)):
            if not read.is_proper_pair or read.is_secondary:
                continue

            if read.query_name not in read_cache:
                read_cache[read.query_name] = read  # type: ignore
                continue

            # Forward and Reverse w/o r1 and r2
            if read.is_reverse:
                forward_read = read_cache.pop(read.query_name)
                reverse_read = read
            else:
                forward_read = read
                reverse_read = read_cache.pop(read.query_name)

            rel_start = forward_read.reference_start - start
            rel_end = cast(int, reverse_read.reference_end) - start

            # Shift read if accounting for offset
            if self.offset_tn5:
                rel_start += 4
                # 0 based, 1 past aligned
                rel_end -= 5

            # Check count method
            if self.count_method is CountMethod.TN5_CUTSITE:
                # Add cut sites to out_array
                if rel_start >= 0 and rel_start < length:
                    out_array[rel_start] += 1
                if rel_end >= 0 and rel_end < length:
                    out_array[rel_end] += 1
            elif self.count_method is CountMethod.TN5_FRAGMENT:
                # Add range to out array
                out_array[rel_start:rel_end] += 1

        # if any reads are still in the cache, then their mate isn't in the region
        for read in read_cache.values():
            # for reverse reads, their mate is in the 5' <- direction
            if read.is_reverse:
                rel_end = cast(int, read.reference_end) - start
                if self.offset_tn5:
                    rel_end -= 5
                    if rel_end < 0:
                        continue
                if self.count_method is CountMethod.TN5_CUTSITE:
                    out_array[rel_end - 1] += 1
                elif self.count_method is CountMethod.TN5_FRAGMENT:
                    out_array[:rel_end] += 1
            # for forward reads, their mate is in the 3' -> direction
            else:
                rel_start = read.reference_start - start
                if self.offset_tn5:
                    rel_start += 4
                    if rel_start >= length:
                        continue
                if self.count_method is CountMethod.TN5_CUTSITE:
                    out_array[rel_start] += 1
                elif self.count_method is CountMethod.TN5_FRAGMENT:
                    out_array[rel_start:] += 1

        return out_array
