import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    Union,
    cast,
)

import joblib
import numpy as np
import polars as pl
import pysam
import zarr
from more_itertools import split_when
from numcodecs import (
    Blosc,
    Delta,
    VLenArray,
    VLenUTF8,
    blosc,  # type: ignore
)
from numpy.typing import NDArray
from tqdm import tqdm
from typing_extensions import assert_never

from seqdata._io.utils import _get_row_batcher
from seqdata.types import DTYPE, PathType, RegionReader

### pysam implementation NOTE ###

# pysam.AlignmentFile.count_coverage
# contig not found => raises KeyError
# start < 0 => raises ValueError
# end > reference length => truncates interval


class CountMethod(str, Enum):
    READS = "reads"
    FRAGMENTS = "fragments"
    ENDS = "ends"


class BAM(RegionReader, Generic[DTYPE]):
    def __init__(
        self,
        name: str,
        bams: Union[PathType, List[PathType]],
        samples: Union[str, List[str]],
        batch_size: int,
        count_method: Union[CountMethod, Literal["reads", "fragments", "ends"]],
        n_jobs=-1,
        threads_per_job=-1,
        dtype: Union[str, Type[np.number]] = np.uint16,
        sample_dim: Optional[str] = None,
        pos_shift: Optional[int] = None,
        neg_shift: Optional[int] = None,
        min_mapping_quality: Optional[int] = None,
    ) -> None:
        """Reader for next-generation sequencing paired-end BAM files. This reader will only count
        reads that are properly paired and not secondary alignments.

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
            Number of BAMs to process in parallel, by default -1. If -1, use the number
            of available cores or the number of BAMs, whichever is smaller. If 0 or 1, process
            BAMs sequentially. Not recommended to set this higher than the number of BAMs.
        threads_per_job : int, optional
            Threads to use per job, by default -1. If -1, uses any extra cores available after
            allocating them to n_jobs. Not recommended to set this higher than the number of cores
            available divided by n_jobs.
        dtype : Union[str, Type[np.number]], optional
            Data type to write the coverage as, by default np.uint16.
        sample_dim : Optional[str], optional
            Name of the sample dimension, by default None
        count_method : Union[CountMethod, Literal["reads", "fragments", "ends"]]
            Count method:
            - "reads" counts the base pairs spanning the aligned sequences of reads.
            - "fragments" counts the base pairs spanning from the start of R1 to the end of R2.
            - "ends" counts only the single base positions for the start of R1 and the end of R2.

        pos_shift : Optional[int], optional
            Shift the forward read by this amount, by default None
        neg_shift : Optional[int], optional
            Shift the negative read by this amount, by default None
        min_mapping_quality : Optional[int], optional
            Minimum mapping quality for reads to be counted, by default None
        """
        if isinstance(bams, str) or isinstance(bams, Path):
            bams = [bams]
        if isinstance(samples, str):
            samples = [samples]

        self.name = name
        self.total_reads_name = f"total_reads_{name}"
        self.bams = bams
        self.samples = samples
        self.batch_size = batch_size
        self.count_method = CountMethod(count_method)
        self.dtype = np.dtype(dtype)
        self.sample_dim = f"{name}_sample" if sample_dim is None else sample_dim
        self.pos_shift = pos_shift
        self.neg_shift = neg_shift
        self.min_mapping_quality = min_mapping_quality

        n_cpus = len(os.sched_getaffinity(0))
        if n_jobs == -1:
            n_jobs = min(n_cpus, len(bams))
        elif n_jobs == 0:
            n_jobs = 1

        if threads_per_job == -1:
            threads_per_job = 1
            if n_cpus > n_jobs:
                threads_per_job = n_cpus // n_jobs

        self.n_jobs = n_jobs
        self.threads_per_job = threads_per_job

    def _write(
        self,
        out: PathType,
        bed: pl.DataFrame,
        fixed_length: Union[int, Literal[False]],
        sequence_dim: str,
        length_dim: Optional[str] = None,
        splice=False,
        overwrite=False,
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
        bed: pl.DataFrame,
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
        with joblib.parallel_config(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)

    def _write_variable_length(
        self,
        out: PathType,
        bed: pl.DataFrame,
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
        bed: pl.DataFrame,
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
        bed: pl.DataFrame,
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

    def _reader(self, bed: pl.DataFrame, f: pysam.AlignmentFile):
        for row in tqdm(bed.iter_rows(), total=len(bed)):
            contig, start, end = row[:3]
            coverage = self._count(f, contig, start, end)
            yield coverage

    def _spliced_reader(self, bed: pl.DataFrame, f: pysam.AlignmentFile):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.iter_rows(),
            lambda x, y: x[3] != y[3],  # 4th column is "name"
        ):
            unspliced: List[NDArray[Any]] = []
            for row in rows:
                pbar.update()
                contig, start, end = row[:3]
                coverage = self._count(f, contig, start, end)
                unspliced.append(coverage)
            yield cast(NDArray[DTYPE], np.concatenate(coverage))  # type: ignore

    def _count(
        self,
        f: pysam.AlignmentFile,
        contig: str,
        start: int,
        end: int,
    ) -> NDArray[DTYPE]:
        length = end - start
        out_array = np.zeros(length, dtype=self.dtype)

        read_cache: Dict[str, pysam.AlignedSegment] = {}

        for read in f.fetch(contig, max(0, start), end):
            if (
                not read.is_proper_pair
                or read.is_secondary
                or (
                    self.min_mapping_quality is not None
                    and read.mapping_quality < self.min_mapping_quality
                )
            ):
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
            # 0-based, 1 past aligned
            # e.g. start:end == 0:2 == [0, 1] so position of end == 1
            rel_end = reverse_read.reference_end - start  # type: ignore | reference_end is defined for proper pairs

            # Shift read if accounting for offset
            if self.pos_shift:
                rel_start = max(0, rel_start + self.pos_shift)
            if self.neg_shift:
                rel_end = min(length, rel_end + self.neg_shift)

            if self.count_method is CountMethod.ENDS:
                # Add cut sites to out_array
                if rel_start >= 0 and rel_start < length:
                    out_array[rel_start] += 1
                if rel_end >= 0 and rel_end <= length:
                    out_array[rel_end - 1] += 1
            elif self.count_method is CountMethod.FRAGMENTS:
                # Add range to out array
                out_array[rel_start:rel_end] += 1
            elif self.count_method is CountMethod.READS:
                out_array[rel_start : forward_read.reference_end - start] += 1  # type: ignore | reference_end is defined for proper pairs
                out_array[reverse_read.reference_start - start : rel_end] += 1
            else:
                assert_never(self.count_method)

        # if any reads are still in the cache, then their mate isn't in the region or didn't meet quality threshold
        for read in read_cache.values():
            # for reverse reads, their mate is in the 5' <- direction
            if read.is_reverse:
                rel_end = read.reference_end - start  # type: ignore | reference_end is defined for proper pairs
                if self.neg_shift:
                    rel_end = min(length, rel_end + self.neg_shift)
                    if rel_end < 0 or rel_end > length:
                        continue
                if self.count_method is CountMethod.ENDS:
                    out_array[rel_end - 1] += 1
                elif self.count_method is CountMethod.FRAGMENTS:
                    out_array[:rel_end] += 1
                elif self.count_method is CountMethod.READS:
                    out_array[read.reference_start - start : rel_end] += 1
                else:
                    assert_never(self.count_method)
            # for forward reads, their mate is in the 3' -> direction
            else:
                rel_start = read.reference_start - start
                if self.pos_shift:
                    rel_start = max(0, rel_start + self.pos_shift)
                    if rel_start < 0 or rel_start >= length:
                        continue
                if self.count_method is CountMethod.ENDS:
                    out_array[rel_start] += 1
                elif self.count_method is CountMethod.FRAGMENTS:
                    out_array[rel_start:] += 1
                elif self.count_method is CountMethod.READS:
                    out_array[rel_start : read.reference_end - start] += 1  # type: ignore | reference_end is defined for proper pairs
                else:
                    assert_never(self.count_method)

        return out_array
