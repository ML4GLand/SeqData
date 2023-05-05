import warnings
from pathlib import Path
from textwrap import dedent
from typing import Generic, List, Optional, Set, Tuple, Type, Union, cast

import cyvcf2
import joblib
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import zarr
from more_itertools import split_when
from natsort import natsorted
from numcodecs import Blosc, Delta, VLenArray, VLenBytes, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata.alphabets import ALPHABETS, SequenceAlphabet
from seqdata.types import DTYPE, FlatReader, PathType, RegionReader

from .utils import _df_to_xr_zarr, _get_row_batcher

### pysam and cyvcf2 implementation NOTE ###

# pysam.FastaFile.fetch
# contig not found => raises KeyError
# if start < 0 => raises ValueError
# if end > reference length => truncates interval

# pysam.AlignmentFile.count_coverage
# contig not found => raises KeyError
# start < 0 => raises ValueError
# end > reference length => truncates interval

# cyvcf2.VCF
# Contig not found => warning
# start < 0 => warning
# start = 0 (despite being 1-indexed) => nothing
# end > contig length => treats end = contig length


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


class FlatFASTA(FlatReader):
    def __init__(
        self,
        name: str,
        fasta: PathType,
        batch_size: int,
        n_threads: int = 1,
        fixed_length: bool = False,
        length_dim: Optional[str] = None,
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        with pysam.FastaFile(str(self.fasta)) as f:
            self.n_seqs = len(f.references)
        self.fixed_length = fixed_length
        self.length_dim = length_dim

    def _reader(self, f: pysam.FastaFile):
        for seq_name in f.references:
            seq = f.fetch(seq_name).encode("ascii")
            yield seq

    def _write_fixed_length(self, out: PathType, overwrite=False):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        z = zarr.open_group(out)
        with pysam.FastaFile(str(self.fasta)) as f:
            seq_names = f.references
            self.length = f.get_reference_length(seq_names[0])

            arr = z.array(
                f"{self.name}_id",
                data=np.array(list(seq_names), object),
                overwrite=overwrite,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

            n_seqs = len(seq_names)
            batch_size = min(n_seqs, self.batch_size)

            seqs = z.empty(
                self.name,
                shape=(n_seqs, self.length),
                dtype=f"|S{self.length}",
                chunks=batch_size,
                overwrite=overwrite,
                compressor=compressor,
            )
            length_dim = (
                f"{self.name}_length" if self.length_dim is None else self.length_dim
            )
            seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence", length_dim]

            batch = np.empty(batch_size, dtype=object)

            row_batcher = _get_row_batcher(self._reader(f), batch_size)
            for last_row, last_in_batch, seq, batch_idx, start_idx in tqdm(
                row_batcher, total=n_seqs
            ):
                if len(seq) != self.length:
                    raise RuntimeError(
                        """
                        Fixed length FlatFASTA reader got sequences with different
                        lengths.
                        """
                    )
                seq = np.frombuffer(seq, f"|S{self.length}")
                batch[batch_idx] = seq
                if last_in_batch or last_row:
                    seqs[start_idx : start_idx + batch_idx + 1] = batch[: batch_idx + 1]

    def _write_variable_length(self, out: PathType, overwrite=False):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        z = zarr.open_group(out)
        with pysam.FastaFile(str(self.fasta)) as f:
            seq_names = f.references

            arr = z.array(
                f"{self.name}_id",
                data=np.array(list(seq_names), object),
                overwrite=overwrite,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

            n_seqs = len(seq_names)
            batch_size = min(n_seqs, self.batch_size)

            seqs = z.empty(
                self.name,
                shape=n_seqs,
                dtype=object,
                chunks=batch_size,
                overwrite=overwrite,
                compressor=compressor,
                object_codec=VLenBytes(),
            )
            seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

            batch = np.empty(batch_size, dtype=object)

            row_batcher = _get_row_batcher(self._reader(f), batch_size)
            for last_row, last_in_batch, seq, batch_idx, start_idx in tqdm(
                row_batcher, total=n_seqs
            ):
                batch[batch_idx] = seq
                if last_in_batch or last_row:
                    seqs[start_idx : start_idx + batch_idx + 1] = batch[: batch_idx + 1]

    # mock single dispatch on class state
    def _write(self, out: PathType, overwrite=False) -> None:
        if self.fixed_length:
            self._write_fixed_length(out=out, overwrite=overwrite)
        else:
            self._write_variable_length(out=out, overwrite=overwrite)


class GenomeFASTA(RegionReader):
    def __init__(
        self,
        name: str,
        fasta: PathType,
        batch_size: int,
        n_threads: int = 1,
        alphabet: Optional[Union[str, SequenceAlphabet]] = None,
        length_dim: Optional[str] = None,
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        if alphabet is None:
            self.alphabet = ALPHABETS["DNA"]
        elif isinstance(alphabet, str):
            self.alphabet = ALPHABETS[alphabet]
        else:
            self.alphabet = alphabet
        self.length_dim = f"{self.name}_length" if length_dim is None else length_dim

    def _reader(self, bed: pd.DataFrame, f: pysam.FastaFile):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = cast(Tuple[str, int, int], row[:3])
            seq = f.fetch(contig, start, end).encode("ascii")
            if (pad_len := end - start - len(seq)) > 0:
                pad_left = start < 0
                if pad_left:
                    seq = (b"N" * pad_len) + seq
                else:
                    seq += b"N" * pad_len
            yield seq

    def _spliced_reader(self, bed: pd.DataFrame, f: pysam.FastaFile):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.itertuples(index=False), lambda x, y: x.name != y.name
        ):
            unspliced: List[bytes] = []
            for row in rows:
                pbar.update()
                contig, start, end = cast(Tuple[str, int, int], row[:3])
                seq = f.fetch(contig, start, end).encode("ascii")
                if (pad_len := end - start - len(seq)) > 0:
                    pad_left = start < 0
                    if pad_left:
                        seq = (b"N" * pad_len) + seq
                    else:
                        seq += b"N" * pad_len
                unspliced.append(seq)
            spliced = b"".join(unspliced)
            yield spliced

    def _write(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: Optional[int] = None,
        overwrite=False,
        splice=False,
    ) -> None:
        if length is None:
            self._write_variable_length(
                out=out, bed=bed, overwrite=overwrite, splice=splice
            )
        else:
            self._write_fixed_length(
                out=out, bed=bed, length=length, overwrite=overwrite, splice=splice
            )

    def _write_fixed_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: int,
        overwrite: bool,
        splice: bool,
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        root = zarr.open_group(out)

        seqs = root.empty(
            self.name,
            shape=(n_seqs, length),
            dtype="|S1",
            chunks=(batch_size, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence", self.length_dim]

        batch = cast(NDArray[np.bytes_], np.empty((batch_size, length), dtype="|S1"))

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                row_batcher = _get_row_batcher(self._spliced_reader(bed, f), batch_size)
            else:
                row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, seq, idx, start in tqdm(
                row_batcher, total=n_seqs
            ):
                seq = np.frombuffer(seq, "|S1")
                batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    to_rc_mask = to_rc[start : start + idx + 1]
                    batch[to_rc_mask] = self.alphabet.rev_comp_byte(batch[to_rc_mask])
                    seqs[start : start + idx + 1] = batch[: idx + 1]

    def _write_variable_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        overwrite: bool,
        splice: bool,
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        root = zarr.open_group(out)

        seqs = root.empty(
            self.name,
            shape=n_seqs,
            dtype=object,
            chunks=batch_size,
            overwrite=overwrite,
            compressor=compressor,
            object_codec=VLenBytes(),
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

        batch = cast(NDArray[np.object_], np.empty(batch_size, dtype=object))

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                row_batcher = _get_row_batcher(self._spliced_reader(bed, f), batch_size)
            else:
                row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, seq, idx, start in tqdm(
                row_batcher, total=n_seqs
            ):
                if to_rc[start + idx]:
                    batch[idx] = self.alphabet.rev_comp_bstring(seq)
                else:
                    batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    seqs[start : start + idx + 1] = batch[: idx + 1]


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
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        batch_size = min(len(bed), self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            f"{self.name}_samples",
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
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        batch_size = min(len(bed), self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            f"{self.name}_samples",
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


class BAM(RegionReader, Generic[DTYPE]):
    def __init__(
        self,
        name: str,
        bams: List[PathType],
        samples: List[str],
        batch_size: int,
        n_jobs=1,
        threads_per_job=1,
        samples_per_chunk=10,
        dtype: Union[str, Type[np.number]] = np.uint16,
    ) -> None:
        self.name = name
        self.bams = bams
        self.samples = samples
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.threads_per_job = threads_per_job
        self.samples_per_chunk = samples_per_chunk
        self.dtype = np.dtype(dtype)

    def _reader(self, bed: pd.DataFrame, f: pysam.AlignmentFile):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            a, c, g, t = f.count_coverage(
                contig, max(start, 0), end, read_callback="all"
            )
            coverage = np.vstack([a, c, g, t]).sum(0).astype(self.dtype)
            if (pad_len := end - start - len(coverage)) > 0:
                pad_arr = np.zeros(pad_len, dtype=self.dtype)
                pad_left = start < 0
                if pad_left:
                    coverage = np.concatenate([pad_arr, coverage])
                else:
                    coverage = np.concatenate([coverage, pad_arr])
            yield cast(NDArray[DTYPE], coverage)

    def _read_bam_fixed_length(
        self,
        coverage: zarr.Array,
        bam: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
        length: int,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.zeros((batch_size, length), self.dtype)

        with pysam.AlignmentFile(str(bam), threads=n_threads) as f:
            row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                batch[idx] = out
                if is_last_in_batch or is_last_row:
                    to_rc_mask = to_rc[start : start + idx + 1]
                    batch[to_rc_mask] = batch[to_rc_mask, ::-1]
                    coverage[start : start + idx + 1, sample_idx] = batch[: idx + 1]

    def _read_bam_variable_length(
        self,
        coverage: zarr.Array,
        bam: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
    ):
        blosc.set_nthreads(n_threads)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        batch = np.empty(batch_size, object)

        with pysam.AlignmentFile(str(bam), threads=n_threads) as f:
            row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, out, idx, start in row_batcher:
                if to_rc[idx]:
                    out = out[::-1]
                batch[idx] = out
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
        self, out: PathType, bed: pd.DataFrame, length: int, overwrite=False
    ):
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        batch_size = min(len(bed), self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            f"{self.name}_samples",
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
                self._read_bam_fixed_length(
                    coverage,
                    bam,
                    bed,
                    batch_size,
                    sample_idx,
                    self.threads_per_job,
                    length=length,
                )
                for bam, sample_idx in zip(self.bams, sample_idxs)
            )
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)

    def _write_variable_length(self, out: PathType, bed: pd.DataFrame, overwrite=False):
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        batch_size = min(len(bed), self.batch_size)
        z = zarr.open_group(out)

        arr = z.array(
            f"{self.name}_samples",
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
                self._read_bam_variable_length(
                    coverage,
                    bam,
                    bed,
                    batch_size,
                    sample_idx,
                    self.threads_per_job,
                )
                for bam, sample_idx in zip(self.bams, sample_idxs)
            )
        ]
        with joblib.parallel_backend(
            "loky", n_jobs=self.n_jobs, inner_max_num_threads=self.threads_per_job
        ):
            joblib.Parallel()(tasks)


class VCF(RegionReader):
    name: str
    vcf: Path
    fasta: Path
    contigs: List[str]

    def __init__(
        self,
        name: str,
        vcf: PathType,
        fasta: PathType,
        samples: List[str],
        batch_size: int,
        n_threads=1,
        samples_per_chunk=10,
        alphabet: Optional[Union[str, SequenceAlphabet]] = None,
    ) -> None:
        self.name = name
        self.vcf = Path(vcf)
        self.fasta = Path(fasta)
        self.samples = samples
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.samples_per_chunk = samples_per_chunk
        if alphabet is None:
            self.alphabet = ALPHABETS["DNA"]
        elif isinstance(alphabet, str):
            self.alphabet = ALPHABETS[alphabet]
        else:
            self.alphabet = alphabet

        with pysam.FastaFile(str(fasta)) as f:
            fasta_contigs = set(f.references)
        _vcf = cyvcf2.VCF(str(vcf))
        vcf_contigs = cast(Set[str], set(_vcf.seqlens))
        _vcf.close()

        self.contigs = natsorted(fasta_contigs & vcf_contigs)
        if len(self.contigs) == 0:
            raise RuntimeError("FASTA and VCF have no contigs in common.")
        contigs_exclusive_to_fasta = natsorted(fasta_contigs - vcf_contigs)
        contigs_exclusive_to_vcf = natsorted(vcf_contigs - fasta_contigs)
        if contigs_exclusive_to_fasta:
            warnings.warn(
                f"FASTA has contigs not found in VCF: {contigs_exclusive_to_fasta}"
            )
        if contigs_exclusive_to_vcf:
            warnings.warn(
                f"VCF has contigs not found in FASTA: {contigs_exclusive_to_vcf}"
            )

    def _get_pos_bases(self, v):
        # change to bytes and extract alleles
        alleles = v.gt_bases.astype("S").reshape(-1, 1).view("S1")[:, [0, 2]]
        # change unknown to reference
        alleles[alleles == "."] = v.REF
        # make position 0-indexed
        return v.POS - 1, alleles

    def _reader(
        self,
        bed: pd.DataFrame,
        f: pysam.FastaFile,
        vcf: cyvcf2.VCF,
        sample_order: NDArray[np.intp],
    ):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            start = cast(int, start)
            seq_bytes = f.fetch(contig, max(start, 0), end).encode("ascii")
            if (pad_len := end - start - len(seq_bytes)) > 0:
                pad_left = start < 0
                if pad_left:
                    seq_bytes = b"N" * pad_len + seq_bytes
                else:
                    seq_bytes += b"N" * pad_len
            seq = cast(NDArray[np.bytes_], np.frombuffer(seq_bytes, "|S1"))
            # (samples haplotypes length)
            tiled_seq = np.tile(seq, (len(self.samples), 2, 1))

            region = f"{contig}:{max(start, 0)+1}-{end}"
            positions_ls, alleles_ls = zip(
                *[self._get_pos_bases(v) for v in vcf(region) if v.is_snp]
            )
            # (variants)
            relative_positions = cast(NDArray[np.int64], np.array(positions_ls)) - start
            # (samples haplotypes variants)
            alleles = cast(NDArray[np.bytes_], np.stack(alleles_ls, -1)[sample_order])
            # (samples haplotypes variants) = (samples haplotypes variants)
            tiled_seq[..., relative_positions] = alleles
            # (samples haplotypes length)
            yield tiled_seq

    def _spliced_reader(
        self,
        bed: pd.DataFrame,
        f: pysam.FastaFile,
        vcf: cyvcf2.VCF,
        sample_order: NDArray[np.intp],
    ):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.itertuples(index=False), lambda x, y: x.name != y.name
        ):
            unspliced: List[NDArray[np.bytes_]] = []
            for row in rows:
                pbar.update()
                contig, start, end = row[:3]
                start = cast(int, start)
                seq_bytes = f.fetch(contig, max(start, 0), end).encode("ascii")
                if (pad_len := end - start - len(seq_bytes)) > 0:
                    pad_left = start < 0
                    if pad_left:
                        seq_bytes = b"N" * pad_len + seq_bytes
                    else:
                        seq_bytes += b"N" * pad_len
                seq = cast(NDArray[np.bytes_], np.frombuffer(seq_bytes, "|S1"))
                # (samples haplotypes length)
                tiled_seq = np.tile(seq, (len(self.samples), 2, 1))

                region = f"{contig}:{max(start, 0)+1}-{end}"
                positions_ls, alleles_ls = zip(
                    *[self._get_pos_bases(v) for v in vcf(region) if v.is_snp]
                )
                # (variants)
                relative_positions = (
                    cast(NDArray[np.int64], np.array(positions_ls)) - start
                )
                # (samples haplotypes variants)
                alleles = cast(
                    NDArray[np.bytes_], np.stack(alleles_ls, -1)[sample_order]
                )
                # (samples haplotypes variants) = (samples haplotypes variants)
                tiled_seq[..., relative_positions] = alleles
                unspliced.append(tiled_seq)
            # list of (samples haplotypes length)
            yield np.concatenate(unspliced, -1)

    def _write(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: Optional[int] = None,
        overwrite=False,
        splice=False,
    ) -> None:
        if length is None:
            self._write_variable_length(
                out=out, bed=bed, overwrite=overwrite, splice=splice
            )
        else:
            self._write_fixed_length(
                out=out, bed=bed, length=length, overwrite=overwrite, splice=splice
            )

    def _write_fixed_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: int,
        overwrite: bool,
        splice: bool,
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)

        z = zarr.open_group(out)

        seqs = z.empty(
            self.name,
            shape=(len(bed), len(self.samples), 2, length),
            dtype="|S1",
            chunks=(batch_size, self.samples_per_chunk, None, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = [
            "_sequence",
            f"{self.name}_sample",
            "haplotype",
            f"{self.name}_length",
        ]

        arr = z.array(
            f"{self.name}_samples",
            np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [f"{self.name}_sample"]

        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        _vcf = cyvcf2.VCF(
            self.vcf, lazy=True, samples=self.samples, threads=self.n_threads
        )
        *_, sample_order = np.intersect1d(
            _vcf.samples, self.samples, assume_unique=True, return_indices=True
        )

        # (batch samples haplotypes length)
        batch = cast(
            NDArray[np.bytes_], np.empty((batch_size, *seqs.shape[1:]), dtype="|S1")
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            row_batcher = _get_row_batcher(
                reader(bed, f, _vcf, sample_order), batch_size
            )
            for is_last_row, is_last_in_batch, seq, idx, start in tqdm(
                row_batcher, total=n_seqs
            ):
                # (samples haplotypes length)
                batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    to_rc_mask = to_rc[start : start + idx + 1]
                    batch[to_rc_mask] = self.alphabet.rev_comp_byte(batch[to_rc_mask])
                    seqs[start : start + idx + 1] = batch[: idx + 1]

        _vcf.close()

    def _write_variable_length(
        self, out: PathType, bed: pd.DataFrame, overwrite: bool, splice: bool
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)

        z = zarr.open_group(out)

        seqs = z.empty(
            self.name,
            shape=(len(bed), len(self.samples), 2),
            dtype=object,
            chunks=(batch_size, self.samples_per_chunk, None, None),
            overwrite=overwrite,
            compressor=compressor,
            object_codec=VLenBytes(),
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = [
            "_sequence",
            f"{self.name}_sample",
            "haplotype",
        ]

        arr = z.array(
            f"{self.name}_samples",
            np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [f"{self.name}_sample"]

        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        _vcf = cyvcf2.VCF(
            self.vcf, lazy=True, samples=self.samples, threads=self.n_threads
        )
        *_, sample_order = np.intersect1d(
            _vcf.samples, self.samples, assume_unique=True, return_indices=True
        )

        # (batch samples haplotypes)
        batch = cast(
            NDArray[np.bytes_], np.empty((batch_size, *seqs.shape[1:]), dtype="|S1")
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            row_batcher = _get_row_batcher(
                reader(bed, f, _vcf, sample_order), batch_size
            )
            for is_last_row, is_last_in_batch, seq, idx, start in tqdm(
                row_batcher, total=n_seqs
            ):
                # (samples haplotypes)
                if to_rc[idx]:
                    seq = self.alphabet.rev_comp_byte(seq)
                batch[idx] = np.array(
                    [
                        np.array(arr)
                        for arr in seq.view(f"|S{seq.shape[-1]}").squeeze().ravel()
                    ],
                    object,
                ).reshape(seq.shape[:-1])
                if is_last_in_batch or is_last_row:
                    seqs[start : start + idx + 1] = batch[: idx + 1]

        _vcf.close()

    def _sequence_generator(self, bed: pd.DataFrame, splice=False):
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        _vcf = cyvcf2.VCF(
            self.vcf, lazy=True, samples=self.samples, threads=self.n_threads
        )
        *_, sample_order = np.intersect1d(
            _vcf.samples, self.samples, assume_unique=True, return_indices=True
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            # (samples haplotypes length)
            for i, seqs in enumerate(reader(bed, f, _vcf, sample_order)):
                if to_rc[i]:
                    seqs = self.alphabet.rev_comp_byte(seqs)
                seqs = seqs.view(f"|S{seqs.shape[-1]}")
                for seq in seqs.ravel():
                    yield seq
