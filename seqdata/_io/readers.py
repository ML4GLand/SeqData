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
from natsort import natsorted
from numcodecs import Blosc, Delta, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata.alphabets import ALPHABETS, SequenceAlphabet
from seqdata.types import DTYPE, FlatReader, PathType, RegionReader

from .utils import _batch_io, _batch_io_bed, _df_to_xr_zarr

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
        self, batch: pd.DataFrame, z: zarr.Group, compressor, overwrite
    ):
        seqs = batch[[self.seq_col]].to_numpy().astype("S").view("|S1")
        obs = batch.drop(columns=self.seq_col)
        z.array(
            self.seq_col,
            data=seqs,
            chunks=(self.batch_size, None),
            compressor=compressor,
            overwrite=overwrite,
        )
        _df_to_xr_zarr(
            obs,
            z.path,
            ["sequence"],
            chunks=self.batch_size,
            compressor=compressor,
            overwrite=overwrite,
        )
        first_cols = obs.columns
        return first_cols

    def _write_batch(self, batch: pd.DataFrame, z: zarr.Group, first_cols, table: Path):
        seqs = batch[[self.seq_col]].to_numpy().astype("S").view("|S1")
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
        z[self.seq_col].append(seqs)  # type: ignore
        for name, series in obs.items():
            z[name].append(series.to_numpy())  # type: ignore  # type: ignore

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
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        with pysam.FastaFile(str(self.fasta)) as f:
            self.n_seqs = len(f.references)

    def _reader(self, f: pysam.FastaFile):
        for seq_name in tqdm(f.references, total=len(f.references)):
            seq = f.fetch(seq_name).encode("ascii")
            out = cast(NDArray[np.bytes_], np.frombuffer(seq, "|S1"))
            yield out

    def _write_row_to_batch(self, row: NDArray[np.bytes_], out: NDArray[np.bytes_]):
        row[:] = out

    def _write_batch_to_sink(
        self, sink: zarr.Array, batch: NDArray[np.bytes_], start_idx: int
    ):
        sink[start_idx : start_idx + len(batch)] = batch

    def _write(self, out: PathType, overwrite=False) -> None:
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
            arr.attrs["_ARRAY_DIMENSIONS"] = ["sequence"]

            n_seqs = len(seq_names)
            length = f.get_reference_length(seq_names[0])
            batch_size = min(n_seqs, self.batch_size)

            seqs = z.empty(
                self.name,
                shape=(n_seqs, length),
                dtype="|S1",
                chunks=(batch_size, None),
                overwrite=overwrite,
                compressor=compressor,
            )
            seqs.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length"]
            batch = np.empty((batch_size, length), dtype="|S1")

            _batch_io(
                seqs,
                batch,
                self._reader(f),
                self._write_row_to_batch,
                self._write_batch_to_sink,
            )


class GenomeFASTA(RegionReader):
    def __init__(
        self,
        name: str,
        fasta: PathType,
        batch_size: int,
        n_threads: int = 1,
        alphabet: Optional[Union[str, SequenceAlphabet]] = None,
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

    def _reader(self, bed: pd.DataFrame, f: pysam.FastaFile):
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
            contig, start, end = row[:3]
            seq = f.fetch(contig, start, end).encode("ascii")
            if (pad_len := end - start - len(seq)) > 0:
                pad_left = start < 0
                if pad_left:
                    seq = (b"N" * pad_len) + seq
                else:
                    seq += b"N" * pad_len
            out = cast(NDArray[np.bytes_], np.frombuffer(seq, "|S1"))
            yield out

    def _write_row_to_batch(self, row: NDArray[np.bytes_], out: NDArray[np.bytes_]):
        row[:] = out

    def _write_batch_to_sink(
        self, sink: zarr.Array, batch: NDArray[np.bytes_], start_idx: int
    ):
        sink[start_idx : start_idx + len(batch)] = batch

    def _write(
        self, out: PathType, length: int, bed: pd.DataFrame, overwrite=False
    ) -> None:
        blosc.set_nthreads(self.n_threads)
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        z = zarr.open_group(out)
        seqs = z.empty(
            self.name,
            shape=(n_seqs, length),
            dtype="|S1",
            chunks=(batch_size, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length"]

        batch = cast(NDArray[np.bytes_], np.empty((batch_size, length), dtype="|S1"))
        with pysam.FastaFile(str(self.fasta)) as f:
            _batch_io_bed(
                seqs,
                batch,
                self._reader(bed, f),
                self._write_row_to_batch,
                self._write_batch_to_sink,
                to_rc,
                self.alphabet,
            )


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
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
            contig, start, end = row[:3]
            intervals = cast(
                List[Tuple[int, int, Union[int, float]]],
                f.intervals(contig, start, end),
            )
            start = cast(int, start)
            yield intervals, start

    def _write_row_to_batch(
        self,
        row: NDArray[DTYPE],
        out: Tuple[List[Tuple[int, int, Union[int, float]]], int],
    ):
        intervals, start = out
        if intervals is not None:
            for interval in intervals:
                rel_start = interval[0] - start
                rel_end = interval[1] - start
                value = interval[2]
                row[rel_start:rel_end] = value

    def _read_bigwig(
        self,
        coverage: zarr.Array,
        bigwig: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
    ):
        def write_batch_to_sink(
            sink: zarr.Array, batch: NDArray[DTYPE], start_idx: int
        ):
            sink[start_idx : start_idx + len(batch), :, sample_idx] = batch

        blosc.set_nthreads(n_threads)
        length = bed.at[0, "chromEnd"] - bed.at[0, "chromStart"]
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())
        batch = np.zeros((batch_size, length), self.dtype)
        with pyBigWig.open(bigwig) as f:
            _batch_io_bed(
                coverage,
                batch,
                self._reader(bed, f),
                self._write_row_to_batch,
                write_batch_to_sink,
                to_rc,
            )

    def _write(
        self, out: PathType, length: int, bed: pd.DataFrame, overwrite=False
    ) -> None:
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
            shape=(len(bed), length, len(self.samples)),
            dtype=self.dtype,
            chunks=(batch_size, None, self.samples_per_chunk),
            overwrite=overwrite,
            compressor=compressor,
            filters=[Delta(self.dtype)],
        )
        coverage.attrs["_ARRAY_DIMENSIONS"] = [
            "sequence",
            "length",
            f"{self.name}_sample",
        ]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(
                self._read_bigwig(
                    coverage, bigwig, bed, batch_size, sample_idx, self.threads_per_job
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
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
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

    def _write_row_to_batch(self, row: NDArray[DTYPE], out: NDArray[DTYPE]):
        row[:] = out

    def _read_bam(
        self,
        coverage: zarr.Array,
        bam: PathType,
        bed: pd.DataFrame,
        batch_size: int,
        sample_idx: int,
        n_threads: int,
    ):
        def write_batch_to_sink(
            sink: zarr.Array, batch: NDArray[DTYPE], start_idx: int
        ):
            sink[start_idx : start_idx + len(batch), :, sample_idx] = batch

        blosc.set_nthreads(n_threads)
        length = bed.at[0, "chromEnd"] - bed.at[0, "chromStart"]
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())
        batch = np.zeros((batch_size, length), self.dtype)
        with pysam.AlignmentFile(str(bam), threads=n_threads) as f:
            _batch_io_bed(
                coverage,
                batch,
                self._reader(bed, f),
                self._write_row_to_batch,
                write_batch_to_sink,
                to_rc,
            )

    def _write(
        self, out: PathType, length: int, bed: pd.DataFrame, overwrite=False
    ) -> None:
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
            shape=(len(bed), length, len(self.samples)),
            dtype=self.dtype,
            chunks=(batch_size, None, self.samples_per_chunk),
            overwrite=overwrite,
            compressor=compressor,
            filters=[Delta(self.dtype)],
        )
        coverage.attrs["_ARRAY_DIMENSIONS"] = [
            "sequence",
            "length",
            f"{self.name}_sample",
        ]

        sample_idxs = np.arange(len(self.samples))
        tasks = [
            joblib.delayed(
                self._read_bam(
                    coverage, bam, bed, batch_size, sample_idx, self.threads_per_job
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
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
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
            # (length samples haplotypes)
            tiled_seq = np.tile(seq, (1, len(self.samples), 2))

            region = f"{contig}:{max(start, 0)+1}-{end}"
            positions_ls, alleles_ls = zip(
                *[self._get_pos_bases(v) for v in vcf(region) if v.is_snp]
            )
            # (variants)
            relative_positions = cast(NDArray[np.int64], np.array(positions_ls)) - start
            # (variants samples haplotypes)
            alleles = cast(
                NDArray[np.bytes_], np.stack(alleles_ls, 0)[:, sample_order, :]
            )
            # (variants samples haplotypes) = (variants samples haplotypes)
            tiled_seq[relative_positions] = alleles
            yield tiled_seq

    def _write_row_to_batch(self, row: NDArray[np.bytes_], out: NDArray[np.bytes_]):
        # (length samples haplotypes) = (length samples haplotypes)
        row[:] = out

    def _write_batch_to_sink(
        self, sink: zarr.Array, batch: NDArray[np.bytes_], start_idx: int
    ):
        # (batch length samples haplotypes) = (batch length samples haplotypes)
        sink[start_idx : start_idx + len(batch)] = batch

    def _write(
        self, out: PathType, length: int, bed: pd.DataFrame, overwrite=False
    ) -> None:
        blosc.set_nthreads(self.n_threads)
        compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)

        z = zarr.open_group(out)
        seqs = z.empty(
            self.name,
            shape=(n_seqs, length, len(self.samples), 2),
            dtype="|S1",
            chunks=(self.batch_size, None, self.samples_per_chunk, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = [
            "sequence",
            "length",
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
        # (sequences length samples haplotypes)
        batch = cast(
            NDArray[np.bytes_],
            np.empty((batch_size, length, len(self.samples), 2), dtype="|S1"),
        )
        with pysam.FastaFile(str(self.fasta)) as f:
            _batch_io_bed(
                seqs,
                batch,
                self._reader(bed, f, _vcf, sample_order),
                self._write_row_to_batch,
                self._write_batch_to_sink,
                to_rc,
                self.alphabet,
            )

        _vcf.close()
