from pathlib import Path
from typing import List, Optional, Union, cast

import joblib
import numba
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import zarr
from cyvcf2 import VCF
from numcodecs import Delta, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from .. import SeqData
from .utils import (
    _df_to_xr_zarr,
    _read_and_concat_dataframes,
    _read_bedlike,
    _set_uniform_length_around_center,
)

PathType = Union[str, Path]


def read_csvs(
    filename: Union[PathType, List[PathType]],
    seq_col: Optional[str] = None,
    name_col: Optional[str] = None,
    sep: str = "\t",
    low_memory: bool = False,
    col_names: Optional[Union[str, List[str]]] = None,
    compression: str = "infer",
    **kwargs,
):
    """Read sequences and sequence metadata into SeqData object from csv/tsv files.

    Parameters
    ----------
    file : str, Path
        File path(s) to read the data from. If a list is provided, the files will be concatenated (must have the same columns).
    seq_col : str, optional
        Column name containing sequences. Defaults to None.
    name_col : str, optional
        Column name containing identifiers. Defaults to None.
    sep : str, optional
        Delimiter to use. Defaults to "\\t".
    low_memory : bool, optional
        Whether to use low memory mode. Defaults to False.
    return_numpy : bool, optional
        Whether to return numpy arrays. Defaults to False.
    return_dataframe : bool, optional
        Whether to return pandas dataframe. Defaults to False.
    col_names : Iterable, optional
        Column names to use. Defaults to None. If not provided, uses first line of file.
    auto_name : bool, optional
        Whether to automatically generate identifiers. Defaults to True.
    compression : str, optional
        Compression type to use. Defaults to "infer".
    **kwargs : kwargs, dict
        Keyword arguments to pass to pandas.read_csv. Defaults to {}.

    Returns
    -------
    sdata : SeqData
        Returns SeqData object containing sequences and identifiers by default
    tuple :
        Returns numpy arrays of identifiers, sequences, reverse complement sequences and annots.
        If return_numpy is True. If any are not provided they are set to none.
    dataframe : pandas.DataFrame
        Returns pandas dataframe containing sequences and identifiers if return_dataframe is True.
    """
    dataframe = _read_and_concat_dataframes(
        file_names=filename,
        col_names=col_names,
        sep=sep,
        low_memory=low_memory,
        compression=compression,
        **kwargs,
    )
    if seq_col is not None and seq_col in dataframe.columns:
        seqs = dataframe[seq_col].to_numpy(dtype="U").reshape(-1, 1).view("U1")
        dataframe.drop(seq_col, axis=1, inplace=True)
    else:
        seqs = None

    ds = dataframe.rename_axis("sequence").to_xarray()
    del ds.coords["sequence"]
    if seqs is not None:
        ds["seqs"] = (("sequence", "length"), seqs)
    if name_col is None:
        n_digits = len(str(len(dataframe) - 1))
        names = np.array([f"seq{i:0{n_digits}}" for i in range(len(dataframe))])
        ds["names"] = ("sequence", names)

    return SeqData(ds)


def read_fasta(
    name: str,
    fasta: PathType,
    out: PathType,
    batch_size: int,
    n_threads: int = 1,
    overwrite=False,
):
    """Write a SeqData from a FASTA file where each contig is a sequence.

    Parameters
    ----------
    name : str
        Name of array for FASTA sequences
    fasta_path : str, Path
    seqdata_path : str, Path
    batch_size : int
        Make this as large as RAM allows. How many sequences to IO at a time.
    """
    blosc.set_nthreads(n_threads)
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

    z = zarr.open_group(out)
    with pysam.FastaFile(str(fasta)) as f:
        seq_names = f.references

        arr = z.array(
            f"{name}_id", data=np.array(list(seq_names), object), overwrite=overwrite
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["sequence"]

        n_seqs = len(seq_names)
        length = f.get_reference_length(seq_names[0])
        batch_size = min(n_seqs, batch_size)

        seqs = z.empty(
            name,
            shape=(n_seqs, length),
            dtype="|S1",
            chunks=(batch_size, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length"]
        batch = np.empty((batch_size, length), dtype="|S1")
        batch_start_idx = 0
        batch_idx = 0
        for contig in seq_names:
            batch[batch_idx] = np.frombuffer(f.fetch(contig).encode("ascii"), "|S1")
            if batch_idx == batch_size - 1:
                seqs[batch_start_idx : batch_start_idx + batch_size] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        seqs[batch_start_idx : batch_start_idx + batch_idx] = batch[:batch_idx]

    zarr.consolidate_metadata(out)  # type: ignore


def read_genome_fasta(
    name: str,
    fasta: PathType,
    bed: PathType,
    out: PathType,
    length: int,
    batch_size: int,
    n_threads: int = 1,
    overwrite=False,
):
    blosc.set_nthreads(n_threads)
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

    _bed = _read_bedlike(bed)
    _set_uniform_length_around_center(_bed, length)
    _df_to_xr_zarr(_bed, out, ["sequence"], compressor=compressor, overwrite=overwrite)

    n_seqs = len(_bed)
    batch_size = min(n_seqs, batch_size)

    z = zarr.open_group(out)
    with pysam.FastaFile(str(fasta)) as f:
        seqs = z.empty(
            name,
            shape=(n_seqs, length),
            dtype="|S1",
            chunks=(batch_size, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length"]
        batch = np.empty((batch_size, length), dtype="|S1")
        batch_start_idx = 0
        batch_idx = 0
        for i, row in tqdm(_bed.iterrows()):
            contig, start, end = row[:3]
            batch[batch_idx] = np.frombuffer(
                f.fetch(contig, start, end).encode("ascii"), "|S1"
            )
            if batch_idx == batch_size - 1:
                seqs[batch_start_idx : batch_start_idx + batch_size] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        seqs[batch_start_idx : batch_start_idx + batch_idx] = batch[:batch_idx]

    zarr.consolidate_metadata(out)  # type: ignore


def _read_bigwig(
    coverage: zarr.Array,
    bigwig: PathType,
    bed: pd.DataFrame,
    batch_size: int,
    sample_idx: int,
    n_threads: int,
):
    blosc.set_nthreads(n_threads)

    length = bed.at[0, "chromEnd"] - bed.at[0, "chromStart"]

    batch = np.zeros((batch_size, length), np.uint16)

    batch_start_idx = 0
    batch_idx = 0
    with pyBigWig.open(bigwig) as f:
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
            contig, start, end = row[:3]
            intervals = f.intervals(contig, start, end)
            if intervals is not None:
                for interval in intervals:
                    rel_start = interval[0] - start
                    rel_end = interval[1] - start
                    value = interval[2]
                    batch[batch_idx, rel_start:rel_end] = value
            if batch_idx == batch_size - 1:
                coverage[
                    batch_start_idx : batch_start_idx + batch_size, :, sample_idx
                ] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        coverage[batch_start_idx : batch_start_idx + batch_idx, :, sample_idx] = batch[
            :batch_idx
        ]


def read_bigwigs(
    name: str,
    bigwigs: List[PathType],
    samples: List[str],
    bed: PathType,
    out: PathType,
    length: int,
    batch_size: int,
    n_jobs: int = 1,
    threads_per_job: int = 1,
    overwrite=False,
):
    """Read multiple bigWigs into a SeqData Zarr.

    Parameters
    ----------
    name : str
        Name of the array in the Zarr.
    bigwig_paths : list[str | Path]
        List of bigWigs to process.
    sample_names : list[str]
        List of sample names corresponding to each bigWig.
    bed_path : str | Path
        Path to BED file specifying regions to write coverage for.
        Centered and uniform length regions will be pulled from this.
    seqdata_path : str | Path
        Path to write SeqData Zarr.
    length : int
        Length of regions of interest.
    batch_size : int
        Number of regions of interest to process at a time. Set this to be as large as RAM allows.
    n_jobs : int, optional
        Number of jobs for parallel processing, by default 1
    threads_per_job : int, optional
        Number of threads per job, by default 1
    """
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

    _bed = _read_bedlike(bed)
    _set_uniform_length_around_center(_bed, length)
    _df_to_xr_zarr(_bed, out, ["sequence"], compressor=compressor, overwrite=overwrite)

    batch_size = min(len(_bed), batch_size)
    z = zarr.open_group(out)

    arr = z.array(
        f"{name}_samples",
        data=np.array(samples, object),
        compressor=compressor,
        overwrite=overwrite,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = [f"{name}_sample"]

    coverage = z.zeros(
        name,
        shape=(len(_bed), length, len(samples)),
        dtype=np.uint16,
        chunks=(batch_size, None),
        overwrite=overwrite,
        compressor=compressor,
        filters=[Delta(np.uint16)],
    )
    coverage.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length", f"{name}_sample"]

    sample_idxs = np.arange(len(samples))
    tasks = [
        joblib.delayed(
            _read_bigwig(
                coverage, bigwig, _bed, batch_size, sample_idx, threads_per_job
            )
            for bigwig, sample_idx in zip(bigwigs, sample_idxs)
        )
    ]
    with joblib.parallel_backend(
        "loky", n_jobs=n_jobs, inner_max_num_threads=threads_per_job
    ):
        joblib.Parallel()(tasks)

    zarr.consolidate_metadata(out)  # type: ignore


def _read_bam(
    coverage: zarr.Array,
    bam: PathType,
    bed: pd.DataFrame,
    batch_size: int,
    sample_idx: int,
    n_threads: int,
):
    blosc.set_nthreads(n_threads)

    length = bed.at[0, "chromEnd"] - bed.at[0, "chromStart"]

    batch = np.zeros((batch_size, length), np.uint16)

    batch_start_idx = 0
    batch_idx = 0
    with pysam.AlignmentFile(str(bam), threads=n_threads) as f:
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
            contig, start, end = row[:3]
            a, c, g, t = f.count_coverage(contig, start, end, read_callback="all")
            batch[batch_idx] = np.vstack([a, c, g, t]).sum(0).astype(np.uint16)
            if batch_idx == batch_size - 1:
                coverage[
                    batch_start_idx : batch_start_idx + batch_size, :, sample_idx
                ] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        coverage[batch_start_idx : batch_start_idx + batch_idx, :, sample_idx] = batch[
            :batch_idx
        ]


def read_bams(
    name: str,
    bams: List[PathType],
    samples: List[str],
    bed: PathType,
    out: PathType,
    length: int,
    batch_size: int,
    n_jobs: int = 1,
    threads_per_job: int = 1,
    overwrite=False,
):
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

    _bed = _read_bedlike(bed)
    _set_uniform_length_around_center(_bed, length)
    _df_to_xr_zarr(_bed, out, ["sequence"], compressor=compressor, overwrite=overwrite)

    batch_size = min(len(_bed), batch_size)
    z = zarr.open_group(out)

    arr = z.array(
        f"{name}_samples",
        data=np.array(samples, object),
        compressor=compressor,
        overwrite=overwrite,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = [f"{name}_sample"]

    coverage = z.zeros(
        name,
        shape=(len(_bed), length, len(samples)),
        dtype=np.uint16,
        chunks=(batch_size, None),
        overwrite=overwrite,
        compressor=compressor,
        filters=[Delta(np.uint16)],
    )
    coverage.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length", f"{name}_sample"]

    sample_idxs = np.arange(len(samples))
    tasks = [
        joblib.delayed(
            _read_bam(coverage, bam, _bed, batch_size, sample_idx, threads_per_job)
            for bam, sample_idx in zip(bams, sample_idxs)
        )
    ]
    with joblib.parallel_backend(
        "loky", n_jobs=n_jobs, inner_max_num_threads=threads_per_job
    ):
        joblib.Parallel()(tasks)

    zarr.consolidate_metadata(out)  # type: ignore


@numba.jit(nopython=True, nogil=True, parallel=True)
def _apply_variants(
    seqs: NDArray[np.bytes_],
    positions: NDArray[np.integer],
    alleles: NDArray[np.bytes_],
    offsets: NDArray[np.unsignedinteger],
):
    # shapes:
    # seqs (i l)
    # variants (v)
    # positions (v)
    # offsets (i+1)

    for i in numba.prange(len(seqs)):
        i_vars = alleles[offsets[i] : offsets[i + 1]]
        i_pos = positions[offsets[i] : offsets[i + 1]]
        seq = seqs[i]
        seq[i_pos] = i_vars


def read_vcfs(
    name: str,
    vcf: PathType,
    fasta: PathType,
    bed: PathType,
    out: PathType,
    samples: List[str],
    length: int,
    batch_size: int,
    n_threads: int = 1,
    samples_per_chunk: int = 10,
    overwrite=False,
):
    raise NotImplementedError
    blosc.set_nthreads(n_threads)
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)

    _bed = _read_bedlike(bed)
    _set_uniform_length_around_center(_bed, length)
    _df_to_xr_zarr(_bed, out, ["sequence"], compressor=compressor, overwrite=overwrite)

    n_seqs = len(_bed)

    z = zarr.open_group(out)
    seqs = z.empty(
        name,
        shape=(n_seqs, length, len(samples), 2),
        dtype="|S1",
        chunks=(batch_size, None, samples_per_chunk, None),
        overwrite=overwrite,
        compressor=compressor,
    )
    seqs.attrs["_ARRAY_DIMENSIONS"] = [
        "sequence",
        "length",
        f"{name}_sample",
        "haplotype",
    ]

    arr = z.array(
        f"{name}_samples",
        np.array(samples, object),
        compressor=compressor,
        overwrite=overwrite,
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = [f"{name}_sample"]

    n_seqs = len(_bed)
    batch_size = min(n_seqs, batch_size)

    def get_pos_bases(v):
        # change to bytes and extract alleles
        alleles = v.gt_bases.astype("S").reshape(-1, 1).view("S1")[:, [0, 2]]
        # change unknown to reference
        alleles[alleles == "."] = v.REF
        # make position 0-indexed
        return v.POS - 1, alleles

    _vcf = VCF(vcf, lazy=True, samples=samples, threads=n_threads)
    *_, sample_order = np.intersect1d(
        _vcf.samples, samples, assume_unique=True, return_indices=True
    )
    _positions = []
    _alleles = []
    counts = np.empty(batch_size, "u4")
    # (sequences length samples haplotypes)
    batch = cast(
        NDArray[np.bytes_], np.empty((batch_size, length, len(samples), 2), dtype="|S1")
    )
    batch_start_idx = 0
    batch_idx = 0
    with pysam.FastaFile(str(fasta)) as f:
        for i, row in tqdm(_bed.iterrows()):
            contig, start, end = row[:3]

            # set reference sequence
            # (length)
            seq = np.frombuffer(f.fetch(contig, start, end).encode("ascii"), "|S1")
            # (length samples haplotypes)
            batch[batch_idx] = np.tile(seq, (1, len(samples), 2))

            # get variant positions & alleles and how many are in each region
            region = f"{contig}:{start+1}-{end}"
            _q_positions, _q_alleles = zip(
                *[get_pos_bases(v) for v in _vcf(region) if v.is_snp]
            )
            # (variants)
            q_positions = cast(NDArray[np.int64], np.array(_q_positions)) - start
            # (variants samples haplotypes)
            q_alleles = cast(
                NDArray[np.bytes_], np.stack(_q_alleles, 0)[:, sample_order, :]
            )
            _positions.append(q_positions)
            _alleles.append(q_alleles)
            counts[batch_idx] = len(q_positions)

            if batch_idx == batch_size - 1:
                positions = np.concatenate(_positions)
                alleles = np.concatenate(_alleles)
                offsets = np.zeros(len(counts) + 1, dtype=counts.dtype)
                counts.cumsum(out=offsets[1:])

                _apply_variants(batch, positions, alleles, offsets)

                seqs[batch_start_idx : batch_start_idx + batch_size] = batch

                _positions = []
                _alleles = []
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        last_batch = batch[:batch_idx]

        positions = np.concatenate(_positions)
        alleles = np.concatenate(_alleles)
        offsets = np.zeros(batch_idx, dtype=counts.dtype)
        counts[:batch_idx].cumsum(out=offsets[1:])

        _apply_variants(last_batch, positions, alleles, offsets)

        seqs[batch_start_idx : batch_start_idx + batch_idx] = last_batch

    _vcf.close()

    zarr.consolidate_metadata(out)  # type: ignore
