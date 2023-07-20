from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np
import seqpro as sp

from seqdata._io.readers import BAM, VCF, BigWig, FlatFASTA, GenomeFASTA, Table
from seqdata.types import ListPathType, PathType
from seqdata.xarray.seqdata import from_flat_files, from_region_files

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


def read_table(
    name: str,
    out: PathType,
    tables: Union[PathType, ListPathType],
    seq_col: str,
    batch_size: int,
    fixed_length: bool,
    overwrite=False,
    **kwargs
) -> "xr.Dataset":
    """Reads sequences and metadata from tabular files (e.g. CSV, TSV, etc.) into xarray.

    Uses polars under the hood to read the table files.

    Parameters
    ----------
    name : str
        Name of the sequence variable in the output dataset.
    out : PathType
        Path to the output Zarr store where the data will be saved.
        Usually something like `/path/to/dataset_name.zarr`.
    tables : Union[PathType, ListPathType]
        Path to the input table file(s). Can be a single file or a list of files.
    seq_col : str
        Name of the column in the table that contains the sequence.
    batch_size : int
        Number of sequences to read at a time. Use as many as you can fit in memory.
    fixed_length : bool
        Whether your sequences have a fixed length or not. If they do, the data will be
        stored in a 2D array as bytes, otherwise it will be stored as unicode strings.
    overwrite : bool
        Whether to overwrite the output Zarr store if it already exists.
    **kwargs
        Additional keyword arguments to pass to the polars `read_csv` function.
    
    Returns
    -------
    xr.Dataset
        The output dataset.
    """
    sdata = from_flat_files(
        Table(
            name=name, tables=tables, seq_col=seq_col, batch_size=batch_size, **kwargs
        ),
        path=out,
        fixed_length=fixed_length,
        overwrite=overwrite,
    )
    return sdata


def read_flat_fasta(
    name: str,
    out: PathType,
    fasta: PathType,
    batch_size: int,
    fixed_length: bool,
    n_threads=1,
    overwrite=False,
) -> "xr.Dataset":
    """Reads sequences from a "flat" FASTA file into xarray.
    
    We differentiate between "flat" and "genome" FASTA files. A flat FASTA file is one
    where each contig in the FASTA file is a sequence in our dataset. A genome FASTA file
    is one where we may pull out multiple subsequences from a given contig.

    Parameters
    ----------
    name : str
        Name of the sequence variable in the output dataset.
    out : PathType
        Path to the output Zarr store where the data will be saved.
        Usually something like `/path/to/dataset_name.zarr`.
    fasta : PathType
        Path to the input FASTA file.
    batch_size : int
        Number of sequences to read at a time. Use as many as you can fit in memory.
    fixed_length : bool
        Whether your sequences have a fixed length or not. If they do, the data will be
        stored in a 2D array as bytes, otherwise it will be stored as unicode strings.
    n_threads : int
        Number of threads to use for reading the FASTA file.
    overwrite : bool
        Whether to overwrite the output Zarr store if it already exists.

    Returns
    -------
    xr.Dataset
        The output dataset.
    """
    sdata = from_flat_files(
        FlatFASTA(name=name, fasta=fasta, batch_size=batch_size, n_threads=n_threads),
        path=out,
        fixed_length=fixed_length,
        overwrite=overwrite,
    )
    return sdata


def read_genome_fasta(
    name: str,
    out: PathType,
    fasta: PathType,
    bed: Union[PathType, "pd.DataFrame"],
    batch_size: int,
    fixed_length: Union[int, bool],
    n_threads=1,
    alphabet: Optional[Union[str, sp.NucleotideAlphabet]] = None,
    max_jitter=0,
    overwrite=False,
) -> "xr.Dataset":
    """Reads sequences from a "genome" FASTA file into xarray.

    We differentiate between "flat" and "genome" FASTA files. A flat FASTA file is one
    where each contig in the FASTA file is a sequence in our dataset. A genome FASTA file
    is one where we may pull out multiple subsequences from a given contig.

    Parameters
    ----------
    name : str
        Name of the sequence variable in the output dataset.
    out : PathType
        Path to the output Zarr store where the data will be saved.
        Usually something like `/path/to/dataset_name.zarr`.
    fasta : PathType
        Path to the input FASTA file.
    bed : Union[PathType, pd.DataFrame]
        Path to the input BED file or a pandas DataFrame with the BED data. Used to 
        define the regions of the genome to pull out. TODO: what does the BED
        have to have?
    batch_size : int
        Number of sequences to read at a time. Use as many as you can fit in memory.
    fixed_length : Union[int, bool]
        Whether your sequences have a fixed length or not. If they do, the data will be
        stored in a 2D array as bytes, otherwise it will be stored as unicode strings.
    n_threads : int
        Number of threads to use for reading the FASTA file.
    alphabet : Optional[Union[str, sp.NucleotideAlphabet]]
        Alphabet to use for reading sequences
    max_jitter : int
        Maximum amount of jitter anticipated. This will read in max_jitter/2 extra sequence
        on either side of the region defined by the BED file. This is useful for training 
        models on coverage data
    overwrite : bool
        Whether to overwrite the output Zarr store if it already exists.
    """
    sdata = from_region_files(
        GenomeFASTA(
            name=name,
            fasta=fasta,
            batch_size=batch_size,
            n_threads=n_threads,
            alphabet=alphabet,
        ),
        path=out,
        fixed_length=fixed_length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
    )
    return sdata


def read_bam(
    seq_name: str,
    cov_name: str,
    out: PathType,
    fasta: PathType,
    bams: ListPathType,
    samples: List[str],
    bed: Union[PathType, "pd.DataFrame"],
    batch_size: int,
    fixed_length: Union[int, bool],
    n_jobs=1,
    threads_per_job=1,
    alphabet: Optional[Union[str, sp.NucleotideAlphabet]] = None,
    dtype: Union[str, Type[np.number]] = np.uint16,
    max_jitter=0,
    overwrite=False,
) -> "xr.Dataset":
    """
    Read in sequences with coverage from a BAM file.

    Parameters
    ----------
    seq_name : str
        Name of the sequence variable in the output dataset.
    cov_name : str
        Name of the coverage variable in the output dataset.
    out : PathType
        Path to the output Zarr store where the data will be saved.
        Usually something like `/path/to/dataset_name.zarr`.
    fasta : PathType
        Path to the reference genome.
    bams : ListPathType
        List of paths to BAM files.
        Can be a single file or a list of files.
    samples : List[str]
        List of sample names to include.
        Should be the same length as `bams`.
    bed : Union[PathType, pd.DataFrame]
        Path to a BED file or a DataFrame with columns "chrom", "start", and "end".
    batch_size : int
        Number of regions to read at once. Use as many as you can fit in memory.
    fixed_length : Union[int, bool]
        Whether your sequences have a fixed length or not. If they do, the data will be
        stored in a 2D array as bytes, otherwise it will be stored as unicode strings.
    n_jobs : int
        Number of parallel jobs. Use if you have multiple BAM files.
    threads_per_job : int
        Number of threads per job.
    alphabet : Optional[Union[str, sp.NucleotideAlphabet]]
        Alphabet the sequences have.
    dtype : Union[str, Type[np.number]]
        Data type to use for coverage.
    max_jitter : int
        Maximum jitter to use for sampling regions. This will read in max_jitter/2 extra sequence
        on either side of the region defined by the BED file. This is useful for training
        models on coverage data
    overwrite : bool
        Whether to overwrite an existing dataset.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions "_sequence" TODO: what are the dimensions?
    """
    sdata = from_region_files(
        GenomeFASTA(
            name=seq_name,
            fasta=fasta,
            batch_size=batch_size,
            n_threads=n_jobs * threads_per_job,
            alphabet=alphabet,
        ),
        BAM(
            name=cov_name,
            bams=bams,
            samples=samples,
            batch_size=batch_size,
            n_jobs=n_jobs,
            threads_per_job=threads_per_job,
            dtype=dtype,
        ),
        path=out,
        fixed_length=fixed_length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
    )
    return sdata


def read_bigwig(
    seq_name: str,
    cov_name: str,
    out: PathType,
    fasta: PathType,
    bigwigs: ListPathType,
    samples: List[str],
    bed: Union[PathType, "pd.DataFrame"],
    batch_size: int,
    fixed_length: Union[int, bool],
    n_jobs=1,
    threads_per_job=1,
    alphabet: Optional[Union[str, sp.NucleotideAlphabet]] = None,
    dtype: Union[str, Type[np.number]] = np.uint16,
    max_jitter=0,
    overwrite=False,
) -> "xr.Dataset":
    """
    Read a bigWig file and return a Dataset.

    Parameters
    ----------
    seq_name : str
        Name of the sequence variable in the output dataset.
    cov_name : str
        Name of the coverage variable in the output dataset.
    out : PathType
        Path to the output Zarr store where the data will be saved.
        Usually something like `/path/to/dataset_name.zarr`.
    fasta : PathType
        Path to the reference genome.
    bigwigs : ListPathType
        List of paths to bigWig files.
        Can be a single file or a list of files.
    samples : List[str]
        List of sample names to include.
        Should be the same length as `bigwigs`.
    bed : Union[PathType, pd.DataFrame]
        Path to a BED file or a DataFrame with columns "chrom", "start", and "end".
    batch_size : int
        Number of regions to read at once. Use as many as you can fit in memory.
    fixed_length : Union[int, bool]
        Whether your sequences have a fixed length or not. If they do, the data will be
        stored in a 2D array as bytes, otherwise it will be stored as unicode strings.
    n_jobs : int
        Number of parallel jobs. Use if you have multiple bigWig files.
    threads_per_job : int
        Number of threads per job.
    alphabet : Optional[Union[str, sp.NucleotideAlphabet]]
        Alphabet the sequences have.
    dtype : Union[str, Type[np.number]]
        Data type to use for coverage.
    max_jitter : int
        Maximum jitter to use for sampling regions.
    overwrite : bool
        Whether to overwrite an existing dataset.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions "_sequence" TODO: what are the dimensions?
    """
    sdata = from_region_files(
        GenomeFASTA(
            name=seq_name,
            fasta=fasta,
            batch_size=batch_size,
            n_threads=n_jobs * threads_per_job,
            alphabet=alphabet,
        ),
        BigWig(
            name=cov_name,
            bigwigs=bigwigs,
            samples=samples,
            batch_size=batch_size,
            n_jobs=n_jobs,
            threads_per_job=threads_per_job,
            dtype=dtype,
        ),
        path=out,
        fixed_length=fixed_length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
    )
    return sdata


def read_vcf(
    name: str,
    out: PathType,
    vcf: PathType,
    fasta: PathType,
    samples: List[str],
    bed: Union[PathType, "pd.DataFrame"],
    batch_size: int,
    fixed_length: Union[int, bool],
    n_threads=1,
    samples_per_chunk=10,
    alphabet: Optional[Union[str, sp.NucleotideAlphabet]] = None,
    max_jitter=0,
    overwrite=False,
    splice=False,
) -> "xr.Dataset":
    """
    Read a VCF file and return a Dataset.

    Parameters
    ----------
    name : str
        Name of the sequence variable in the output dataset.
    out : PathType
        Path to the output Zarr store where the data will be saved.
        Usually something like `/path/to/dataset_name.zarr`.
    vcf : PathType
        Path to the VCF file.
    fasta : PathType
        Path to the reference genome.
    samples : List[str]
        List of sample names to include.
    bed : Union[PathType, pd.DataFrame]
        Path to a BED file or a DataFrame with columns "chrom", "start", and "end".
    batch_size : int
        Number of regions to read at once. Use as many as you can fit in memory.
    fixed_length : Union[int, bool]
        Whether your sequences have a fixed length or not. If they do, the data will be
        stored in a 2D array as bytes, otherwise it will be stored as unicode strings.
    n_threads : int
        Number of threads to use for reading the VCF file.
    samples_per_chunk : int
        Number of samples to read at a time.
    alphabet : Optional[Union[str, sp.NucleotideAlphabet]]
        Alphabet the sequences have.
    max_jitter : int
        Maximum jitter to use for sampling regions.
    overwrite : bool
        Whether to overwrite an existing dataset.
    splice : bool
        TODO
    Returns
    -------
    xr.Dataset
        xarray dataset
    """
    sdata = from_region_files(
        VCF(
            name=name,
            vcf=vcf,
            fasta=fasta,
            samples=samples,
            batch_size=batch_size,
            n_threads=n_threads,
            samples_per_chunk=samples_per_chunk,
            alphabet=alphabet,
        ),
        path=out,
        fixed_length=fixed_length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
        splice=splice,
    )
    return sdata
