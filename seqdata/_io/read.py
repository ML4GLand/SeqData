from typing import TYPE_CHECKING, List, Optional, Type, Union

import numpy as np

from seqdata._core.seqdata import from_files
from seqdata._io.readers import BAM, VCF, BigWig, FlatFASTA, GenomeFASTA, Table
from seqdata.alphabets import SequenceAlphabet
from seqdata.types import PathType

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


def read_table(
    name: str,
    out: PathType,
    tables: Union[PathType, List[PathType]],
    seq_col: str,
    batch_size: int,
    overwrite=False,
) -> "xr.Dataset":
    sdata = from_files(
        Table(name=name, tables=tables, seq_col=seq_col, batch_size=batch_size),
        path=out,
        overwrite=overwrite,
    )
    return sdata


def read_flat_fasta(
    name: str,
    out: PathType,
    fasta: PathType,
    batch_size: int,
    n_threads=1,
    overwrite=False,
) -> "xr.Dataset":
    sdata = from_files(
        FlatFASTA(name=name, fasta=fasta, batch_size=batch_size, n_threads=n_threads),
        path=out,
        overwrite=overwrite,
    )
    return sdata


def read_genome_fasta(
    name: str,
    out: PathType,
    fasta: PathType,
    bed: PathType,
    batch_size: int,
    length: Optional[int] = None,
    n_threads=1,
    alphabet: Optional[Union[str, SequenceAlphabet]] = None,
    max_jitter=0,
    overwrite=False,
) -> "xr.Dataset":
    sdata = from_files(
        GenomeFASTA(
            name=name,
            fasta=fasta,
            batch_size=batch_size,
            n_threads=n_threads,
            alphabet=alphabet,
        ),
        path=out,
        length=length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
    )
    return sdata


def read_bam(
    name: str,
    out: PathType,
    bams: List[PathType],
    samples: List[str],
    bed: PathType,
    batch_size: int,
    length: Optional[int] = None,
    n_jobs=1,
    threads_per_job=1,
    samples_per_chunk=10,
    dtype: Union[str, Type[np.number]] = np.uint16,
    max_jitter=0,
    overwrite=False,
) -> "xr.Dataset":
    sdata = from_files(
        BAM(
            name=name,
            bams=bams,
            samples=samples,
            batch_size=batch_size,
            n_jobs=n_jobs,
            threads_per_job=threads_per_job,
            samples_per_chunk=samples_per_chunk,
            dtype=dtype,
        ),
        path=out,
        length=length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
    )
    return sdata


def read_bigwig(
    name: str,
    out: PathType,
    bigwigs: List[PathType],
    samples: List[str],
    bed: PathType,
    batch_size: int,
    length: Optional[int] = None,
    n_jobs=1,
    threads_per_job=1,
    samples_per_chunk=10,
    dtype: Union[str, Type[np.number]] = np.uint16,
    max_jitter=0,
    overwrite=False,
) -> "xr.Dataset":
    sdata = from_files(
        BigWig(
            name=name,
            bigwigs=bigwigs,
            samples=samples,
            batch_size=batch_size,
            n_jobs=n_jobs,
            threads_per_job=threads_per_job,
            samples_per_chunk=samples_per_chunk,
            dtype=dtype,
        ),
        path=out,
        length=length,
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
    length: Optional[int] = None,
    n_threads=1,
    samples_per_chunk=10,
    alphabet: Optional[Union[str, SequenceAlphabet]] = None,
    max_jitter=0,
    overwrite=False,
    splice=False,
) -> "xr.Dataset":
    sdata = from_files(
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
        length=length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
        splice=splice,
    )
    return sdata
