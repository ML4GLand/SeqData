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
