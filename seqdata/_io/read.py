from typing import List, Optional, Type, Union

import numpy as np

from seqdata._core.seqdata import SeqData
from seqdata._io.readers import BAM, VCF, BigWig, FlatFASTA, GenomeFASTA, Table
from seqdata.alphabets import SequenceAlphabet
from seqdata.types import PathType


def read_table(
    name: str,
    out: PathType,
    tables: Union[PathType, List[PathType]],
    seq_col: str,
    batch_size: int,
    overwrite=False,
) -> SeqData:
    sdata = SeqData.from_files(
        Table(name, tables, seq_col, batch_size), path=out, overwrite=overwrite
    )
    return sdata


def read_flat_fasta(
    name: str,
    out: PathType,
    fasta: PathType,
    batch_size: int,
    n_threads=1,
    overwrite=False,
) -> SeqData:
    sdata = SeqData.from_files(
        FlatFASTA(name, fasta, batch_size, n_threads), path=out, overwrite=overwrite
    )
    return sdata


def read_genome_fasta(
    name: str,
    out: PathType,
    fasta: PathType,
    length: int,
    bed: PathType,
    batch_size: int,
    n_threads=1,
    alphabet: Optional[Union[str, SequenceAlphabet]] = None,
    max_jitter=0,
    overwrite=False,
) -> SeqData:
    sdata = SeqData.from_files(
        GenomeFASTA(name, fasta, batch_size, n_threads, alphabet),
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
    length: int,
    bed: PathType,
    batch_size: int,
    n_jobs=1,
    threads_per_job=1,
    samples_per_chunk=10,
    dtype: Union[str, Type[np.number]] = np.uint16,
    max_jitter=0,
    overwrite=False,
) -> SeqData:
    sdata = SeqData.from_files(
        BAM(
            name,
            bams,
            samples,
            batch_size,
            n_jobs,
            threads_per_job,
            samples_per_chunk,
            dtype,
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
    bams: List[PathType],
    samples: List[str],
    length: int,
    bed: PathType,
    batch_size: int,
    n_jobs=1,
    threads_per_job=1,
    samples_per_chunk=10,
    dtype: Union[str, Type[np.number]] = np.uint16,
    max_jitter=0,
    overwrite=False,
) -> SeqData:
    sdata = SeqData.from_files(
        BigWig(
            name,
            bams,
            samples,
            batch_size,
            n_jobs,
            threads_per_job,
            samples_per_chunk,
            dtype,
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
    length: int,
    bed: PathType,
    batch_size: int,
    n_threads=1,
    samples_per_chunk=10,
    alphabet: Optional[Union[str, SequenceAlphabet]] = None,
    max_jitter=0,
    overwrite=False,
) -> SeqData:
    sdata = SeqData.from_files(
        VCF(
            name,
            vcf,
            fasta,
            samples,
            batch_size,
            n_threads,
            samples_per_chunk,
            alphabet,
        ),
        path=out,
        length=length,
        bed=bed,
        max_jitter=max_jitter,
        overwrite=overwrite,
    )
    return sdata
