"""Annotated sequence data"""

from ._core.seqdata import (
    from_flat_files,
    from_region_files,
    get_torch_dataloader,
    open_zarr,
)
from ._io.bed_ops import read_bedlike
from ._io.read import (
    read_bam,
    read_bigwig,
    read_flat_fasta,
    read_genome_fasta,
    read_table,
    read_vcf,
)
from ._io.readers import BAM, VCF, BigWig, FlatFASTA, GenomeFASTA, Table
from .alphabets import ALPHABETS, SequenceAlphabet

__all__ = [
    "from_flat_files",
    "from_region_files",
    "open_zarr",
    "get_torch_dataloader",
    "read_bedlike",
    "read_bam",
    "read_bigwig",
    "read_flat_fasta",
    "read_genome_fasta",
    "read_table",
    "read_vcf",
    "BAM",
    "VCF",
    "BigWig",
    "FlatFASTA",
    "GenomeFASTA",
    "Table",
    "SequenceAlphabet",
    "ALPHABETS",
]
