"""Annotated sequence data"""

from ._core.seqdata import (
    add_layers_from_files,
    from_files,
    get_torch_dataloader,
    open_zarr,
)
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
    "from_files",
    "open_zarr",
    "add_layers_from_files",
    "get_torch_dataloader",
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
