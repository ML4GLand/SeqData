"""Annotated sequence data"""

from ._io.bed_ops import (
    add_bed_to_sdata,
    mark_sequences_for_classification,
    read_bedlike,
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
from .torch import get_torch_dataloader
from .xarray.seqdata import from_flat_files, from_region_files, open_zarr, to_zarr

__all__ = [
    "from_flat_files",
    "from_region_files",
    "open_zarr",
    "to_zarr",
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
    "add_bed_to_sdata",
    "mark_sequences_for_classification",
]
