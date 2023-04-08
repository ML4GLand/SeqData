"""Annotated sequence data"""

from ._core import SeqData, concat, open_zarr
from ._io.read import (
    read_bam,
    read_bigwig,
    read_flat_fasta,
    read_genome_fasta,
    read_table,
    read_vcf,
)
from ._io.readers import BAM, VCF, BigWig, FlatFASTA, GenomeFASTA, Table
