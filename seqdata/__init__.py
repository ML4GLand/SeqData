"""Annotated sequence data"""
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "seqdata"
__version__ = importlib_metadata.version(package_name)

from ._io.bed_ops import add_bed_to_sdata, label_overlapping_regions, read_bedlike

from ._io.read import (
    read_bam,
    read_bigwig,
    read_flat_fasta,
    read_genome_fasta,
    read_table,
    read_vcf,
)
from ._io.readers import BAM, VCF, BigWig, FlatFASTA, GenomeFASTA, Table
from .xarray.seqdata import (
    from_flat_files,
    from_region_files,
    merge_obs,
    open_zarr,
    to_zarr,
)

try:
    from .torch import get_torch_dataloader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    def no_torch():
        raise ImportError(
            "Install PyTorch to use functionality from SeqData's torch submodule."
        )
    get_torch_dataloader = no_torch

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
    "label_overlapping_regions",
    "merge_obs",
]
