"""Annotated sequence data"""

from ._core.seqdata import SeqData
from ._core.merge import concat
from ._io.read import (
    read_csv,
    read_fasta,
    read_bed,
    read_h5sd,
    read
)