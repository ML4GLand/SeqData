from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

PathType = Union[str, Path]
T = TypeVar("T")
DTYPE = TypeVar("DTYPE", bound=np.generic, covariant=True)


class FlatReader(ABC):
    name: str
    n_seqs: Optional[int]

    @abstractmethod
    def _write(self, out: PathType, overwrite=False) -> None:
        """Write data from the reader to a SeqData Zarr on disk.

        Parameters
        ----------
        out : str, Path
            Output file, should be a `.zarr` file.
        overwrite : bool, default False
            Whether to overwrite existing output file.
        """
        ...


class RegionReader(ABC):
    name: str

    @abstractmethod
    def _write(
        self,
        out: PathType,
        bed: "pd.DataFrame",
        length: Optional[int] = None,
        overwrite=False,
        splice=False,
    ) -> None:
        """Write data in regions specified from a BED file.

        Parameters
        ----------
        out : str, Path
            Output file, should be a `.zarr` file.
        bed : pd.DataFrame
            DataFrame corresponding to a BED file.
        length : int, default None
            Length of regions to write. If not specified, will write variable length
            sequences. If specified, will write uniform length sequences centered at
            each region.
        overwrite : bool, default False
            Whether to overwrite existing output file.
        splice : bool, default False
            Whether to splice together regions with the same `name` (i.e. the 4th BED
            column). For example, to splice together exons from transcripts or coding
            sequences of proteins.
        """
        ...
