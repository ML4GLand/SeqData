from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

PathType = Union[str, Path]
ListPathType = Union[List[str], List[Path]]
T = TypeVar("T")
DTYPE = TypeVar("DTYPE", bound=np.generic, covariant=True)


class FlatReader(ABC):
    name: str

    @abstractmethod
    def _write(
        self,
        out: PathType,
        fixed_length: bool,
        sequence_dim: str,
        length_dim: Optional[str] = None,
        overwrite=False,
    ) -> None:
        """Write data from the reader to a SeqData Zarr on disk.

        Parameters
        ----------
        out : str, Path
            Output file, should be a `.zarr` file.
        fixed_length : bool
        sequence_dim : str
            Name of sequence dimension.
        length_dim : str
            Name of length dimension.
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
        fixed_length: Union[int, Literal[False]],
        sequence_dim: str,
        length_dim: Optional[str] = None,
        splice=False,
        overwrite=False,
    ) -> None:
        """Write data in regions specified from a BED file.

        Parameters
        ----------
        out : str, Path
            Output file, should be a `.zarr` file.
        bed : pd.DataFrame
            DataFrame corresponding to a BED file.
        fixed_length : int, bool
            `int`: length of sequences. `False`: write variable length sequences.
        sequence_dim : str
            Name of sequence dimension.
        length_dim : str, optional
            Name of length dimension. Ignored if fixed_length = False.
        splice : bool, default False
            Whether to splice together regions with the same `name` (i.e. the 4th BED
            column). For example, to splice together exons from transcripts or coding
            sequences of proteins.
        overwrite : bool, default False
            Whether to overwrite existing output file.
        """
        ...
