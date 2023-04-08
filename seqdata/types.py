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
    ) -> None:
        ...
