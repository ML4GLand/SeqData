from ast import Raise
import numpy as np
import pandas as pd
import pyranges as pr
from typing import Any, Union, Optional, List
from typing import Dict, Iterable, Sequence, Mapping
from os import PathLike
from collections import OrderedDict
from functools import singledispatch
from pandas.api.types import is_string_dtype
from copy import deepcopy
Index1D = Union[slice, int, str, np.int64, np.ndarray]
from .utils import _gen_dataframe, convert_to_dict


class SeqData:
    """SeqData object used to containerize and store data for EUGENe workflows.
    
    Attributes
    ----------
    seqs : np.ndarray
        Numpy array of sequences.
    names : np.ndarray
        Numpy array of names.
    seqs_annot : pr.PyRanges
        PyRanges object or dict of sequences annotations.
    pos_annot : pr.PyRanges
        PyRanges object or dict of positions annotations.
    seqsm : np.ndarray
        Numpy array of sequences or dict of sequences.
    uns : dict
        Dict of additional/unstructured information.
    rev_seqs : np.ndarray
        Numpy array of reverse complement sequences.
    seqsidx : Index1D
        Index of sequences to use.
    ohe_seqs : np.ndarray
        Numpy array of one-hot encoded sequences.
    ohe_rev_seqs : np.ndarray
        Numpy array of one-hot encoded reverse complement sequences.
    """

    def __init__(
        self,
        seqs: np.ndarray = None,
        names: np.ndarray = None,
        seqs_annot: Optional[Union[pd.DataFrame, Mapping[str, Iterable[Any]]]] = None,
        pos_annot: Union[pr.PyRanges, Dict, str] = None,
        seqsm: Optional[Union[np.ndarray, Mapping[str, Sequence[Any]]]] = None,
        uns: Optional[Mapping[str, Any]] = None,
        rev_seqs: np.ndarray = None,
        ohe_seqs: np.ndarray = None,
        ohe_rev_seqs: np.ndarray = None,
        seqidx: Index1D = None,
    ):  
        # Set up the sequence index
        if seqidx is not None and len(seqidx) > 0:
            self.seqidx = seqidx
        elif seqs is not None:
            self.seqidx = range(seqs.shape[0])
        elif ohe_seqs is not None:
            self.seqidx = range(ohe_seqs.shape[0])
        elif seqs_annot is not None:
            self.seqidx = range(seqs_annot.shape[0])
        else:
            raise ValueError("No sequences or sequence metadata provided.")

        # Sequence representations
        self.seqs = np.array(seqs[self.seqidx]) if seqs is not None else None
        self.names = np.array(names[self.seqidx]) if names is not None else None
        self.rev_seqs = np.array(rev_seqs[self.seqidx]) if rev_seqs is not None else None
        self.ohe_seqs = np.array(ohe_seqs[self.seqidx]) if ohe_seqs is not None else None
        self.ohe_rev_seqs = np.array(ohe_rev_seqs[self.seqidx]) if ohe_rev_seqs is not None else None

        # n_obs
        self._n_obs = len(self.seqidx)

        # seq_annot (handled by gen dataframe)
        if isinstance(self.seqidx, slice):
            self.seqs_annot = _gen_dataframe(
                seqs_annot, self._n_obs, ["obs_names", "row_names"]
            )[self.seqidx]
        elif type(self.seqidx[0]) in [bool, np.bool_]:
            self.seqs_annot = _gen_dataframe(
                seqs_annot, self._n_obs, ["obs_names", "row_names"]
            ).loc[self.seqidx]
        else:
            self.seqs_annot = _gen_dataframe(
                seqs_annot, self._n_obs, ["obs_names", "row_names"]
            ).iloc[self.seqidx]

        # pos_annot
        if isinstance(pos_annot, dict):
            self.pos_annot = pr.from_dict(pos_annot)
        elif isinstance(pos_annot, str):
            self.pos_annot = pr.read_bed(pos_annot)
        else:
            self.pos_annot = pos_annot

        # uns
        self.uns = uns or OrderedDict()

        # seqsm TODO: Think about consequences of making obsm a group in hdf
        if seqsm is not None:
            seqsm = seqsm.copy()
            for key in seqsm:
                seqsm[key] = seqsm[key][self.seqidx]
        self.seqsm = convert_to_dict(seqsm)
            
    @property
    def seqs(self) -> np.ndarray:
        """np.ndarray: Numpy array of string representation of sequences."""
        return self._seqs

    @seqs.setter
    def seqs(self, seqs: np.ndarray):
        self._seqs = seqs

    @property
    def names(self) -> np.ndarray:
        """np.ndarray: Numpy array of names or identifiers of sequences."""
        return self._names

    @names.setter
    def names(self, names: np.ndarray):
        self._names = names

    @property
    def n_obs(self) -> int:
        """int: Number of sequences contained in the object."""
        return self._n_obs

    @property
    def rev_seqs(self) -> np.ndarray:
        """np.ndarray: Numpy array of reverse complement sequences."""
        return self._rev_seqs

    @rev_seqs.setter
    def rev_seqs(self, rev_seqs: np.ndarray):
        self._rev_seqs = rev_seqs

    @property
    def ohe_seqs(self) -> np.ndarray:
        """np.ndarray: Numpy array of one-hot encoded sequences."""
        return self._ohe_seqs

    @ohe_seqs.setter
    def ohe_seqs(self, ohe_seqs: np.ndarray):
        self._ohe_seqs = ohe_seqs

    @property
    def seqs_annot(self) -> pd.DataFrame:
        """pd.DataFrame: Pandas dataframe of per sequence annotations."""
        return self._seqs_annot

    @seqs_annot.setter
    def seqs_annot(self, seqs_annot: Union[pd.DataFrame, Mapping[str, Iterable[Any]]]):
        self._seqs_annot = _gen_dataframe(
            seqs_annot, self._n_obs, ["obs_names", "row_names"]
        )

    @property
    def pos_annot(self) -> pr.PyRanges:
        """pr.PyRanges: PyRanges object of per sequence annotations.""" 
        return self._pos_annot

    @pos_annot.setter
    def pos_annot(self, pos_annot: pr.PyRanges):
        self._pos_annot = pos_annot

    @property
    def ohe_rev_seqs(self) -> np.ndarray:
        """np.ndarray: Numpy array of one-hot encoded reverse complement sequences."""
        return self._ohe_rev_seqs

    @ohe_rev_seqs.setter
    def ohe_rev_seqs(self, ohe_rev_seqs: np.ndarray):
        self._ohe_rev_seqs = ohe_rev_seqs

    @property
    def seqsm(self) -> Mapping[str, Sequence[Any]]:
        """Mapping[str, Sequence[Any]]: Dictionary of multidimensional sequence representations."""
        return self._seqsm

    @seqsm.setter
    def seqsm(self, seqsm: Mapping[str, Sequence[Any]]):
        self._seqsm = seqsm

    @property
    def uns(self) -> Mapping[str, Any]:
        """Mapping[str, Any]: Dictionary of unstructured annotations."""
        return self._uns

    @uns.setter
    def uns(self, uns: Mapping[str, Any]):
        self._uns = uns

    def __getitem__(self, index):
        """Get item from data. Defines slicing of object."""
        if isinstance(index, str):
            return self.seqs_annot[index]
        elif isinstance(index, slice):
            index = np.arange(self.n_obs)[index]
            return SeqData(
                seqs=self.seqs,
                names=self.names,
                rev_seqs=self.rev_seqs,
                seqs_annot=self.seqs_annot,
                pos_annot=self.pos_annot,
                ohe_seqs=self.ohe_seqs,
                ohe_rev_seqs=self.ohe_rev_seqs,
                seqsm=self.seqsm,
                uns=self.uns,
                seqidx=index,
            )
        else:
            return SeqData(
                seqs=self.seqs,
                names=self.names,
                rev_seqs=self.rev_seqs,
                seqs_annot=self.seqs_annot,
                pos_annot=self.pos_annot,
                ohe_seqs=self.ohe_seqs,
                ohe_rev_seqs=self.ohe_rev_seqs,
                seqsm=self.seqsm,
                uns=self.uns,
                seqidx=index,
            )

    def __setitem__(self, index, value):
        """Add a column to seqs_annot."""
        if isinstance(index, str):
            self.seqs_annot[index] = value
        else:
            raise ValueError(
                "SeqData only supports setting seq_annot columns with indexing."
            )

    def __repr__(self):
        """Representation of SeqData object."""
        descr = f"SeqData object with = {self._n_obs} seqs"
        for attr in [
            "seqs",
            "names",
            "rev_seqs",
            "ohe_seqs",
            "ohe_rev_seqs",
            "seqs_annot",
            "pos_annot",
            "seqsm",
            "uns",
        ]:
            if attr in ["seqs", "names", "rev_seqs", "ohe_seqs", "ohe_rev_seqs"]:
                if getattr(self, attr) is not None:
                    descr += f"\n{attr} = {getattr(self, attr).shape}"
                else:
                    descr += f"\n{attr} = None"
            elif attr in ["seqs_annot"]:
                keys = getattr(self, attr).keys()
                if len(keys) > 0:
                    descr += f"\n{attr}: {str(list(keys))[1:-1]}"
            elif attr in ["pos_annot"]:
                if getattr(self, attr) is not None:
                    descr += f"\n{attr}: PyRanges object with {len(getattr(self, attr))} features"
                else:
                    descr += f"\n{attr}: None"
            elif attr in ["seqsm", "uns"]:
                if len(getattr(self, attr)) > 0:
                    descr += f"\n{attr}: {str(list(getattr(self, attr).keys()))[1:-1]}"
                else:
                    descr += f"\n{attr}: None"
        return descr

    def copy(self):
        """Return a copy of the SeqData object."""
        return deepcopy(self)

    def write_h5sd(self, path: PathLike, mode: str = "w"):
        """Write SeqData object to h5sd file.

        Parameters
        ----------
        path: PathLike
            Path to h5sd file. If file exists, it will be overwritten.
        mode: str, optional
            Mode to open h5sd file. Default is "w".
        """
        from .._io import write_h5sd

        write_h5sd(self, path, mode)

    def make_names_unique(self):
        """Make sequence names unique by appending a number to the end of each name."""
        n_digits = len(str(self.n_obs))
        new_index = np.array(["seq{num:0{width}}".format(num=i, width=n_digits)for i in range(self.n_obs)])
        self.names = new_index
        self.seqs_annot["index"] = self.seqs_annot.index
        self.seqs_annot.index = new_index
