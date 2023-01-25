import h5py
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Iterable
from os import PathLike
import pyranges as pr
from .. import SeqData
from .utils import _read_and_concat_dataframes


def read_csv(
    filename: Union[PathLike, List[PathLike]],
    seq_col: Optional[str] = None,
    name_col: Optional[str] = None,
    sep: str = "\t",
    low_memory: bool = False,
    col_names: Iterable = None,
    compression: str = "infer",
    **kwargs,
):
    """Read sequences and sequence metadata into SeqData object from csv/tsv files.

    Parameters
    ----------
    file : PathLike
        File path(s) to read the data from. If a list is provided, the files will be concatenated (must have the same columns).
    seq_col : str, optional
        Column name containing sequences. Defaults to None.
    name_col : str, optional
        Column name containing identifiers. Defaults to None.
    sep : str, optional
        Delimiter to use. Defaults to "\\t".
    low_memory : bool, optional
        Whether to use low memory mode. Defaults to False.
    return_numpy : bool, optional
        Whether to return numpy arrays. Defaults to False.
    return_dataframe : bool, optional
        Whether to return pandas dataframe. Defaults to False.
    col_names : Iterable, optional
        Column names to use. Defaults to None. If not provided, uses first line of file.
    auto_name : bool, optional
        Whether to automatically generate identifiers. Defaults to True.
    compression : str, optional
        Compression type to use. Defaults to "infer".
    **kwargs : kwargs, dict
        Keyword arguments to pass to pandas.read_csv. Defaults to {}.

    Returns
    -------
    sdata : SeqData
        Returns SeqData object containing sequences and identifiers by default
    tuple :
        Returns numpy arrays of identifiers, sequences, reverse complement sequences and annots.
        If return_numpy is True. If any are not provided they are set to none.
    dataframe : pandas.DataFrame
        Returns pandas dataframe containing sequences and identifiers if return_dataframe is True.
    """
    dataframe = _read_and_concat_dataframes(
        file_names=filename,
        col_names=col_names,
        sep=sep,
        low_memory=low_memory,
        compression=compression,
        **kwargs,
    )
    if seq_col is not None and seq_col in dataframe.columns:
        seqs = dataframe[seq_col].to_numpy(dtype=str)
        dataframe.drop(seq_col, axis=1, inplace=True)
    else:
        seqs = None
    if name_col is not None and name_col in dataframe.columns:
        names = dataframe[name_col].to_numpy(dtype=str)
        dataframe.set_index(name_col, inplace=True)
    else:
        n_digits = len(str(len(dataframe) - 1))
        dataframe.index = np.array(["seq{num:0{width}}".format(num=i, width=n_digits) for i in range(len(dataframe))])
        names = dataframe.index.to_numpy()
    return SeqData(
        names=names,
        seqs=seqs,
        seqs_annot=dataframe
    )

def read_fasta(
    seq_file, 
    annot_file=None,
    sep="\t",
):
    """Read sequences into SeqData object from fasta files.

    Parameters
    ----------
    seq_file : str
        Fasta file path to read
    annot_file : str
        Delimited file path containing annotation file. Defaults to None.

    Returns
    -------
    sdata : SeqData
        Returns SeqData object containing sequences and identifiers
    """
    seqs = np.array([x.rstrip() for (i, x) in enumerate(open(seq_file)) if i % 2 == 1])
    names = np.array([x.rstrip().replace(">", "") for (i, x) in enumerate(open(seq_file)) if i % 2 == 0])
    seqs_annot = pd.read_csv(annot_file, sep=sep) if annot_file is not None else None
    return SeqData(
        names=names,
        seqs=seqs,
        seqs_annot=seqs_annot if seqs_annot is not None else None,
    )

def read_bed(
    bed_file: str,
    ref_file: str = None,
    **kwargs,
):
    """Modify to just load in bed region info into seqs_annot

    Parameters
    ----------
    bed_file : str
        Path to the BED file where peaks are stored.
    ref_file : str, optional
        Path to the reference genome file. If provided, the sequences are extracted with pyRanges. Defaults to None.
    **kwargs : dict
        Additional arguments to pass to as Janggu's parameters for loading.

    Returns
    -------
    sdata : SeqData
        SeqData object containing the peaks.
    """
    pass

def read_h5sd(
    filename: PathLike,
):
    """
    Read sequences into SeqData objects from h5sd files.

    Parameters
    ----------
    filename : PathLike
        File path to read the data from.

    Returns
    -------
        sdata: SeqData object.
    """
    with h5py.File(filename, "r") as f:
        d = {}
        
        # read in seqs
        if "seqs" in f:
            d["seqs"] = np.array([n.decode("ascii", "ignore") for n in f["seqs"][:]])
        
        # read in names
        if "names" in f:
            d["names"] = np.array([n.decode("ascii", "ignore") for n in f["names"][:]])
        
        # read in ohe_seqs
        if "ohe_seqs" in f:
            d["ohe_seqs"] = f["ohe_seqs"][:]
        
        # read in seqs_annot
        if "seqs_annot" in f:
            out_dict = {}
            for key in f["seqs_annot"].keys():
                out = f["seqs_annot"][key][()]
                if isinstance(out[0], bytes):
                    out_dict[key] = np.array([n.decode("ascii", "ignore") for n in out])
                else:
                    out_dict[key] = out
            if "names" in f:
                idx = d["names"]
            else:
                n_digits = len(str(len(d["seqs"])))
                idx = np.array(["seq{num:0{width}}".format(num=i, width=n_digits)for i in range(len(d["seqs"]))])
            d["seqs_annot"] = pd.DataFrame(index=idx, data=out_dict).replace("NA", np.nan)
        
        # read in pos_annot
        if "pos_annot" in f:
            out_dict = {}
            for key in f["pos_annot"].keys():
                out = f["pos_annot"][key][()]
                if isinstance(out[0], bytes):
                    out_dict[key] = np.array([n.decode("ascii", "ignore") for n in out])
                else:
                    out_dict[key] = out
            d["pos_annot"] = pr.from_dict(out_dict)
        
        # read in seqsm
        if "seqsm" in f:
            out_dict = {}
            for key in f["seqsm"].keys():
                out = f["seqsm"][key][()]
                if isinstance(out[0], bytes):
                    out_dict[key] = np.array([n.decode("ascii", "ignore") for n in out])
                else:
                    out_dict[key] = out
            d["seqsm"] = out_dict
        
        # read in uns
        if "uns" in f:
            out_dict = {}
            for key in f["uns"].keys():
                if key == "pfms":
                    pfm_dfs = {}
                    for i, pfm in enumerate(f["uns"][key][()]):
                        pfm_dfs[i] = pd.DataFrame(pfm, columns=["A", "C", "G", "T"])
                    out_dict[key] = pfm_dfs
                else:
                    out = f["uns"][key][()]
                    if isinstance(out[0], bytes):
                        out_dict[key] = np.array(
                            [n.decode("ascii", "ignore") for n in out]
                        )
                    else:
                        out_dict[key] = out
            d["uns"] = out_dict
    return SeqData(**d)

def read(
    seq_file, *args, **kwargs):
    """Wrapper function to read sequences based on file extension.

    Parameters
    ----------
    seq_file : str
        File path containing sequences.
    *args : dict
        Positional arguments from read_csv, read_fasta, read_numpy, etc.
    **kwargs : dict
        Keyword arguments from read_csv, read_fasta, read_numpy, etc.

    Returns
    -------
    sdata : SeqData
        SeqData object containing sequences and identifiers
    tuple :
        Numpy arrays of identifiers, sequences, reverse complement sequences and annots.
        If any are not provided they are set to none.
    """
    seq_file_extension = seq_file.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        return read_csv(seq_file, *args, **kwargs)
    elif seq_file_extension in ["fasta", "fa"]:
        return read_fasta(seq_file, *args, **kwargs)
    elif seq_file_extension in ["bed"]:
        return read_bed(seq_file, *args, **kwargs)
    elif seq_file_extension in ["h5sd", "h5"]:
        return read_h5sd(seq_file, *args, **kwargs)
    else:
        raise ValueError("File extension not recognized. \
            Please provide a file with one of the following extensions: \
            csv, fasta, bed or h5sd"
        )