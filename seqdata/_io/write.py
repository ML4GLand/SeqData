import h5py
import numpy as np
import pandas as pd
from typing import Optional
from os import PathLike

def write_csv(
    sdata, 
    filename,
    sep: str = "\t",
    compression: str = "infer",
    **kwargs,
):
    """Write sequences from SeqData to csv files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    sep : str, optional
        Delimiter to use. Defaults to "\\t".
    compression : str, optional
        Compression type to use. Defaults to "infer".
    **kwargs : kwargs, dict
        Keyword arguments to pass to pandas.read_csv. Defaults to {}.
    """
    if sdata.seqs_annot is not None:
        if sdata.seqs is not None:
            sdata.seqs_annot["seq"] = sdata.seqs
        sdata.seqs_annot.to_csv(filename, sep=sep, compression=compression, **kwargs)
    else:
        raise ValueError("SeqData object does not contain sequence annotations.")

def write_fasta(
    sdata, 
    filename
):
    """Write sequences from SeqData to fasta files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    """
    with open(filename, "w") as f:
        for i in range(len(sdata.seqs)):
            f.write(">" + sdata.names[i] + "\n")
            f.write(sdata.seqs[i] + "\n")

def write_bed(
    sdata, 
    filename,
    chrom_col: str = "chr",
    start_col: str = "start",
    end_col: str = "end",
):
    """Write sequences from SeqData to bed files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    """
    bed_cols = [chrom_col, start_col, end_col]
    if sdata.seqs_annot is not None:
        non_bed_cols = [col for col in sdata.seqs_annot.columns if col not in bed_cols]
    else:
        raise ValueError("SeqData object does not contain sequence annotations.")
    if all([col in sdata.seqs_annot.columns for col in bed_cols]):
        sdata.seqs_annot[bed_cols + non_bed_cols].to_csv(filename, sep="\t", index=False. header=False)
    else:
        raise ValueError("SeqData object does not contain chr, start, and end columns specified.")

def write_numpy(
    sdata, 
    filename
):
    """Write sequences from SeqData to numpy files.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    ohe : bool
        Whether to include ohe sequences in a separate file.
    target_key : str, optional
        Optionally save targets from a SeqData object using a key.
    """
    if sdata.seqs is not None:
        np.save(filename + "_seqs.npy", sdata.seqs)
    if sdata.names is not None:
        np.save(filename + "_names.npy", sdata.names)
    if sdata.rev_seqs is not None:
        np.save(filename + "_rev_seqs.npy", sdata.rev_seqs)
    if sdata.ohe_rev_seqs is not None:
        np.save(filename + "_ohe_rev_seqs.npy", sdata.ohe_rev_seqs)
    if sdata.seqs_annot is not None:
        pd.to_csv(filename + "_seqs_annot.csv", sdata.seqs_annot)

def write_h5sd(
    sdata, 
    filename: Optional[PathLike] = None, 
    mode: str = "w"
):
    """Write SeqData object to h5sd file.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str, optional
        File path to write to. Defaults to None.
    mode : str, optional
        Mode to open file. Defaults to "w".
    """
    with h5py.File(filename, mode) as f:
        f = f["/"]
        f.attrs.setdefault("encoding-type", "SeqData")
        f.attrs.setdefault("encoding-version", "0.0.1")
        if sdata.seqs is not None:
            f.create_dataset("seqs", data=np.array([n.encode("ascii", "ignore") for n in sdata.seqs]))
        if sdata.names is not None:
            f.create_dataset("names", data=np.array([n.encode("ascii", "ignore") for n in sdata.names]))
        if sdata.ohe_seqs is not None:
            f.create_dataset("ohe_seqs", data=sdata.ohe_seqs)
        if sdata.rev_seqs is not None:
            f.create_dataset("rev_seqs", data=np.array([n.encode("ascii", "ignore") for n in sdata.rev_seqs]))
        if sdata.ohe_rev_seqs is not None:
            f.create_dataset("ohe_rev_seqs", data=sdata.ohe_rev_seqs)
        if sdata.seqs_annot is not None:
            for key, item in dict(sdata.seqs_annot).items():
                if item.dtype == "object":
                    f["seqs_annot/" + str(key)] = np.array([n.encode("ascii", "ignore") for n in item.replace(np.nan, "NA")])
                else:
                    f["seqs_annot/" + str(key)] = item
        if sdata.pos_annot is not None:
            for key, item in dict(sdata.pos_annot.df).items():
                if item.dtype in ["object", "category"]:
                    f["pos_annot/" + str(key)] = np.array([n.encode("ascii", "ignore")for n in item.replace(np.nan, "NA")])
                else:
                    f["pos_annot/" + str(key)] = item
        if sdata.seqsm is not None:
            for key, item in dict(sdata.seqsm).items():
                f["seqsm/" + str(key)] = item
        if sdata.uns is not None:
            for key, item in dict(sdata.uns).items():
                if "pfms" in key:
                    pfms = np.zeros((len(item), *item[list(item.keys())[0]].shape))
                    for i, in_key in enumerate(item.keys()):
                        pfms[i, :, :] = item[in_key]
                    item = pfms
                try:
                    f["uns/" + str(key)] = item
                except TypeError:
                    print(f"Unsupported type for {key}")
                    continue

def write(sdata, filename, *args, **kwargs):
    """Wrapper function to write SeqData objects to various file types.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences and identifiers.
    filename : str
        File path to write to.
    *args : args, dict
        Positional arguments from write_csv, write_fasta, write_numpy.
    **kwargs : kwargs, dict
        Keyword arguments from write_csv, write_fasta, write_numpy.
    """
    seq_file_extension = filename.split(".")[-1]
    if seq_file_extension in ["csv", "tsv"]:
        write_csv(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["fasta", "fa"]:
        write_fasta(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["bed"]:
        write_bed(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["npy"]:
        write_numpy(sdata, filename, *args, **kwargs)
    elif seq_file_extension in ["h5sd", "h5"]:
        write_h5sd(sdata, filename, *args, **kwargs)
    else:
        print("Sequence file type not currently supported.")
        return