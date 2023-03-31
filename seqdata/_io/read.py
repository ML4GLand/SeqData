from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import zarr
from numcodecs import Delta, blosc
from tqdm import tqdm

from .. import SeqData
from .utils import _read_and_concat_dataframes, _read_bedlike, _set_uniform_length_around_center, _df_to_xr_zarr

PathType = Union[str, Path]


def read_csvs(
    filename: Union[PathType, List[PathType]],
    seq_col: Optional[str] = None,
    name_col: Optional[str] = None,
    sep: str = "\t",
    low_memory: bool = False,
    col_names: Optional[Union[str, List[str]]] = None,
    compression: str = "infer",
    **kwargs,
):
    """Read sequences and sequence metadata into SeqData object from csv/tsv files.

    Parameters
    ----------
    file : str, Path
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
        seqs = dataframe[seq_col].to_numpy(dtype="U").reshape(-1, 1).view("U1")
        dataframe.drop(seq_col, axis=1, inplace=True)
    else:
        seqs = None

    ds = dataframe.rename_axis("sequence").to_xarray()
    del ds.coords["sequence"]
    if seqs is not None:
        ds["seqs"] = (("sequence", "length"), seqs)
    if name_col is None:
        n_digits = len(str(len(dataframe) - 1))
        names = np.array([f"seq{i:0{n_digits}}" for i in range(len(dataframe))])
        ds["names"] = ("sequence", names)

    return SeqData(ds)


def read_fasta(
    seq_name: str,
    id_name: str,
    fasta_path: PathType,
    seqdata_path: PathType,
    batch_size: int,
):
    """Write a SeqData from a FASTA file where each contig is a sequence.

    Parameters
    ----------
    seq_name : str
        Name of array for FASTA sequences
    id_name : str
        Name of array for FASTA sequence IDs
    fasta_path : str, Path
    seqdata_path : str, Path
    batch_size : int
        Make this as large as RAM allows. How many sequences to IO at a time.
    """
    z = zarr.open_group(seqdata_path)
    with pysam.FastaFile(str(fasta_path)) as f:
        seq_names = f.references

        arr = z.array(id_name, data=np.array(list(seq_names), object), overwrite=True)
        arr.attrs["_ARRAY_DIMENSIONS"] = ["sequence"]

        n_seqs = len(seq_names)
        length = f.get_reference_length(seq_names[0])
        batch_size = min(n_seqs, batch_size)

        seqs = z.empty(
            seq_name,
            shape=(n_seqs, length),
            dtype="|S1",
            chunks=(int(1e3), None),
            overwrite=True,
            compressor=blosc.Blosc("zstd", clevel=7, shuffle=-1),
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length"]
        batch = np.empty((batch_size, length), dtype="|S1")
        batch_start_idx = 0
        batch_idx = 0
        for contig in seq_names:
            batch[batch_idx] = np.frombuffer(f.fetch(contig).encode("ascii"), "|S1")
            if batch_idx == batch_size - 1:
                seqs[batch_start_idx : batch_start_idx + batch_size] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        seqs[batch_start_idx : batch_start_idx + batch_idx] = batch[:batch_idx]
    zarr.consolidate_metadata(seqdata_path)  # type: ignore


def read_bigwig(
    out_arr: zarr.Array,
    bigwig_path: PathType,
    bed: pd.DataFrame,
    batch_size: int,
    sample_idx: int
):
    length = bed.at[0, 'chromEnd'] - bed.at[0, 'chromStart']
    batch = np.zeros((batch_size, length), np.uint16)
    batch_start_idx = 0
    batch_idx = 0
    with pyBigWig.open(bigwig_path) as f:
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
            contig, start, end = row[:3]
            intervals = f.intervals(contig, start, end)
            if intervals is not None:
                for interval in intervals:
                    rel_start = interval[0] - start
                    rel_end = interval[1] - start
                    value = interval[2]
                    batch[batch_idx, rel_start:rel_end] = value
            if batch_idx == batch_size - 1:
                out_arr[batch_start_idx : batch_start_idx + batch_size, :, sample_idx] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        out_arr[batch_start_idx : batch_start_idx + batch_idx, :, sample_idx] = batch[:batch_idx]


def read_bigwigs(
    name: str,
    bigwig_paths: List[PathType],
    sample_names: List[str],
    bed_path: PathType,
    seqdata_path: PathType,
    length: int,
    batch_size: int,
    n_jobs: int = 1,
    threads_per_job: int = 1,
):
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)
    
    bed = _read_bedlike(bed_path)
    _set_uniform_length_around_center(bed, length)
    _df_to_xr_zarr(bed, seqdata_path, ["sequence"])

    batch_size = min(len(bed), batch_size)
    z = zarr.open_group(seqdata_path)
    
    arr = z.array('samples', data=np.array(sample_names, object), chunks=100, compressor=compressor, overwrite=True)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["sample"]
    
    coverage = z.zeros(
        name,
        shape=(len(bed), length),
        dtype=np.uint16,
        chunks=(batch_size, None),
        overwrite=True,
        compressor=compressor,
        filters=[Delta(np.uint16)],
    )
    coverage.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length"]
    
    sample_idxs = np.arange(len(sample_names))
    tasks = [joblib.delayed(read_bigwig(coverage, bigwig, bed, batch_size, sample_idx) for bigwig, sample_idx in zip(bigwig_paths, sample_idxs))]
    with joblib.parallel_backend('loky', n_jobs=n_jobs, inner_max_num_threads=threads_per_job):
        joblib.Parallel()(tasks)

    zarr.consolidate_metadata(seqdata_path)  # type: ignore


def read_bam(
    out_arr: zarr.Array,
    bam_path: PathType,
    bed: pd.DataFrame,
    batch_size: int,
    sample_idx: int
):
    length = bed.at[0, 'chromEnd'] - bed.at[0, 'chromStart']
    batch = np.zeros((batch_size, length), np.uint16)
    batch_start_idx = 0
    batch_idx = 0
    with pysam.AlignmentFile(str(bam_path)) as f:
        for i, row in tqdm(bed.iterrows(), total=len(bed)):
            contig, start, end = row[:3]
            a, c, g, t = f.count_coverage(contig, start, end, read_callback="all")
            batch[batch_idx] = np.vstack([a, c, g, t]).sum(0).astype(np.uint16)
            if batch_idx == batch_size - 1:
                out_arr[batch_start_idx : batch_start_idx + batch_size, :, sample_idx] = batch
                batch_idx = 0
                batch_start_idx += batch_size
            else:
                batch_idx += 1
    if batch_idx != batch_size:
        out_arr[batch_start_idx : batch_start_idx + batch_idx, :, sample_idx] = batch[:batch_idx]


def read_bams(
    name: str,
    bam_paths: List[PathType],
    sample_names: List[str],
    bed_path: PathType,
    seqdata_path: PathType,
    length: int,
    batch_size: int,
    n_jobs: int = 1,
    threads_per_job: int = 1,
):
    compressor = blosc.Blosc("zstd", clevel=7, shuffle=-1)
    
    bed = _read_bedlike(bed_path)
    _set_uniform_length_around_center(bed, length)
    _df_to_xr_zarr(bed, seqdata_path, ["sequence"], compressor=compressor)
    
    batch_size = min(len(bed), batch_size)
    z = zarr.open_group(seqdata_path)
    
    arr = z.array('samples', data=np.array(sample_names, object), chunks=100, compressor=compressor, overwrite=True)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["sample"]
    
    coverage = z.zeros(
        name,
        shape=(len(bed), length),
        dtype=np.uint16,
        chunks=(batch_size, None),
        overwrite=True,
        compressor=compressor,
        filters=[Delta(np.uint16)],
    )
    coverage.attrs["_ARRAY_DIMENSIONS"] = ["sequence", "length", "sample"]
    
    sample_idxs = np.arange(len(sample_names))
    tasks = [joblib.delayed(read_bam(coverage, bam, bed, batch_size, sample_idx) for bam, sample_idx in zip(bam_paths, sample_idxs))]
    with joblib.parallel_backend('loky', n_jobs=n_jobs, inner_max_num_threads=threads_per_job):
        joblib.Parallel()(tasks)
    
    zarr.consolidate_metadata(seqdata_path)  # type: ignore
