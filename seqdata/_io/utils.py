from pathlib import Path
from typing import Optional, Union, List
import pandas as pd

PathType = Union[str, Path]


def _read_and_concat_dataframes(
    file_names: Union[PathType, List[PathType]],
    col_names: Optional[Union[str, List[str]]] = None,
    sep: str = "\t",
    low_memory: bool = False,
    compression: str = "infer",
    **kwargs
) -> pd.DataFrame:
    """Reads a list of files and concatenates them into a single dataframe.

    Parameters
    ----------
    file_names : str or list
        Path to file or list of paths to files.
    col_names : str or list, optional
        Column names to use for the dataframe. If not provided, the column names will be the first row of the file.
    sep : str, optional
        Separator to use for the dataframe. Defaults to "\t".
    low_memory : bool, optional
        If True, the dataframe will be stored in memory. If False, the dataframe will be stored on disk. Defaults to False.
    compression : str, optional
        Compression to use for the dataframe. Defaults to "infer".
    **kwargs
        Additional arguments to pass to pd.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(file_names, list):
        file_names = [file_names]
    if not isinstance(col_names, list) and col_names is not None:
        col_names = [col_names]
    dfs = []
    for file_name in file_names:
        x = pd.read_csv(
            file_name,
            sep=sep,
            low_memory=low_memory,
            names=col_names,
            compression=compression,  # type: ignore
            header=0,
            **kwargs
        )
        dfs.append(x)
    dataframe = pd.concat(dfs, ignore_index=True)
    return dataframe