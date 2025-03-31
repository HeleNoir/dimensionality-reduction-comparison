from pathlib import Path
from typing import Union
from joblib import Parallel, delayed

import cbor2 as cb
import pandas as pd


def read_log(file_path: Union[Path, str]) -> pd.DataFrame:
    """Read the log file in `file_path` and convert it to dataframe."""
    file_path = Path(file_path)
    with file_path.open('rb') as fp:
        obj = cb.load(fp)
    names = obj['names']
    entries = [{names[key]: value for key, value in entry.items()} for entry in obj['entries']]
    frame = pd.DataFrame.from_records(data=entries, columns=names)
    return frame


def read_log_and_name(file_path: Union[Path, str]) -> (pd.DataFrame, str):
    """Read the log file in `file_path` and convert it to dataframe, returning also the file name."""
    file_path = Path(file_path)
    with file_path.open('rb') as fp:
        obj = cb.load(fp)
    names = obj['names']
    entries = [{names[key]: value for key, value in entry.items()} for entry in obj['entries']]
    frame = pd.DataFrame.from_records(data=entries, columns=names)
    file_name = file_path.stem
    return frame, file_name


def read_log_dir(dir_path: Union[Path, str]):
    """Read all log files in `dir_path`, convert them to dataframes and return them as dict."""
    dir_path = Path(dir_path)
    df_dict = {}
    for file in dir_path.glob("*.cbor"):
        df = read_log(dir_path / file)
        df_dict[file.stem] = df
    return df_dict


def read_log_dir_with_joblib(dir_path: Union[Path, str], n_jobs=-1):
    """
    Read all log files in `dir_path` in parallel using joblib, convert them to dataframes and return them as dict.
    """
    dir_path = Path(dir_path)
    dataframes = Parallel(n_jobs=n_jobs)(
        delayed(read_log_and_name)(dir_path / file) for file in dir_path.glob("*.cbor")
    )
    dfs, names = zip(*dataframes)
    df_dict = dict(map(lambda i, j: (i, j), names, dfs))
    return df_dict


def read_partial_logs_with_joblib(dir_path: Union[Path, str], n_jobs=-1):
    """
    Read log files in `dir_path` starting with "1" or "2" in parallel using joblib,
    convert them to dataframes, and return them as a dictionary.
    """
    dir_path = Path(dir_path)
    # Filter files starting with "1" or "2"
    dataframes = Parallel(n_jobs=n_jobs)(
        delayed(read_log_and_name)(file)
        for file in dir_path.glob("*.cbor")
        if file.name.startswith(("1", "2"))
    )
    dfs, names = zip(*dataframes)
    df_dict = dict(zip(names, dfs))
    return df_dict
