import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
from typing import Optional

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import utils

RandomState = Optional[int | np.random.Generator]


def prepare_ts_dataset(df: pd.DataFrame, behaviour_ts: str, granularity: int, algorithm: str = None, scale: str = '') -> pd.DataFrame:
    """
    Filter time series data from dataframe and convert to format to be used in TS analysis or ML approaches.

    :param df: Dataframe with all data to
    :param behaviour_ts: commonly 'DistanceToOptimum' or 'BestObjectiveValue', or specific other behaviour data such as diversity
    :param algorithm: algorithm for which ts dataset is created; if None, only config is used
    :param granularity: number of steps to use, e.g. 10 takes only every 10th TS entry (but always the first and last)
    :param scale: Type of scikit scaler to use; default: no scaler

    :return: Dataframe with columns for 'Config' and time steps with corresponding values of interest.
    """
    labels = ['Config']
    if algorithm:
        alg_labels = []
        if algorithm == 'GA':
            alg_labels = utils.GA_labels
        elif algorithm == 'PSO':
            alg_labels = utils.PSO_labels
        elif algorithm == 'DE':
            alg_labels = utils.DE_labels
        elif algorithm == 'ES':
            alg_labels = utils.ES_labels
        labels = ['Config'] + utils.config_labels + alg_labels

    ts_data = df[[behaviour_ts]].apply(lambda x: x[behaviour_ts], axis=1, result_type='expand')
    cols = ts_data.columns.tolist()
    cols_to_keep = [col for i, col in enumerate(cols[1:-1], start=1) if i % granularity == 0]
    cols_to_keep.insert(0, cols[0])
    cols_to_keep.insert(len(cols_to_keep), cols[-1])
    ts_data = ts_data[cols_to_keep]
    ts_data = pd.concat([df[labels], ts_data], axis=1)

    if scale == 'minmax':
        scaler = MinMaxScaler()
        ts_data[behaviour_ts] = scaler.fit_transform(ts_data[behaviour_ts])
    elif scale == 'standardize':
        scaler = StandardScaler()
        ts_data[behaviour_ts] = scaler.fit_transform(ts_data[behaviour_ts])
    else:
        return ts_data

    return ts_data


def prepare_ts_dataset_means(df: pd.DataFrame, behaviour_ts: str, granularity: int, algorithm: str = None, scale: str = '') -> pd.DataFrame:
    """
    Filter time series data from dataframe and convert to format to be used in TS analysis or ML approaches.
    Calculates the mean of subsequent steps, with the number of steps defined by 'granularity'.

    :param df: Dataframe with all data to
    :param behaviour_ts: commonly 'DistanceToOptimum' or 'BestObjectiveValue', or specific other behaviour data such as diversity
    :param algorithm: algorithm for which ts dataset is created; if None, only config is used
    :param granularity: number of steps to use, e.g. 10 takes every 10 TS entries (but first and last are always kept)
    :param scale: Type of scikit scaler to use; default: no scaler

    :return: Dataframe with columns for 'Config' and time steps with corresponding values of interest.
    """
    labels = ['Config']
    if algorithm:
        alg_labels = []
        if algorithm == 'GA':
            alg_labels = utils.GA_labels
        elif algorithm == 'PSO':
            alg_labels = utils.PSO_labels
        elif algorithm == 'DE':
            alg_labels = utils.DE_labels
        elif algorithm == 'ES':
            alg_labels = utils.ES_labels
        labels = ['Config'] + utils.config_labels + alg_labels

    ts_data = df[[behaviour_ts]].apply(lambda x: x[behaviour_ts], axis=1, result_type='expand')
    transposed = ts_data.T
    transposed = transposed.rolling(granularity, min_periods=1, step=granularity).mean()
    ts_data = transposed.T
    ts_data = pd.concat([df[labels], ts_data], axis=1)

    if scale == 'minmax':
        scaler = MinMaxScaler()
        ts_data[behaviour_ts] = scaler.fit_transform(ts_data[behaviour_ts])
    elif scale == 'standardize':
        scaler = StandardScaler()
        ts_data[behaviour_ts] = scaler.fit_transform(ts_data[behaviour_ts])
    else:
        return ts_data

    return ts_data


def prepare_ts_dataset_aocc(df: pd.DataFrame, behaviour_ts: str, granularity: int, algorithm: str = None, scale: str = '') -> pd.DataFrame:
    """
    Filter time series data from dataframe and convert to format to be used in TS analysis or ML approaches.
    Calculates the AOCC value of subsequent steps, with the number of steps defined by 'granularity'.

    :param df: Dataframe with all data to
    :param behaviour_ts: commonly 'DistanceToOptimum' or 'BestObjectiveValue', or specific other behaviour data such as diversity
    :param algorithm: algorithm for which ts dataset is created; if None, only config is used
    :param granularity: number of steps to use, e.g. 10 takes every 10 TS entries (but first and last are always kept)
    :param scale: Type of scikit scaler to use; default: no scaler


    :return: Dataframe with columns for 'Config' and time steps with corresponding values of interest.
    """
    labels = ['Config']
    if algorithm:
        alg_labels = []
        if algorithm == 'GA':
            alg_labels = utils.GA_labels
        elif algorithm == 'PSO':
            alg_labels = utils.PSO_labels
        elif algorithm == 'DE':
            alg_labels = utils.DE_labels
        elif algorithm == 'ES':
            alg_labels = utils.ES_labels
        labels = ['Config'] + utils.config_labels + alg_labels

    ts_data = df[[behaviour_ts]].apply(lambda x: x[behaviour_ts], axis=1, result_type='expand')
    transposed = ts_data.T
    aocc_values = lambda x: utils.aocc(x)
    transposed = transposed.rolling(granularity, min_periods=1, step=granularity).apply(aocc_values)
    ts_data = transposed.T
    ts_data = pd.concat([df[labels], ts_data], axis=1)

    if scale == 'minmax':
        scaler = MinMaxScaler()
        ts_data[behaviour_ts] = scaler.fit_transform(ts_data[behaviour_ts])
    elif scale == 'standardize':
        scaler = StandardScaler()
        ts_data[behaviour_ts] = scaler.fit_transform(ts_data[behaviour_ts])
    else:
        return ts_data
    
    return ts_data


def load_time_series(name: str = None, drop_labels: list = 'Config', path: str | Path = None, n: Optional[int] = None, frac: Optional[float] = None,
                     random_state: RandomState = None, return_df: bool = False) -> np.ndarray | pd.DataFrame:
    """
    Read time series dataset from .feather and convert to dataframe or numpy array.
    :param name: Name of dataset to read.
    :param drop_labels: List of labels to drop from ts dataframe; drops 'Config', but should drop all additional columns
    :param path: Optional. Path to dataset, only if deviating from standard path.
    :param n: Optional. Number of random examples from dataset to include.
    :param frac: Optional. Fraction of random examples from dataset to include.
    :param random_state: Optional. Random state for sampling random examples from dataset.
    :param return_df: Optional. If True, returns dataframe instead of numpy array.
    :return: Dataset as numpy array or pandas dataframe.
    """
    if path is None:
        base_path = Path(__file__).parent
        path = (base_path / f"../data/exploratory/dataframes/time_series/{name}.feather").resolve()
    else:
        path = (path / f'{name}.feather').resolve()
    df = pd.read_feather(path)
    df.drop(drop_labels, axis=1, inplace=True)
    if n is not None or frac is not None:
        df = df.sample(n=n, frac=frac, random_state=random_state)

    if return_df:
        return df

    data = df.to_numpy(dtype=float)

    return data
