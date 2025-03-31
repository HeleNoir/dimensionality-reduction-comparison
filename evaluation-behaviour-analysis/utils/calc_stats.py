import pandas as pd
import numpy as np

__all__ = ["basic_statistics", "calculate_statistics", "calculate_list_statistics", "aocc"]


def basic_statistics(df, step_columns, value_column):
    """
    Calculate basic statistics for value of interest (e.g. best found solution, diversity etc.) depending on
    steps_column (usually either iterations or evaluations)
    """
    stats_df = df.groupby(step_columns).agg(
        Algorithm=pd.NamedAgg('Algorithm', 'first'),
        mean_opt=pd.NamedAgg(column=value_column, aggfunc=np.mean),
        std_opt=pd.NamedAgg(column=value_column, aggfunc=np.std),
        min_opt=pd.NamedAgg(column=value_column, aggfunc="min"),
        max_opt=pd.NamedAgg(column=value_column, aggfunc="max"),
        median_opt=pd.NamedAgg(column=value_column, aggfunc=np.median),
    )
    return stats_df


def calculate_statistics(group):
    """ (Note: This function was created with the help of ChatGPT-4o.) """
    return pd.Series({
        'mean': group.mean(),
        'std': group.std(),
        'median': group.median(),
        'min': group.min(),
        'max': group.max()
    })


def calculate_list_statistics(arrays):
    """ (Note: This function was created with the help of ChatGPT-4o.) """
    arrays = np.array(arrays)
    return {
        'mean': arrays.mean(axis=0).tolist(),
        'std': arrays.std(axis=0).tolist(),
        'median': np.median(arrays, axis=0).tolist(),
        'min': arrays.min(axis=0).tolist(),
        'max': arrays.max(axis=0).tolist()
    }


def f(x, lb, ub) -> float:
    """
    Helper for computing the inner sum of the AOCC.
    :param x: current precision
    :param lb: lower bound
    :param ub: upper bound
    :return: inner sum of AOCC as float
    """
    return 1.0 - (min([max([x, lb]), ub]) - lb) / (ub - lb)


def aocc(df, lb: float = 0.00000001, ub: float = 100.0) -> float:
    """
    Computes the AOCC for the column specified in 'value_column'.
    Adapted from: Vermetten et al. 2024, 'Large-scale Benchmarking of Metaphor-based Optimization Heuristics'

    :param df: Dataframe with performance data.
    :param lb: Lower bound of precision values.
    :param ub: Upper bound of precision values.
    :return: AOCC value.
    """
    result = [f(np.log10(x), np.log10(lb), np.log10(ub)) for x in df]
    aocc_sum = sum(result)
    aocc_value = aocc_sum / len(df.index)

    return aocc_value
