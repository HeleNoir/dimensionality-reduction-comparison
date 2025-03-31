import os
from pathlib import Path

import pandas as pd
import numpy as np

import utils


def summarise_final_stats(df, config_columns, value_columns, additional_columns=None, dataset: str = '',
                          output_name='', save_directory=''):
    """
    Summarises single-value data columns grouped by selected configuration columns.
    (Note: This function was created with the help of ChatGPT-4o.)

    Parameters:
        df (pd.DataFrame): The input dataframe.
        config_columns (list of str): Columns to group by for summarising.
        value_columns (list of str): Columns containing single values to summarise.
        additional_columns (list of str, optional): Additional columns to retain if values are consistent.
        dataset (str): Name of the used dataset.
        output_name (str): Name of the output file.
        save_directory: Path to save output.

    Returns:
        Returns pd.DataFrame: A DataFrame with summary statistics for the specified data columns.
        If output_name is given, saves the data as csv and feather.
    """

    grouped = df.groupby(config_columns, observed=False)
    results = []

    for group_name, group in grouped:
        row = {col: group_name[i] if isinstance(group_name, tuple) else group_name
               for i, col in enumerate(config_columns)}

        # Handle additional columns
        if additional_columns:
            for col in additional_columns:
                unique_values = group[col].unique()
                if len(unique_values) > 1:
                    raise ValueError(
                        f"Column '{col}' has inconsistent values within group {group_name}."
                    )
                row[col] = unique_values[0]

        # Add final number of iterations and evaluations
        for list_col in ['Iterations', 'Evaluations']:
            row[f"{list_col}_last"] = group[list_col].iloc[0][-1]

        # Process data columns
        for data_col in value_columns:
            stats = utils.calculate_statistics(group[data_col])
            for stat_name, stat_value in stats.items():
                row[f"{data_col}_{stat_name}"] = stat_value
        results.append(row)

    if output_name != '':
        directory_results = save_directory / f'descriptive/{dataset}/'
        directory = save_directory / 'dataframes/'
        Path.mkdir(directory_results, parents=True, exist_ok=True)
        Path.mkdir(directory, parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(directory_results / f'{dataset}_{output_name}_summaries.csv')
        pd.DataFrame(results).to_feather(directory / f'{dataset}_{output_name}_summaries.feather')

    return pd.DataFrame(results)


def summarise_process_stats(df, config_columns, value_columns, step_column=None, additional_columns=None,
                            dataset: str = '', output_name='', save_directory=''):
    """
    Summarises data columns containing lists, grouped by selected configuration columns.
    (Note: This function was created with the help of ChatGPT-4o.)

    Parameters:
        :param df: (pd.DataFrame) The input dataframe.
        :param config_columns: (list of str) Columns to group by for summarising.
        :param value_columns: (list of str) Columns containing lists to summarise.
        :param step_column: (str, optional) Column specifying the step numbers of the lists.
        :param additional_columns: (list of str, optional) Additional columns to retain if values are consistent.
        :param dataset: (str) Name of the dataset.
        :param output_name: Name of the output file.
        :param save_directory: Path to save output.
    Returns:
        Returns pd.DataFrame: Summary statistics for the specified data columns.
        If output_name is given, saves the data as feather.
    """

    grouped = df.groupby(config_columns, observed=False)
    results = []

    for group_name, group in grouped:
        row = {col: group_name[i] for i, col in enumerate(config_columns)}

        if step_column:
            row[step_column] = group[step_column].iloc[0]

        # Handle additional columns
        if additional_columns:
            for col in additional_columns:
                unique_values = group[col].unique()
                if len(unique_values) > 1:
                    raise ValueError(
                        f"Column '{col}' has inconsistent values within group {group_name}."
                    )
                row[col] = unique_values[0]

        # Process data
        for data_col in value_columns:
            data = np.vstack(group[data_col].values)
            stats = utils.calculate_list_statistics(data)
            for stat_name, stat_values in stats.items():
                row[f"{data_col}_{stat_name}"] = stat_values
        results.append(row)

    if output_name != '':
        directory = save_directory / 'dataframes/'
        Path.mkdir(directory, parents=True, exist_ok=True)
        pd.DataFrame(results).to_feather(directory / f'{dataset}_{output_name}_process_summaries.feather')

    return pd.DataFrame(results)
