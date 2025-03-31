import os
from pathlib import Path

import pandas as pd

import utils


# TODO here general functions that combine multiple plots implemented in figures.py


# Plot individual info (all data for one or more configs)
# TODO data: 'FitnessImprovement'?
def plot_descriptive_individual_run(df: pd.DataFrame, value_columns: [str] = None, hue: str = None, style: str = None,
                                    dataset: str = None, algorithm: str = None, config_name: str = None, save_directory: str | Path = None):
    """
    Creates a .png with the plot for every specified columns over the iterations.
    If a dataframe with more than one row is provided, the plots contain the mean line and the standard deviation.
    If a dataframe with more than one row and a hue is provided, the plots contain individual lines depending on the hue.

    :param df: Dataframe with data to be plotted.
    :param value_columns: List of columns to be plotted, defaults to 'DistanceToOptimum', 'PopulationDistances_mean', 'RadiusDiversity', 'MeanStepSize', 'StepSizeVariance'.
    :param hue: Hue of lines (optional) if several rows are provided.
    :param style: Style of lines (optional) if several rows are provided.
    :param config_name: Name to add to the saved .png (optional); if not provided, will use the first config of the dataframe.
    """
    if value_columns is None:
        value_columns = ['DistanceToOptimum', 'PopulationDistances_mean']
    output_dir = save_directory / f'descriptive/plots/{dataset}/individual_lineplots/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    for col in value_columns:
        df_full = df[list({'Config', 'Iterations', col, hue, style})]
        df_full = df_full.explode(['Iterations', col])
        df_full = df_full.dropna()
        utils.plot_descriptive_lineplots(df_full, 'Iterations', col, hue, style,
                                         f'{output_dir}/{algorithm}_{config_name}')
        del df_full


# Plot summarised behaviour data in lineplots
# seaborn lineplot: summary per function
# data: all behaviour data
def plot_summarised_lineplots(df: pd.DataFrame, value_columns, hue, style: str = None, dataset: str = '', config_name: str = '', save_directory: str | Path = None):
    if value_columns is None:
        value_columns = ['DistanceToOptimum_mean', 'PopulationDistances_mean_mean']
    output_dir = save_directory / f'descriptive/plots/{dataset}/summary_lineplots/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for col in value_columns:
        if style is None:
            df_full = df[list({'Iterations', col, hue})]
        else:
            df_full = df[list({'Iterations', col, hue, style})]
        df_full = df_full.explode(['Iterations', col])
        df_full = df_full.dropna()
        utils.plot_descriptive_lineplots(df_full, 'Iterations', col, hue, style, f'{output_dir}/{config_name}')
        del df_full


def plot_distributions(df: pd.DataFrame, dataset, config_name, save_directory: str | Path):
    """
    Plot distribution of FinalDistance of results in normal and log scale
    """
    plot_df = df[['FinalDistance']]
    plot_df_aocc = df[['AOCC']]
    output_dir = save_directory / f'descriptive/plots/{dataset}/distributions/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    utils.plot_descriptive_distributions(plot_df, 'FinalDistance', f'{output_dir}/{dataset}_{config_name}')
    utils.plot_descriptive_distributions(plot_df_aocc, 'AOCC', f'{output_dir}/{dataset}_{config_name}')
    del plot_df


# Plot diversity comparison
# seaborn lineplot: individual runs or summary per function
# data: all three diverstiy measures, maybe also 'DistanceToOptimum' and 'PopulationDistances_mean'
def plot_diversity_comparison(df: pd.DataFrame, value_columns, dataset: str, config_name: str, save_directory: str | Path):
    output_dir = save_directory / f'descriptive/plots/{dataset}/diversity/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for col in value_columns:
        df_full = df[list({'Iterations', col})]
        df_full = df_full.explode(['Iterations', col])
        df_full = df_full.dropna()
        utils.plot_descriptive_lineplots(df_full, 'Iterations', col, None, None, f'{output_dir}/{config_name}')
        del df_full


def plot_diversity_summarised(df: pd.DataFrame, mean_columns, std_columns, dataset: str, config_name: str, save_directory: str | Path):
    output_dir = save_directory / f'descriptive/plots/{dataset}/diversity/summary/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    for mean, std in zip(mean_columns, std_columns):
        df_full = df[['Iterations', mean, std]]
        df_full = df_full.explode(['Iterations', mean, std])
        utils.plot_descriptive_summarised_lineplots(df_full, 'Iterations', mean, std, f'{output_dir}/{config_name}')
        del df_full


# Plot AOCC heatmap per config
# seaborn heatmap
# data: mean AOCC per config
def plot_aocc_heatmap(df: pd.DataFrame, index: str, columns: str, dataset: str, config_name: str, save_directory: str | Path):
    output_dir = save_directory / f'descriptive/plots/{dataset}/heatmaps/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    heatmap_df = df.pivot(index=index, columns=columns, values="AOCC_mean")
    utils.plot_descriptive_heatmap(heatmap_df, f'{output_dir}/AOCC_heatmap_{config_name}_{index}_{columns}')


# Plot ECDF
# seaborn ecdfplot: groups? configs?
# data: 'FinalDistance', 'AOCC' (mean over functions/groups/configs)
def plot_ecdf_results(df: pd.DataFrame, value_column: str, dataset: str, config_name: str, save_directory: str | Path):
    output_dir = save_directory / f'descriptive/plots/{dataset}/ecdfs/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    utils.plot_descriptive_ecdf(df, value_column, f'{output_dir}/{dataset}_{config_name}_{value_column}_ecdf')


# Plot comparison config aspects
# seaborn catplot/pointplot
# influence parameter setting or operator selection on 'FinalDistance' and 'AOCC'
def plot_component_influences(df: pd.DataFrame, algorithm: str, hue: str = None, dataset: str = None, config: str = None, save_directory: str | Path = None):

    cat_columns = []
    if algorithm:
        if algorithm == 'GA':
            cat_columns = utils.GA_split_labels
        elif algorithm == 'PSO':
            cat_columns = utils.PSO_labels
        elif algorithm == 'DE':
            cat_columns = utils.DE_split_labels
        elif algorithm == 'ES':
            cat_columns = utils.ES_split_labels

    output_dir = save_directory / f'descriptive/plots/{dataset}/components/'
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    for col in cat_columns:
        if col == 'Pc':
            groups = [y for x, y in df.groupby('Crossover', observed=False)]
            for group in groups:
                cr = group['Crossover'].iloc[0]
                if hue is None:
                    df_cat = group[list({'AOCC', 'FinalDistance', col})]
                else:
                    df_cat = group[list({'AOCC', 'FinalDistance', col, hue})]
                utils.plot_descriptive_categorical(df_cat, 'AOCC', col, hue,
                                                   f'{output_dir}/{dataset}_{cr}_{col}_{config}_AOCC')
                utils.plot_descriptive_categorical(df_cat, 'FinalDistance', col, hue,
                                                   f'{output_dir}/{dataset}_{cr}_{col}_{config}_FinalDistance')
                del df_cat

        elif col == 'MutationDistribution':
            groups = [y for x, y in df.groupby('Mutation', observed=False)]
            for group in groups:
                mut = group['Mutation'].iloc[0]
                if hue is None:
                    df_cat = group[list({'AOCC', 'FinalDistance', col})]
                else:
                    df_cat = group[list({'AOCC', 'FinalDistance', col, hue})]
                utils.plot_descriptive_categorical(df_cat, 'AOCC', col, hue,
                                                   f'{output_dir}/{dataset}_{mut}_{col}_{config}_AOCC')
                utils.plot_descriptive_categorical(df_cat, 'FinalDistance', col, hue,
                                                   f'{output_dir}/{dataset}_{mut}_{col}_{config}_FinalDistance')
                del df_cat

        else:
            if hue is None:
                df_cat = df[list({'AOCC', 'FinalDistance', col})]
            else:
                df_cat = df[list({'AOCC', 'FinalDistance', col, hue})]
            utils.plot_descriptive_categorical(df_cat, 'AOCC', col, hue, f'{output_dir}/{dataset}_{col}_{config}_AOCC')
            utils.plot_descriptive_categorical(df_cat, 'FinalDistance', col, hue,
                                               f'{output_dir}/{dataset}_{col}_{config}_FinalDistance')
            del df_cat
