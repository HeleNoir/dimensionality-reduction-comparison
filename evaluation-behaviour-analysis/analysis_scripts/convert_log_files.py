import os
from typing import Optional


import click
import pathlib

import pandas as pd

import utils

@click.command()
@click.option('-a', '--algorithm', type=click.STRING, default='RS')
@click.option('-d', '--dimension', type=click.STRING, default='d2')
@click.option('-p', '--pop_size', type=click.STRING, default='p10')
@click.option('-e', '--experiment', type=click.STRING, default='exploratory')
@click.option('-s', '--datasplit', type=click.STRING, default='none')
def main(algorithm: str, dimension: str, pop_size: str, experiment: str, datasplit: str) -> None:
    base_path = pathlib.Path(__file__).parent
    if datasplit == 'none':
        folder = rf'{experiment}/{algorithm}/{dimension}_{pop_size}'
    else:
        folder = rf'{experiment}/{algorithm}/{dimension}_{pop_size}/{datasplit}'
    log_directory = (pathlib.Path(base_path / '..' / '..'/'experiments-behaviour-analysis'/'data'/folder)).resolve()
    save_directory = (pathlib.Path(base_path / '..' / 'data' / experiment)).resolve()
    dataset_directory = (pathlib.Path(base_path / '..' / 'datasets'))
    pathlib.Path.mkdir(save_directory, parents=True, exist_ok=True)

    print(__file__)
    print(log_directory)
    print(save_directory)

    if dimension == 'd2':
        dim = 'd02'
    elif dimension == 'd5':
        dim = 'd05'
    else:
        dim = dimension

    logs = utils.read_log_dir_with_joblib(log_directory)

    df = pd.DataFrame
    # TODO provide options for basic and detailed experiments
    if experiment == 'exploratory':
        df = utils.exploratory_dict_to_df(logs, algorithm)
        print(df.head())

    pathlib.Path.mkdir(save_directory / f'dataframes', parents=True, exist_ok=True)

    utils.add_dist_to_opt(df, dataset_directory)
    utils.add_final_distance(df)
    utils.add_final_aocc(df)

    df = utils.add_population_distances_faster(df, dataset_directory)
    if datasplit == 'none':
        df.to_feather(save_directory / f'dataframes/{algorithm}_{dim}_{pop_size}_full.feather')
    else:
        df.to_feather(save_directory / f'dataframes/{algorithm}_{dim}_{pop_size}_{datasplit}_full.feather')


if __name__ == '__main__':
    main()
