import click
from pathlib import Path
import pandas as pd
import os

import utils


@click.command()
@click.option('-d', '--dataset', type=click.STRING, default='RS_d02_p10')
@click.option('-e', '--experiment', type=click.STRING, default='exploratory')
def main(dataset: str, experiment: str) -> None:
    base_path = Path(__file__).parent
    save_directory = (Path(base_path / '..' / 'data' / experiment)).resolve()
    df = pd.read_feather(save_directory / f'dataframes/{dataset}_full.feather')

    Path.mkdir(save_directory / f'dataframes/time_series', parents=True, exist_ok=True)

    config = dataset.split('_')
    algorithm = config[0]

    if experiment == 'exploratory':
        granularities = [10, 50, 100]
        for granularity in granularities:
            df_ts = utils.prepare_ts_dataset(df, 'DistanceToOptimum', granularity, algorithm)
            mean_ts = utils.prepare_ts_dataset_means(df, 'DistanceToOptimum', granularity, algorithm)
            aocc_ts = utils.prepare_ts_dataset_aocc(df, 'DistanceToOptimum', granularity, algorithm)

            df_diversity_dw = utils.prepare_ts_dataset(df, 'DimensionWiseDiversity', granularity, algorithm)
            df_diversity_dw_mean = utils.prepare_ts_dataset_means(df, 'DimensionWiseDiversity', granularity, algorithm)
            df_diversity_pwd = utils.prepare_ts_dataset(df, 'PairwiseDistanceDiversity', granularity, algorithm)
            df_diversity_pwd_mean = utils.prepare_ts_dataset_means(df, 'PairwiseDistanceDiversity', granularity, algorithm)
            df_diversity_mid = utils.prepare_ts_dataset(df, 'MinimumIndividualDistance', granularity, algorithm)
            df_diversity_mid_mean = utils.prepare_ts_dataset_means(df, 'MinimumIndividualDistance', granularity, algorithm)
            df_diversity_rd = utils.prepare_ts_dataset(df, 'RadiusDiversity', granularity, algorithm)
            df_diversity_rd_mean = utils.prepare_ts_dataset_means(df, 'RadiusDiversity', granularity, algorithm)

            df_ts.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_{granularity}.feather')
            mean_ts.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_mean_{granularity}.feather')
            aocc_ts.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_aocc_{granularity}.feather')

            df_diversity_dw.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_dw_{granularity}.feather')
            df_diversity_dw_mean.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_dw_mean_{granularity}.feather')
            df_diversity_pwd.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_pwd_{granularity}.feather')
            df_diversity_pwd_mean.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_pwd_mean_{granularity}.feather')
            df_diversity_mid.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_mid_{granularity}.feather')
            df_diversity_mid_mean.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_mid_mean_{granularity}.feather')
            df_diversity_rd.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_rd_{granularity}.feather')
            df_diversity_rd_mean.to_feather(save_directory / f'dataframes/time_series/{dataset}_ts_rd_mean_{granularity}.feather')

            del df_ts, mean_ts, aocc_ts, df_diversity_dw, df_diversity_dw_mean, df_diversity_pwd, df_diversity_pwd_mean, df_diversity_mid, df_diversity_mid_mean, df_diversity_rd, df_diversity_rd_mean


if __name__ == '__main__':
    main()
