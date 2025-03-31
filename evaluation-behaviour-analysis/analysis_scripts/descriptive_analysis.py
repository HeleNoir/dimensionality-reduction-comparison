import pathlib

import click
from pathlib import Path
import pandas as pd
import os

import utils
import descriptive_statistics
import utils.ba_specifics


@click.command()
@click.option('-d', '--dataset', type=click.STRING, default='RS_d02_p10')
@click.option('-a', '--algorithm', type=click.STRING, default='RS')
@click.option('-e', '--experiment', type=click.STRING, default='exploratory')
def main(dataset: str, algorithm: str, experiment: str) -> None:
    base_path = Path(__file__).parent
    save_directory = (Path(base_path / '..' / 'data' / experiment)).resolve()

    df = pd.read_feather(save_directory / f'dataframes/{dataset}_full.feather')
    experiment_directory = save_directory / f'descriptive/{dataset}/'
    pathlib.Path.mkdir(experiment_directory, parents=True, exist_ok=True)

    # Number of best and worst configs to add details from to csv
    n_best = 10
    n_worst = 10

    conf_labels = utils.ba_specifics.config_labels
    alg_labels = []
    if algorithm == 'GA':
        alg_labels = utils.GA_labels
    elif algorithm == 'PSO':
        alg_labels = utils.PSO_labels
    elif algorithm == 'DE':
        alg_labels = utils.DE_labels
    elif algorithm == 'ES':
        alg_labels = utils.ES_labels

    split_labels = []
    if algorithm == 'GA':
        split_labels = utils.GA_split_labels
    elif algorithm == 'PSO':
        split_labels = utils.PSO_labels
    elif algorithm == 'DE':
        split_labels = utils.DE_split_labels
    elif algorithm == 'ES':
        split_labels = utils.ES_split_labels

    value_columns = []
    mean_columns = []
    diversity_columns = []
    div_mean_columns = []
    div_std_columns = []
    if experiment == 'exploratory':
        value_columns = ['DistanceToOptimum',
                         'DimensionWiseDiversity',
                         'PairwiseDistanceDiversity',
                         'MinimumIndividualDistance',
                         'RadiusDiversity',
                         'PopulationDistances_mean']
        mean_columns = ['DistanceToOptimum_mean',
                        'DimensionWiseDiversity_mean',
                        'PairwiseDistanceDiversity_mean',
                        'MinimumIndividualDistance_mean',
                        'RadiusDiversity_mean',
                        'PopulationDistances_mean_mean']
        diversity_columns = ['DistanceToOptimum',
                             'DimensionWiseDiversity',
                             'PairwiseDistanceDiversity',
                             'MinimumIndividualDistance',
                             'RadiusDiversity',
                             'PopulationDistances_mean']
        div_mean_columns = ['DistanceToOptimum_mean',
                            'DimensionWiseDiversity_mean',
                            'PairwiseDistanceDiversity_mean',
                            'MinimumIndividualDistance_mean',
                            'RadiusDiversity_mean',
                            'PopulationDistances_mean_mean']
        div_std_columns = ['DistanceToOptimum_std',
                           'DimensionWiseDiversity_std',
                           'PairwiseDistanceDiversity_std',
                           'MinimumIndividualDistance_std',
                           'RadiusDiversity_std',
                           'PopulationDistances_mean_std']
    elif experiment == 'basic':
        # TODO adapt to basic experiment
        value_columns = ['DistanceToOptimum',
                         'PairwiseDistanceDiversity',
                         'PopulationDistances_mean']
        mean_columns = ['DistanceToOptimum_mean',
                        'PairwiseDistanceDiversity_mean',
                        'PopulationDistances_mean_mean']
        diversity_columns = ['DistanceToOptimum',
                             'PairwiseDistanceDiversity',
                             'PopulationDistances_mean']
        div_mean_columns = ['DistanceToOptimum_mean',
                            'PairwiseDistanceDiversity_mean',
                            'PopulationDistances_mean_mean']
        div_std_columns = ['DistanceToOptimum_std',
                           'PairwiseDistanceDiversity_std',
                           'PopulationDistances_mean_std']

    # Most general
    # best and worst configs considering AOCC
    best_aocc = df.nlargest(n_best, 'AOCC')
    worst_aocc = df.nsmallest(n_worst, 'AOCC')

    # best and worst configs considering FinalDistance
    best_distance_to_opt = df.nsmallest(n_best, 'FinalDistance')
    worst_distance_to_opt = df.nlargest(n_worst, 'FinalDistance')

    best_aocc.to_csv(experiment_directory / f'{dataset}_best_aocc.csv',
                     columns=conf_labels + alg_labels + ['AOCC', 'FinalDistance'])
    worst_aocc.to_csv(experiment_directory / f'{dataset}_worst_aocc.csv',
                      columns=conf_labels + alg_labels + ['AOCC', 'FinalDistance'])
    best_distance_to_opt.to_csv(experiment_directory / f'{dataset}_best_distance_to_opt.csv',
                                columns=conf_labels + alg_labels + ['AOCC', 'FinalDistance'])
    worst_distance_to_opt.to_csv(experiment_directory / f'{dataset}_worst_distance_to_opt.csv',
                                 columns=conf_labels + alg_labels + ['AOCC', 'FinalDistance'])

    descriptive_statistics.plot_distributions(df, dataset, 'all', save_directory)

    del best_aocc, worst_aocc, best_distance_to_opt, worst_distance_to_opt

    group_dfs = [y for x, y in df.groupby('Group', observed=False)]
    for group_df in group_dfs:
        group = group_df['Group'].iloc[0]
        descriptive_statistics.plot_distributions(group_df, dataset, f'all_{group}', save_directory)

    del group_dfs

    function_dfs = [y for x, y in df.groupby('Function', observed=False)]
    for function_df in function_dfs:
        f = function_df['Function'].iloc[0]
        descriptive_statistics.plot_distributions(function_df, dataset, f'all_{f}', save_directory)

    del function_dfs

    descriptive_statistics.plot_ecdf_results(df, 'AOCC', dataset, 'all', save_directory)

    # Summary statistics of process data, summarised over all
    df_process_stats_all = descriptive_statistics.summarise_process_stats(df,
                                                                          ['Algorithm'],
                                                                          value_columns,
                                                                          'Iterations',
                                                                          None,
                                                                          dataset,
                                                                          'all',
                                                                          save_directory)

    # Summary statistics of final values, summarised over all
    df_final_stats_all = descriptive_statistics.summarise_final_stats(df,
                                                                      ['Algorithm'],
                                                                      ['FinalDistance', 'AOCC'],
                                                                      None,
                                                                      dataset,
                                                                      'all',
                                                                      save_directory)

    descriptive_statistics.plot_summarised_lineplots(df_process_stats_all,
                                                     value_columns=mean_columns,
                                                     hue='Algorithm',
                                                     dataset=dataset,
                                                     config_name='all_process', save_directory=save_directory)

    descriptive_statistics.plot_diversity_summarised(df_process_stats_all, div_mean_columns, div_std_columns, dataset, f'{dataset}_all_diversity',
                                                     save_directory)

    del df_process_stats_all, df_final_stats_all

    # Problem-dependent analysis
    # Summary statistics of process data, summarised by function group
    df_group_process_stats = descriptive_statistics.summarise_process_stats(df,
                                                                            ['Group'],
                                                                            value_columns,
                                                                            'Iterations',
                                                                            ['Algorithm'],
                                                                            dataset,
                                                                            f'groups',
                                                                            save_directory)

    # Summary statistics of process data, summarised by function
    df_function_process_stats = descriptive_statistics.summarise_process_stats(df,
                                                                               ['Function'],
                                                                               value_columns,
                                                                               'Iterations',
                                                                               ['Algorithm', 'Group'],
                                                                               dataset,
                                                                               f'functions',
                                                                               save_directory)

    # Summary statistics of final values, summarised by function group
    df_group_stats = descriptive_statistics.summarise_final_stats(df,
                                                                  ['Group'],
                                                                  ['FinalDistance', 'AOCC'],
                                                                  ['Algorithm'],
                                                                  dataset,
                                                                  f'groups',
                                                                  save_directory)

    # Summary statistics of final values, summarised by function
    df_function_stats = descriptive_statistics.summarise_final_stats(df,
                                                                     ['Function'],
                                                                     ['FinalDistance', 'AOCC'],
                                                                     ['Algorithm'],
                                                                     dataset,
                                                                     f'functions',
                                                                     save_directory)

    descriptive_statistics.plot_summarised_lineplots(df_function_process_stats,
                                                     value_columns=mean_columns,
                                                     hue='Function',
                                                     dataset=dataset,
                                                     config_name='function_process', save_directory=save_directory)

    descriptive_statistics.plot_summarised_lineplots(df_group_process_stats,
                                                     value_columns=mean_columns,
                                                     hue='Group',
                                                     dataset=dataset,
                                                     config_name='group_process', save_directory=save_directory)

    descriptive_statistics.plot_aocc_heatmap(df_group_stats, 'Algorithm', 'Group', dataset, 'all', save_directory)
    descriptive_statistics.plot_aocc_heatmap(df_function_stats, 'Function', 'Algorithm', dataset, 'all', save_directory)

    descriptive_statistics.plot_ecdf_results(df_function_stats, 'AOCC_mean', dataset, 'function', save_directory)
    descriptive_statistics.plot_ecdf_results(df_function_stats, 'FinalDistance_mean', dataset, 'function',
                                             save_directory)

    descriptive_statistics.plot_ecdf_results(df_group_stats, 'AOCC_mean', dataset, 'group', save_directory)
    descriptive_statistics.plot_ecdf_results(df_group_stats, 'FinalDistance_mean', dataset, 'group', save_directory)

    group_param_dfs = [y for x, y in df_group_process_stats.groupby('Group', observed=False)]
    for group_param_df in group_param_dfs:
        value = group_param_df['Group'].iloc[0]
        descriptive_statistics.plot_diversity_summarised(group_param_df, div_mean_columns, div_std_columns, dataset, f'{dataset}_{value}', save_directory)

    del df_group_process_stats, df_function_process_stats, df_group_stats, df_function_stats

    if algorithm != 'RS':
        # Component-dependent analysis

        descriptive_statistics.plot_component_influences(df, algorithm, None, dataset, 'all', save_directory)
        descriptive_statistics.plot_component_influences(df, algorithm, 'Group', dataset, 'groups', save_directory)
        descriptive_statistics.plot_component_influences(df, algorithm, 'Function', dataset, 'functions',
                                                         save_directory)

        for parameter in split_labels:
            if parameter == 'Pc':
                groups = [y for x, y in df.groupby('Crossover', observed=False)]
                for group in groups:
                    cr = group['Crossover'].iloc[0]
                    df_group_process_stats_conf = descriptive_statistics.summarise_process_stats(group,
                                                                                                 ['Group', parameter],
                                                                                                 value_columns,
                                                                                                 'Iterations',
                                                                                                 ['Algorithm'],
                                                                                                 dataset,
                                                                                                 f'groups_{cr}_{parameter}',
                                                                                                 save_directory)

                    df_function_process_stats_conf = descriptive_statistics.summarise_process_stats(group,
                                                                                                    ['Function', parameter],
                                                                                                    value_columns,
                                                                                                    'Iterations',
                                                                                                    ['Algorithm', 'Group'],
                                                                                                    dataset,
                                                                                                    f'functions_{cr}_{parameter}',
                                                                                                    save_directory)

                    df_group_stats_conf = descriptive_statistics.summarise_final_stats(group,
                                                                                       ['Group', parameter],
                                                                                       ['FinalDistance', 'AOCC'],
                                                                                       ['Algorithm'],
                                                                                       dataset,
                                                                                       f'groups_{cr}_{parameter}',
                                                                                       save_directory)

                    df_function_stats_conf = descriptive_statistics.summarise_final_stats(group,
                                                                                          ['Function', parameter],
                                                                                          ['FinalDistance', 'AOCC'],
                                                                                          ['Algorithm'],
                                                                                          dataset,
                                                                                          f'functions_{cr}_{parameter}',
                                                                                          save_directory)

                    descriptive_statistics.plot_summarised_lineplots(df_function_process_stats_conf,
                                                                     value_columns=mean_columns,
                                                                     hue='Function',
                                                                     style=parameter,
                                                                     dataset=dataset,
                                                                     config_name=f'function_process_{cr}_{parameter}',
                                                                     save_directory=save_directory)

                    descriptive_statistics.plot_summarised_lineplots(df_group_process_stats_conf,
                                                                     value_columns=mean_columns,
                                                                     hue='Group',
                                                                     style=parameter,
                                                                     dataset=dataset,
                                                                     config_name=f'group_process_{cr}_{parameter}',
                                                                     save_directory=save_directory)

                    del df_function_process_stats_conf

                    param_dfs = [y for x, y in group.groupby(parameter, observed=False)]
                    for param_df in param_dfs:
                        value = param_df[parameter].iloc[0].replace('.', '')
                        descriptive_statistics.plot_distributions(param_df, dataset, f'{cr}_{parameter}_{value}',
                                                                  save_directory)

                    del param_dfs

                    descriptive_statistics.plot_aocc_heatmap(df_group_stats_conf, parameter, 'Group', dataset,
                                                             f'{cr}_{parameter}', save_directory)
                    descriptive_statistics.plot_aocc_heatmap(df_function_stats_conf, 'Function', parameter, dataset,
                                                             f'{cr}_{parameter}', save_directory)

                    del df_group_stats_conf, df_function_stats_conf

                    group_param_dfs = [y for x, y in
                                       df_group_process_stats_conf.groupby(['Group', parameter], observed=False)]
                    for group_param_df in group_param_dfs:
                        value = group_param_df['Group'].iloc[0]
                        param_value = group_param_df[parameter].iloc[0].replace('.', '')
                        descriptive_statistics.plot_diversity_summarised(group_param_df, div_mean_columns,
                                                                         div_std_columns, dataset,
                                                                         f'{dataset}_{cr}_{parameter}_{param_value}_{value}',
                                                                         save_directory)

                    del group_param_dfs, df_group_process_stats_conf

            elif parameter == 'MutationDistribution':
                groups = [y for x, y in df.groupby('Mutation', observed=False)]
                for group in groups:
                    mut = group['Mutation'].iloc[0]
                    df_group_process_stats_conf = descriptive_statistics.summarise_process_stats(group,
                                                                                                 ['Group', parameter],
                                                                                                 value_columns,
                                                                                                 'Iterations',
                                                                                                 ['Algorithm'],
                                                                                                 dataset,
                                                                                                 f'groups_{mut}_{parameter}',
                                                                                                 save_directory)

                    df_function_process_stats_conf = descriptive_statistics.summarise_process_stats(group,
                                                                                                    ['Function', parameter],
                                                                                                    value_columns,
                                                                                                    'Iterations',
                                                                                                    ['Algorithm', 'Group'],
                                                                                                    dataset,
                                                                                                    f'functions_{mut}_{parameter}',
                                                                                                    save_directory)

                    df_group_stats_conf = descriptive_statistics.summarise_final_stats(group,
                                                                                       ['Group', parameter],
                                                                                       ['FinalDistance', 'AOCC'],
                                                                                       ['Algorithm'],
                                                                                       dataset,
                                                                                       f'groups_{mut}_{parameter}',
                                                                                       save_directory)

                    df_function_stats_conf = descriptive_statistics.summarise_final_stats(group,
                                                                                          ['Function', parameter],
                                                                                          ['FinalDistance', 'AOCC'],
                                                                                          ['Algorithm'],
                                                                                          dataset,
                                                                                          f'functions_{mut}_{parameter}',
                                                                                          save_directory)
                    descriptive_statistics.plot_summarised_lineplots(df_function_process_stats_conf,
                                                                     value_columns=mean_columns,
                                                                     hue='Function',
                                                                     style=parameter,
                                                                     dataset=dataset,
                                                                     config_name=f'function_process_{mut}_{parameter}',
                                                                     save_directory=save_directory)

                    descriptive_statistics.plot_summarised_lineplots(df_group_process_stats_conf,
                                                                     value_columns=mean_columns,
                                                                     hue='Group',
                                                                     style=parameter,
                                                                     dataset=dataset,
                                                                     config_name=f'group_process_{mut}_{parameter}',
                                                                     save_directory=save_directory)

                    del df_function_process_stats_conf

                    param_dfs = [y for x, y in group.groupby(parameter, observed=False)]
                    for param_df in param_dfs:
                        value = param_df[parameter].iloc[0].replace('.', '')
                        descriptive_statistics.plot_distributions(param_df, dataset, f'{mut}_{parameter}_{value}',
                                                                  save_directory)

                    del param_dfs

                    descriptive_statistics.plot_aocc_heatmap(df_group_stats_conf, parameter, 'Group', dataset,
                                                             f'{mut}_{parameter}', save_directory)
                    descriptive_statistics.plot_aocc_heatmap(df_function_stats_conf, 'Function', parameter, dataset,
                                                             f'{mut}_{parameter}', save_directory)

                    del df_group_stats_conf, df_function_stats_conf

                    group_param_dfs = [y for x, y in
                                       df_group_process_stats_conf.groupby(['Group', parameter], observed=False)]
                    for group_param_df in group_param_dfs:
                        value = group_param_df['Group'].iloc[0]
                        param_value = group_param_df[parameter].iloc[0].replace('.', '')
                        descriptive_statistics.plot_diversity_summarised(group_param_df, div_mean_columns,
                                                                         div_std_columns, dataset,
                                                                         f'{dataset}_{mut}_{parameter}_{param_value}_{value}',
                                                                         save_directory)

                    del group_param_dfs, df_group_process_stats_conf

            else:
                df_group_process_stats_conf = descriptive_statistics.summarise_process_stats(df,
                                                                                             ['Group', parameter],
                                                                                             value_columns,
                                                                                             'Iterations',
                                                                                             ['Algorithm'],
                                                                                             dataset,
                                                                                             f'groups_{parameter}',
                                                                                             save_directory)

                df_function_process_stats_conf = descriptive_statistics.summarise_process_stats(df,
                                                                                                ['Function', parameter],
                                                                                                value_columns,
                                                                                                'Iterations',
                                                                                                ['Algorithm', 'Group'],
                                                                                                dataset,
                                                                                                f'functions_{parameter}',
                                                                                                save_directory)

                df_group_stats_conf = descriptive_statistics.summarise_final_stats(df,
                                                                                   ['Group', parameter],
                                                                                   ['FinalDistance', 'AOCC'],
                                                                                   ['Algorithm'],
                                                                                   dataset,
                                                                                   f'groups_{parameter}', save_directory)

                df_function_stats_conf = descriptive_statistics.summarise_final_stats(df,
                                                                                      ['Function', parameter],
                                                                                      ['FinalDistance', 'AOCC'],
                                                                                      ['Algorithm'],
                                                                                      dataset,
                                                                                      f'functions_{parameter}',
                                                                                      save_directory)

                descriptive_statistics.plot_summarised_lineplots(df_function_process_stats_conf,
                                                                 value_columns=mean_columns,
                                                                 hue='Function',
                                                                 style=parameter,
                                                                 dataset=dataset,
                                                                 config_name=f'function_process_{parameter}',
                                                                 save_directory=save_directory)

                descriptive_statistics.plot_summarised_lineplots(df_group_process_stats_conf,
                                                                 value_columns=mean_columns,
                                                                 hue='Group',
                                                                 style=parameter,
                                                                 dataset=dataset,
                                                                 config_name=f'group_process_{parameter}',
                                                                 save_directory=save_directory)

                del df_function_process_stats_conf

                param_dfs = [y for x, y in df.groupby(parameter, observed=False)]
                for param_df in param_dfs:
                    value = param_df[parameter].iloc[0].replace('.', '')
                    descriptive_statistics.plot_distributions(param_df, dataset, f'{parameter}_{value}', save_directory)

                del param_dfs

                descriptive_statistics.plot_aocc_heatmap(df_group_stats_conf, parameter, 'Group', dataset, parameter,
                                                         save_directory)
                descriptive_statistics.plot_aocc_heatmap(df_function_stats_conf, 'Function', parameter, dataset,
                                                         parameter, save_directory)

                del df_group_stats_conf, df_function_stats_conf

                group_param_dfs = [y for x, y in df_group_process_stats_conf.groupby(['Group', parameter], observed=False)]
                for group_param_df in group_param_dfs:
                    value = group_param_df['Group'].iloc[0]
                    param_value = group_param_df[parameter].iloc[0].replace('.', '')
                    descriptive_statistics.plot_diversity_summarised(group_param_df, div_mean_columns, div_std_columns, dataset,
                                                                     f'{dataset}_{parameter}_{param_value}_{value}',
                                                                     save_directory)

                del group_param_dfs, df_group_process_stats_conf


if __name__ == '__main__':
    main()
