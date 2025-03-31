"""
Specific functions for experiments-behaviour analysis.
Need to be adapted for other experiments
"""
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import utils

config_labels = [
    'Run',
    'Function',
    'Instance',
    'Dimension',
    'Pop_size',
    'Group'
]

config_dict = {
    'Run': 'int32',
    'Function': 'category',
    'Instance': 'category',
    'Dimension': 'category',
    'Pop_size': 'int32',
    'Group': 'category',
}

exploratory_common_labels = [
    'Iterations',
    'Evaluations',
    'BestObjectiveValue',
    'PopulationObjectiveValues',
    'DimensionWiseDiversity',
    'PairwiseDistanceDiversity',
    'MinimumIndividualDistance',
    'RadiusDiversity',
]

# TODO change for basic experiments
basic_common_labels = [
    'Iterations',
    'Evaluations',
    'BestObjectiveValue',
    'BestSolution',
    'PopulationObjectiveValues',
    'PairwiseDistanceDiversity',
    'MinimumIndividualDistance',
    'RadiusDiversity',
    'MeanStepSize',
    'StepSizeVariance',
]

GA_labels = [
    'Selection',
    'Crossover',
    'Pc',
    'Mutation',
    'MutationDistribution'
]

GA_dict = {
    'Selection': 'category',
    'Crossover': 'category',
    'Pc': 'category',
    'Mutation': 'category',
    'MutationDistribution': 'category'
}

GA_split_labels = [
    'Crossover',
    'Pc',
    'Mutation',
    'MutationDistribution'
]

DE_labels = [
    'Selection',
    'Y',
    'F',
    'Crossover',
    'Pc',
]

DE_split_labels = [
    'F',
    'Crossover',
    'Pc',
]

DE_dict = {
    'Selection': 'category',
    'Y': 'category',
    'F': 'category',
    'Crossover': 'category',
    'Pc': 'category',
}

PSO_labels = [
    'Weight',
    'C1',
    'C2'
]

PSO_dict = {
    'Weight': 'category',
    'C1': 'category',
    'C2': 'category'
}

ES_labels = [
    'Lambda',
    'Replacement',
    'Mutation',
    'MutationDistribution'
]

ES_dict = {
    'Lambda': 'category',
    'Replacement': 'category',
    'Mutation': 'category',
    'MutationDistribution': 'category'
}

# When we split the ES data for each lambda, we do not need lambda as a category for further analysis
ES_split_labels = [
    'Replacement',
    'Mutation',
    'MutationDistribution'
]

# BBOB function groups
group1 = ['f001', 'f002', 'f003', 'f004', 'f005']
group2 = ['f006', 'f007', 'f008', 'f009']
group3 = ['f010', 'f011', 'f012', 'f013', 'f014']
group4 = ['f015', 'f016', 'f017', 'f018', 'f019']
group5 = ['f020', 'f021', 'f022', 'f023', 'f024']


def exploratory_dict_to_df(data_dict, algorithm) -> pd.DataFrame:
    """
    Turn the dictionary of dataframes from exploratory experiments into one single dataframe, splitting the config
    information into individual columns.
    Labels are specified for experiments-behaviour-analysis
    """

    labels = ['Config'] + ['Algorithm'] + config_labels
    if algorithm == 'GA':
        labels += GA_labels
    elif algorithm == 'DE':
        labels += DE_labels
    elif algorithm == 'PSO':
        labels += PSO_labels
    elif algorithm == 'ES':
        labels += ES_labels

    labels += exploratory_common_labels

    convert_dict = {
        'Config': 'string',
        'Algorithm': 'string',
    }
    convert_dict.update(config_dict)

    df = pd.DataFrame(columns=labels)

    for key, value in data_dict.items():
        config = key.split('_')
        entry = [key, algorithm, config[0], config[2], config[3], config[4], config[5]]
        if config[2] in group1:
            entry.append('Group1')
        elif config[2] in group2:
            entry.append('Group2')
        elif config[2] in group3:
            entry.append('Group3')
        elif config[2] in group4:
            entry.append('Group4')
        elif config[2] in group5:
            entry.append('Group5')
        iterations = value['mahf::state::common::Iterations'].tolist()
        corrected_iterations = [iterations[0]] + [i + 1 for i in iterations[1:]]
        if algorithm == 'GA':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8])
            entry.append(config[9])
            entry.append(config[10].removesuffix(".cbor"))
            config_dict.update(GA_dict)
        elif algorithm == 'DE':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8])
            entry.append(config[9])
            entry.append(config[10].removesuffix(".cbor"))
            convert_dict.update(DE_dict)
        elif algorithm == 'PSO':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8].removesuffix(".cbor"))
            convert_dict.update(PSO_dict)
        elif algorithm == 'ES':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8])
            entry.append(config[9].removesuffix(".cbor"))
            convert_dict.update(ES_dict)
        entry.append(corrected_iterations)
        entry.append(value['mahf::state::common::Evaluations'].tolist())
        entry.append(value['BestObjectiveValue'].tolist())
        entry.append(value['Population Objective Values'].tolist())
        entry.append(value['mahf::components::measures::diversity::DimensionWiseDiversity'].tolist())
        entry.append(value['mahf::components::measures::diversity::PairwiseDistanceDiversity'].tolist())
        entry.append(value['mahf::components::measures::diversity::MinimumIndividualDistance'].tolist())
        entry.append(value['mahf::components::measures::diversity::RadiusDiversity'].tolist())

        print(len(entry))
        print(labels)
        df.loc[len(df)] = entry #TODO error mismatch length

    df = df.astype(convert_dict)

    return df


# TODO adapt this after deciding on data for basic experiments
def basic_dict_to_df(data_dict, algorithm, experiment, save_directory) -> pd.DataFrame:
    """
    Turn a dictionary of dataframes into one single dataframe, splitting the config information into individual columns.
    Labels are specified for experiments-behaviour-analysis
    """
    base_path = Path(__file__).parent

    labels = ['Config'] + ['Algorithm'] + config_labels
    if algorithm == 'GA':
        labels += GA_labels
    elif algorithm == 'DE':
        labels += DE_labels
    elif algorithm == 'PSO':
        labels += PSO_labels
    elif algorithm == 'ES':
        labels += ES_labels

    labels += basic_common_labels

    convert_dict = {
        'Config': 'string',
        'Algorithm': 'string',
    }
    convert_dict.update(config_dict)

    df = pd.DataFrame(columns=labels)
    df_pop = pd.DataFrame(columns=['Config', 'Population', 'PopulationObjectiveValues', 'EuclideanStepSize', 'FitnessImprovement'])
    for key, value in data_dict.items():
        config = key.split('_')
        entry = [key, algorithm, config[0], config[2], config[3], config[4], config[5]]
        if config[2] in group1:
            entry.append('Group1')
        elif config[2] in group2:
            entry.append('Group2')
        elif config[2] in group3:
            entry.append('Group3')
        elif config[2] in group4:
            entry.append('Group4')
        elif config[2] in group5:
            entry.append('Group5')
        iterations = value['mahf::state::common::Iterations'].tolist()
        corrected_iterations = [iterations[0]] + [i + 1 for i in iterations[1:]]
        if algorithm == 'GA':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8])
            entry.append(config[9])
            entry.append(config[10].removesuffix(".cbor"))
            config_dict.update(GA_dict)
        elif algorithm == 'DE':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8])
            entry.append(config[9])
            entry.append(config[10].removesuffix(".cbor"))
            convert_dict.update(DE_dict)
        elif algorithm == 'PSO':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8].removesuffix(".cbor"))
            convert_dict.update(PSO_dict)
        elif algorithm == 'ES':
            entry.append(config[6])
            entry.append(config[7])
            entry.append(config[8])
            entry.append(config[9].removesuffix(".cbor"))
            convert_dict.update(ES_dict)
        entry.append(corrected_iterations)
        entry.append(value['mahf::state::common::Evaluations'].tolist())
        entry.append(value['BestObjectiveValue'].tolist())
        entry.append(value['BestSolution'].tolist())
        entry.append(value['Population Objective Values'].tolist())
        entry.append(value['mahf::components::measures::diversity::PairwiseDistanceDiversity'].tolist())
        entry.append(value['mahf::components::measures::diversity::MinimumIndividualDistance'].tolist())
        entry.append(value['mahf::components::measures::diversity::RadiusDiversity'].tolist())
        entry.append(value['Mean Step Size'].tolist())
        entry.append(value['Step Size Variance'].tolist())

        entry_pop = [key, value['Population'].tolist(), value['Population Objective Values'].tolist(), value['mahf::components::measures::stepsize::EuclideanStepSize'].tolist(), value['mahf::components::measures::improvement::FitnessImprovement'].tolist()]
        df_pop.loc[len(df_pop)] = entry_pop

        df.loc[len(df)] = entry

    # Specify output folder
    directory = save_directory / f'dataframes'
    print(__file__)
    print(directory)
    Path.mkdir(directory, parents=True, exist_ok=True)
    df_pop.to_feather(f'{directory}/{experiment}_population.feather')

    del df_pop

    df = df.astype(convert_dict)

    return df


def add_dist_to_opt(df: pd.DataFrame, dataset_directory, functions=None):
    """
    Add a column for the distance to the actual optimum for the BBOB function-instance combinations included in a list.
    If no list is provided, all function-instance combinations are used.
    Requires a csv file with bbob function name and optimum values.
    """
    optima_path = dataset_directory / 'bbob_optima.csv'
    optima = pd.read_csv(optima_path, header=None, names=['Function', 'Optimum'])
    if functions is None:
        all_functions = optima['Function'].str.split('_')
        function_list = ['_'.join([all_functions[x][1]] + [all_functions[x][2]]) for x in range(0, len(all_functions))]
        function_list = list(set(function_list))
        df['DistanceToOptimum'] = df.apply(calculate_difference, axis=1, df2=optima, function_list=function_list)
        df['DistanceToOptimum'] = df['DistanceToOptimum'].apply(lambda arr: np.where(arr <= 0.0, sys.float_info.epsilon, arr))
    else:
        df['DistanceToOptimum'] = df.apply(calculate_difference, axis=1, df2=optima, function_list=functions)
        df['DistanceToOptimum'] = df['DistanceToOptimum'].apply(lambda arr: np.where(arr <= 0.0, sys.float_info.epsilon, arr))

    # Drop column after calculating DistanceToOptimum as we (hopefully) don't need it anymore
    df.drop(columns=['BestObjectiveValue'])


def calculate_difference(df1, df2, function_list):
    """ (Note: This function was created with the help of ChatGPT-4o.) """
    for name in function_list:
        if name in df1['Config']:
            matching_rows = df2[df2['Function'].str.contains(name, na=False)]

            if not matching_rows.empty:
                return df1['BestObjectiveValue'] - matching_rows['Optimum'].values[0]
    return None


def add_final_distance(df: pd.DataFrame):
    """
    Add a column for the final distance to the optimum (requires column 'DistanceToOptimum').
    """
    assert 'DistanceToOptimum' in df, 'Cannot add final distance: Column DistanceToOptimum is missing!'
    df['FinalDistance'] = df['DistanceToOptimum'].str[-1]


def add_final_aocc(df: pd.DataFrame):
    """
    Add a column for the overall AOCC value of the run.
    """
    assert 'DistanceToOptimum' in df, 'Cannot add AOCC: Column DistanceToOptimum is missing!'
    aocc_data = df[['DistanceToOptimum']].apply(lambda x: x['DistanceToOptimum'], axis=1, result_type='expand')
    transposed = aocc_data.T
    aocc_values = lambda x: utils.aocc(x)
    transposed = transposed.apply(aocc_values)
    aocc_data = transposed.T
    df['AOCC'] = aocc_data


def add_population_distances(df: pd.DataFrame, dataset_directory):
    """
    Add a column 'PopulationDistances' in the same style as 'PopulationObjectiveValues', but with the distance to the optimum instead of the
    absolute objective value of each individual at different iterations.
    Also adds columns for 'PopulationDistances_mean', 'PopulationDistances_std', 'PopulationDistances_median', 'PopulationDistances_min' and
    'PopulationDistances_max'.

    Beware: This takes forever, as the 'PopulationDistances' were logged in every iteration!
    """
    optima_path = dataset_directory / 'bbob_optima.csv'
    optima = pd.read_csv(optima_path, header=None, names=['Function', 'Optimum'])
    all_functions = optima['Function'].str.split('_')
    function_list = ['_'.join([all_functions[x][1]] + [all_functions[x][2]]) for x in range(0, len(all_functions))]
    function_list = list(set(function_list))

    df_population = df[['Config', 'PopulationObjectiveValues']]
    df_population = df_population.explode('PopulationObjectiveValues')
    df_population = df_population.dropna()
    df_population['PopulationDistances'] = df_population.apply(calculate_pop_differences, axis=1, df2=optima, function_list=function_list)
    df_population['PopulationDistances_mean'] = df_population['PopulationDistances'].apply(lambda x: np.array(x).mean(axis=0))
    df_population['PopulationDistances_std'] = df_population['PopulationDistances'].apply(lambda x: np.array(x).std(axis=0))
    df_population['PopulationDistances_median'] = df_population['PopulationDistances'].apply(lambda x: np.median(np.array(x), axis=0))
    df_population['PopulationDistances_min'] = df_population['PopulationDistances'].apply(lambda x: np.array(x).min(axis=0))
    df_population['PopulationDistances_max'] = df_population['PopulationDistances'].apply(lambda x: np.array(x).max(axis=0))

    df_population_distances = df_population.groupby([df_population.index, 'Config']).agg({'PopulationDistances': lambda x: x.tolist(),
                                                                                          'PopulationDistances_mean': lambda x: x.tolist(),
                                                                                          'PopulationDistances_std': lambda x: x.tolist(),
                                                                                          'PopulationDistances_median': lambda x: x.tolist(),
                                                                                          'PopulationDistances_min': lambda x: x.tolist(),
                                                                                          'PopulationDistances_max': lambda x: x.tolist()})
    df['PopulationDistances'] = df_population_distances['PopulationDistances'].values
    df['PopulationDistances_mean'] = df_population_distances['PopulationDistances_mean'].values
    df['PopulationDistances_std'] = df_population_distances['PopulationDistances_std'].values
    df['PopulationDistances_median'] = df_population_distances['PopulationDistances_median'].values
    df['PopulationDistances_min'] = df_population_distances['PopulationDistances_min'].values
    df['PopulationDistances_max'] = df_population_distances['PopulationDistances_max'].values

    # Drop column after calculating population statistics as we (hopefully) don't need it anymore
    df.drop(columns=['PopulationObjectiveValues'])


def calculate_pop_differences(df1, df2, function_list):
    for name in function_list:
        if name in df1['Config']:
            matching_rows = df2[df2['Function'].str.contains(name, na=False)]

            if not matching_rows.empty:
                return df1['PopulationObjectiveValues'] - matching_rows['Optimum'].values[0]
    return None


def add_population_distances_faster(df: pd.DataFrame, dataset_directory):
    """
    Add a column 'PopulationDistances' and related statistics (calculated per sublist) to the DataFrame.

    Adaptation from 'add_population_distances' created with the help of ChatGPT-4o.
    For RS_d02_p10 about 10x faster than 'add_population_distances'.
    If further problems arise, either vectorise, parallelise, or reduce logging interval.
    """
    # Load the optima data
    optima_path = dataset_directory / 'bbob_optima.csv'
    optima = pd.read_csv(optima_path, header=None, names=['Function', 'Optimum'])

    # Preprocess function list
    all_functions = optima['Function'].str.split('_')
    optima['ShortName'] = ['_'.join([func[1], func[2]]) for func in all_functions]

    # Map Config to Optimum values
    config_to_optimum = optima.set_index('ShortName')['Optimum'].to_dict()

    # Filter relevant configs
    df['ConfigShortName'] = df['Config'].str.extract(f"({'|'.join(config_to_optimum.keys())})", expand=False)
    filtered_df = df[df['ConfigShortName'].notna()]  # Drop irrelevant rows

    # Prepare storage for results
    population_distances = []
    distances_mean = []
    distances_std = []
    distances_median = []
    distances_min = []
    distances_max = []

    # Calculate distances for each row
    for _, row in filtered_df.iterrows():
        optimum = config_to_optimum[row['ConfigShortName']]
        distances = np.array(row['PopulationObjectiveValues']) - optimum

        # Append the full distances
        population_distances.append(distances.tolist())

        # Compute statistics for each list within the distances
        distances_mean.append([np.mean(sublist) for sublist in distances])
        distances_std.append([np.std(sublist) for sublist in distances])
        distances_median.append([np.median(sublist) for sublist in distances])
        distances_min.append([np.min(sublist) for sublist in distances])
        distances_max.append([np.max(sublist) for sublist in distances])

    # Assign computed values back to the filtered DataFrame
    filtered_df['PopulationDistances'] = population_distances
    filtered_df['PopulationDistances_mean'] = distances_mean
    filtered_df['PopulationDistances_std'] = distances_std
    filtered_df['PopulationDistances_median'] = distances_median
    filtered_df['PopulationDistances_min'] = distances_min
    filtered_df['PopulationDistances_max'] = distances_max

    # Merge results back into the original DataFrame
    result_df = df.merge(
        filtered_df[['PopulationDistances', 'PopulationDistances_mean', 'PopulationDistances_std',
                     'PopulationDistances_median', 'PopulationDistances_min', 'PopulationDistances_max']],
        left_index=True,
        right_index=True,
        how='left'
    )

    # Drop the helper column
    result_df.drop(columns=['ConfigShortName'], inplace=True)

    # Drop column after calculating population statistics as we (hopefully) don't need it anymore
    result_df.drop(columns=['PopulationObjectiveValues'])

    return result_df
