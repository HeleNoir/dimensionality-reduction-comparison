import pandas as pd

import utils
import dimensionality_reduction

from pathlib import Path


def comparison(algorithm, seed, training_set, components, perplexities, neighbors, distances, samples_per_plot, hues,
               additional_labels, save_directory):
    """
    Run the comparison of all five dimensionality reduction techniques with the specified parameters on the training set.
    Plots the 2D representations and provides the results as json.

    :param algorithm: Algorithm from which the data results.
    :param seed: Seed.
    :param training_set: Training set name.
    :param components: List of number of components to be tested with PCA, MDS and ISOMAP.
    :param perplexities: List of perplexities to be tested with t-SNE.
    :param neighbors: List of number of neighbors to be tested with ISOMAP and UMAP.
    :param distances: List of distances to be tested with UMAP.
    :param samples_per_plot: Number of samples to be included in plots; maximum should be the number of examples in the respective trainingset.
    :param hues: List of hues to apply to plots; options are 'Function', 'Instance', 'Instance per Function', 'Groups', or a custum hue that corresponds to a column in the coonfig dataframe
    :param additional_labels: List of algorithm components that are included in config dataframe.
    :param save_directory: Directory to get datasets from and save results in.
    :return:
    """
    labels = ['Config'] + utils.config_labels + additional_labels

    df_ts = pd.read_feather(save_directory / f'dataframes/time_series/{training_set}.feather')
    config = df_ts[labels]

    for n_components in components:
        dimensionality_reduction.behaviour_pca(seed, samples_per_plot, hues, config,
                                               training_set, n_components, save_directory)
        if samples_per_plot <= 1000:
            dimensionality_reduction.behaviour_mds(seed, samples_per_plot, hues, config,
                                                   training_set, n_components, True, save_directory)

        for n_neighbors in neighbors:
            dimensionality_reduction.behaviour_isomap(seed, samples_per_plot, hues,
                                                      config, training_set, n_components, n_neighbors, save_directory)

    for perplexity in perplexities:
        dimensionality_reduction.behaviour_tsne(seed, samples_per_plot, hues, config,
                                                training_set, 2, perplexity, save_directory)

    for n_neighbors in neighbors:
        for min_dist in distances:
            dimensionality_reduction.behaviour_umap(seed, samples_per_plot, hues,
                                                    config, training_set, 2, n_neighbors, min_dist, 'euclidean',
                                                    save_directory)
