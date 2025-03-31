import json
import os
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import utils


def behaviour_tsne(seed: int, samples_per_plot: int, plot_hues: list, config: pd.DataFrame, training_set: str,
                   n_components: int, perplexity: int, save_directory: Path | str):

    output_dir = Path(save_directory / f'dimensionality_reduction/{training_set}/tsne')
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_data_path = Path(save_directory / f'dataframes/time_series')

    random_state = seed

    np.random.seed(seed)
    random.seed(seed)

    scaler = MinMaxScaler()

    X = utils.load_time_series(name=training_set, path=ts_data_path, drop_labels=config.columns, random_state=random_state)
    X = scaler.fit_transform(X)

    sample_length = len(X[0])
    number_samples = len(X)

    start = datetime.now()
    print(f"--- STARTED T-SNE EXPERIMENT ON {training_set} AT {start} ---")

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init='random', n_jobs=-1)
    X_new = tsne.fit_transform(X)

    print("--- EXPORTING RESULTS ---")

    if n_components == 2:
        for i in plot_hues:
            if i == 'Function':
                utils.plot_2d_with_function(X_new, config, 'Function', samples_per_plot, random_state,
                                            output_dir / f'tsne_functions_{n_components}_{perplexity}_{training_set}')
            elif i == 'Instance per Function':
                utils.plot_2d_with_custom_hue_per_function(X_new, config, 'Instance', samples_per_plot, random_state,
                                                           output_dir / f'tsne_instances_per_function_{n_components}_{perplexity}_{training_set}')
            elif i == 'Function per Group':
                utils.plot_2d_with_custom_hue_per_group(X_new, config, 'Function',
                                            output_dir / f'tsne_functions_per_group_{n_components}_{perplexity}_{training_set}')
            elif i == 'Groups':
                utils.plot_2d_with_group(X_new, config, 'Groups', samples_per_plot, random_state,
                                         output_dir / f'tsne_groups_{n_components}_{perplexity}_{training_set}')
            else:
                utils.plot_2d_with_custom_hue(X_new, config, i, samples_per_plot, random_state,
                                              output_dir / f'tsne_{i}_{n_components}_{perplexity}_{training_set}')

        utils.plot_2d(X_new, samples_per_plot, random_state,
                      output_dir / f'tsne_{n_components}_{perplexity}_{training_set}')

    Path.mkdir(save_directory / f'dimensionality_reduction/dataframes/{training_set}', parents=True, exist_ok=True)
    results = pd.DataFrame(X_new)
    results_data = pd.concat([config, results], axis=1)
    results_data.to_feather(save_directory / f'dimensionality_reduction/dataframes/{training_set}/tsne_data_{n_components}_{perplexity}_{training_set}.feather')
    results_values = {'Sample_length': sample_length,
                      'Number_samples': number_samples,
                      'KullbackLeiblerDivergence': tsne.kl_divergence_}
    with open(save_directory / f"dimensionality_reduction/{training_set}/tsne/tsne_values_{n_components}_{perplexity}_{training_set}.json", "w") as outfile:
        json.dump(results_values, outfile)

    end = datetime.now()

    print(f"--- FINISHED EXPERIMENT at {end} ---")
    print(f"--- TOTAL TIME: {end - start} ---")
