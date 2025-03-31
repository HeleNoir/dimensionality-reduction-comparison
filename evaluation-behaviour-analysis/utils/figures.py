from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# TODO see is and how figures can be set up better
def setup_figs(fig_size: (float, float) = (6.4, 4.8), font_scale: float = 1.6, fig_scale: float = 1.3):
    """
    Sets a fitting seaborn/matplotlib theme.
    """
    scaled_fig_size = (fig_size[0] * fig_scale, fig_size[1] * fig_scale)
    sns.set_theme(
        font_scale=font_scale,
        style='whitegrid',
        palette='colorblind',
        font='serif',
        context='paper',
        rc={'figure.figsize': scaled_fig_size}
    )


def setup_figs_descriptive():
    sns.set_theme(style="whitegrid",
                  font="Times New Roman",
                  font_scale=1,
                  context='paper',
                  palette='colorblind',
                  rc={
                      "lines.linewidth": 1,
                      "pdf.fonttype": 42,
                      "ps.fonttype": 42
                  })

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 200

    plt.tight_layout()


def setup_figs_latex():
    """
    Initializes the LaTeX/PGF backend.
    """
    # Always process .pdf files with pgf backend.
    import matplotlib.backend_bases
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

    # Set LaTeX parameters
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'text.usetex': True,
        'text.latex.preamble': '\n'.join([r"\usepackage{libertine}", r"\usepackage{amsmath}"]),
        'pgf.preamble': '\n'.join([r"\usepackage{libertine}", r"\usepackage{amsmath}"]),
        'pgf.rcfonts': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _save_or_show(save: Optional[str | Path] = None) -> None:
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


# ---
# General descriptive plots
# ---
def plot_descriptive_lineplots(df, x_col, y_col, hue, style, save):
    setup_figs_descriptive()
    sns.lineplot(df, x=x_col, y=y_col, hue=hue, style=style)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    _save_or_show(f'{save}_{x_col}_{y_col}')
    ax = sns.lineplot(df, x=x_col, y=y_col, hue=hue, style=style)
    ax.set(yscale='log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    _save_or_show(f'{save}_{x_col}_{y_col}_logscale')


def plot_descriptive_summarised_lineplots(df, x_col, y_col, std, save):
    setup_figs_descriptive()
    setup_figs()
    sns.lineplot(x=df[x_col], y=df[y_col])
    if y_col in ['DimensionWiseDiversity_mean', 'PairwiseDistanceDiversity_mean', 'MinimumIndividualDistance_mean', 'RadiusDiversity_mean']:
        plt.ylim((-0.2, 1))
    plus_std = df[y_col].to_numpy() + df[std].to_numpy()
    minus_std = df[y_col].to_numpy() - df[std].to_numpy()
    sns.lineplot(x=df[x_col], y=plus_std, color='b')
    sns.lineplot(x=df[x_col], y=minus_std, color='b')
    plt.fill_between(df[x_col].astype(int), minus_std.astype(float), plus_std.astype(float), alpha=.3)
    _save_or_show(f'{save}_{x_col}_{y_col}')


def plot_descriptive_distributions(df, x_col, save):
    setup_figs_descriptive()
    if x_col == 'AOCC':
        sns.histplot(data=df, x=x_col, binwidth=0.02, binrange=(0, 1))
        plt.xlim((0, 1))
    else:
        sns.histplot(data=df, x=x_col, bins=50)
    _save_or_show(f'{save}_{x_col}')
    sns.histplot(data=df, x=x_col, bins=50, log_scale=True)
    _save_or_show(f'{save}_{x_col}_logscale')


def plot_descriptive_heatmap(df, save):
    setup_figs_descriptive()
    sns.heatmap(df, annot=True, vmin=0.0, vmax=1.0)
    _save_or_show(save)


def plot_descriptive_ecdf(df, x_col, save):
    setup_figs_descriptive()
    sns.ecdfplot(data=df, x=x_col)
    _save_or_show(save)


def plot_descriptive_categorical(df, x_col, y_col, hue, save):
    setup_figs_descriptive()
    if hue == 'Function':
        ax = sns.catplot(df, x=x_col, y=y_col, hue=hue, kind='box', height=11, aspect=8/11)
    else:
        ax = sns.catplot(df, x=x_col, y=y_col, hue=hue, kind='box')
    if hue is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    _save_or_show(save)


# ---
# Plotting for dimensionality reduction
# ---
def plot_2d(data: np.ndarray, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction.
    """
    df = pd.DataFrame(data)
    sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, legend='full', s=50)
    _save_or_show(save)


def plot_2d_with_custom_hue(data: np.ndarray, config: pd.DataFrame, hue: str, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction, giving a different hue for every instance.
    """
    assert hue in config, 'Column for hue does not exist in dataframe!'
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    if hue == 'Pc':
        df[hue] = config[hue]
        df['Crossover'] = config['Crossover']
        groups = [y for x, y in df.groupby('Crossover', observed=False)]
        for group in groups:
            cr = group['Crossover'].iloc[0]
            hue_order = list(dict.fromkeys(group[hue].tolist()))
            sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            _save_or_show(f'{save}_{cr}')
    elif hue == 'MutationDistribution':
        df[hue] = config[hue]
        df['Mutation'] = config['Mutation']
        groups = [y for x, y in df.groupby('Mutation', observed=False)]
        for group in groups:
            mut = group['Mutation'].iloc[0]
            hue_order = list(dict.fromkeys(group[hue].tolist()))
            sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            _save_or_show(f'{save}_{mut}')
    else:
        df[hue] = config[hue]
        hue_order = list(dict.fromkeys(df[hue].tolist()))
        sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        _save_or_show(save)


def plot_2d_with_custom_hue_per_function(data: np.ndarray, config: pd.DataFrame, hue: str, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction of each individual function, giving a different hue for the set config parameter.
    """
    assert hue in config, 'Column for hue does not exist in dataframe!'
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    df[hue] = config[hue]
    df['Function'] = config['Function']
    functions = df['Function'].unique()
    n = int(n / len(functions))
    min_x = df[0].min()
    max_x = df[0].max()
    min_y = df[1].min()
    max_y = df[1].max()
    for f in functions:
        if hue == 'Pc':
            df['Crossover'] = config['Crossover']
            cr_groups = [y for x, y in df.groupby('Crossover', observed=False)]
            for cr_group in cr_groups:
                cr = cr_group['Crossover'].iloc[0]
                hue_order = list(dict.fromkeys(cr_group[hue].tolist()))
                sns.scatterplot(df.loc[df['Function'] == f].sample(n=n, random_state=random_state), x=0, y=1, hue=hue,
                                hue_order=hue_order, legend='full', s=50)
                plt.xlim((min_x, max_x))
                plt.ylim((min_y, max_y))
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                _save_or_show(f'{save}_{f}_{cr}')
        elif hue == 'MutationDistribution':
            df['Mutation'] = config['Mutation']
            mut_groups = [y for x, y in df.groupby('Mutation', observed=False)]
            for mut_group in mut_groups:
                mut = mut_group['Mutation'].iloc[0]
                hue_order = list(dict.fromkeys(mut_group[hue].tolist()))
                sns.scatterplot(df.loc[df['Function'] == f].sample(n=n, random_state=random_state), x=0, y=1, hue=hue,
                                hue_order=hue_order, legend='full', s=50)
                plt.xlim((min_x, max_x))
                plt.ylim((min_y, max_y))
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                _save_or_show(f'{save}_{f}_{mut}')
        else:
            hue_order = list(dict.fromkeys(df[hue].tolist()))
            sns.scatterplot(df.loc[df['Function'] == f].sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
            plt.xlim((min_x, max_x))
            plt.ylim((min_y, max_y))
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            _save_or_show(f'{save}_{f}')


def plot_2d_with_custom_hue_per_group(data: np.ndarray, config: pd.DataFrame, hue: str, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction of each function group, giving a different hue for the set config parameter.
    """
    assert hue in config, 'Column for hue does not exist in dataframe!'
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    df[hue] = config[hue]
    df['Group'] = config['Group']
    groups = df['Group'].unique()
    min_x = df[0].min()
    max_x = df[0].max()
    min_y = df[1].min()
    max_y = df[1].max()
    for g in groups:
        if hue == 'Pc':
            df['Crossover'] = config['Crossover']
            cr_groups = [y for x, y in df.groupby('Crossover', observed=False)]
            for cr_group in cr_groups:
                cr = cr_group['Crossover'].iloc[0]
                hue_order = list(dict.fromkeys(cr_group[hue].tolist()))
                sns.scatterplot(df.loc[df['Group'] == g], x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
                plt.xlim((min_x, max_x))
                plt.ylim((min_y, max_y))
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                _save_or_show(f'{save}_{g}_{cr}')
        elif hue == 'MutationDistribution':
            df['Mutation'] = config['Mutation']
            mut_groups = [y for x, y in df.groupby('Mutation', observed=False)]
            for mut_group in mut_groups:
                mut = mut_group['Mutation'].iloc[0]
                hue_order = list(dict.fromkeys(mut_group[hue].tolist()))
                sns.scatterplot(df.loc[df['Group'] == g], x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
                plt.xlim((min_x, max_x))
                plt.ylim((min_y, max_y))
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                _save_or_show(f'{save}_{g}_{mut}')
        else:
            hue_order = list(dict.fromkeys(df[hue].tolist()))
            sns.scatterplot(df.loc[df['Group'] == g], x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
            plt.xlim((min_x, max_x))
            plt.ylim((min_y, max_y))
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            _save_or_show(f'{save}_{g}')


# Can be done using plot_2d_with_custom_hue.
def plot_2d_with_instance(data: np.ndarray, config: pd.DataFrame, hue: str, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction, giving a different hue for every instance.
    """
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    df_hue = config
    c = df_hue['Config'].str.split('_')
    instances = [x[3] for x in c]
    df['Instance'] = instances
    hue_order = list(dict.fromkeys(instances))
    sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    _save_or_show(save)


# Can be done using plot_2d_with_custom_hue.
def plot_2d_with_function(data: np.ndarray, config: pd.DataFrame, hue: str, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction, giving a different hue for every function.
    """
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    df_hue = config
    c = df_hue['Config'].str.split('_')
    functions = [x[2] for x in c]
    df['Function'] = functions
    hue_order = list(dict.fromkeys(functions))
    ncols = 2
    sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=ncols, borderaxespad=0)
    _save_or_show(save)


def plot_2d_with_function_and_instance(data: np.ndarray, config: pd.DataFrame, hue: str, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction, giving a different hue for every function-instance combination.
    """
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    df_hue = config
    c = df_hue['Config'].str.split('_')
    functions = [x[2] for x in c]
    instances = [x[3] for x in c]
    df['Function'] = functions
    df['Instance'] = instances
    df['Function and Instance'] = df['Function'] + df['Instance']
    hue_order = list(dict.fromkeys(df['Function and Instance'].tolist()))
    ncols = 2
    sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=ncols, borderaxespad=0)
    _save_or_show(save)


# Should no longer be necessary as groups are now a column in the config dataframe.
# Can be done using plot_2d_with_custom_hue_per_function.
def plot_2d_with_group(data: np.ndarray, config: pd.DataFrame, hue: str, n: int, random_state: int, save: Optional[str | Path] = None):
    """
    Plots the 2d results from dimensionality reduction, giving a different hue for every function.
    """
    setup_figs((10, 7))
    df = pd.DataFrame(data)
    df_hue = config
    c = df_hue['Config'].str.split('_')
    functions = [x[2] for x in c]
    group1 = ['f001', 'f002', 'f003', 'f004', 'f005']
    group2 = ['f006', 'f007', 'f008', 'f009']
    group3 = ['f010', 'f011', 'f012', 'f013', 'f014']
    group4 = ['f015', 'f016', 'f017', 'f018', 'f019']
    group5 = ['f020', 'f021', 'f022', 'f023', 'f024']
    groups = []
    for f in functions:
        if f in group1:
            groups.append('Group1')
        elif f in group2:
            groups.append('Group2')
        elif f in group3:
            groups.append('Group3')
        elif f in group4:
            groups.append('Group4')
        elif f in group5:
            groups.append('Group5')
    df['Groups'] = groups
    hue_order = list(dict.fromkeys(groups))
    ncols = 1
    sns.scatterplot(df.sample(n=n, random_state=random_state), x=0, y=1, hue=hue, hue_order=hue_order, legend='full', s=50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=ncols, borderaxespad=0)
    _save_or_show(save)


# ---
# Plotting from former clustering approach
# ---
def plot_latent_clustering_2d(latent: np.ndarray, clustering: np.ndarray, n: int = 1000,
                              save: Optional[str | Path] = None):
    """
    Plots the clustering on a latent 2d space.
    """

    data = pd.DataFrame(latent)
    data['cluster'] = clustering
    sns.scatterplot(data.sample(n=n), x=0, y=1, hue='cluster', style='cluster', legend='full', s=100)
    _save_or_show(save)


def plot_latent_clustering_3d(latent: np.ndarray, clustering: np.ndarray, n: int = 1000,
                              save: Optional[str | Path] = None):
    """
    Plots the clustering on a latent 3d space.
    """

    data = pd.DataFrame(latent)
    ax = plt.figure().add_subplot(111, projection='3d')
    df = data.sample(n=n)
    x = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    z = np.array(df.iloc[:, 2])
    ax.scatter(x, y, z, c=df['cluster'], s=100, cmap="RdBu")
    _save_or_show(save)


def plot_cluster_frequency(clustering: np.ndarray, save: Optional[str | Path] = None):
    """
    Plots the frequency of each cluster in the clustering.
    """

    cluster, counts = np.unique(clustering, return_counts=True)
    data = pd.DataFrame({"cluster": cluster, "counts": counts})
    ax = sns.barplot(data, x="cluster", y="counts", hue="cluster", legend=False)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Frequency")
    _save_or_show(save)


def plot_series(X: np.ndarray, y_pred: np.ndarray, centroids: Optional[np.ndarray] = None,
                save: Optional[str | Path] = None):
    """
    Plots all series in each cluster along with the center series, if possible.
    """

    n_clusters = len(np.unique(y_pred))
    plt.figure(figsize=(7, 8))
    for yi in range(n_clusters):
        ax = plt.subplot(np.ceil(n_clusters / 2).astype(int), 2, yi + 1)
        for xx in X[y_pred == yi]:
            ax.plot(xx.ravel(), "k-", alpha=.2)
        if centroids is not None:
            ax.plot(centroids[yi].ravel(), "r-")
        ax.set_xlabel(f"Cluster {yi}")

    _save_or_show(save)


def plot_time_series(time_series: np.ndarray, save: Optional[str | Path] = None):
    """
    Plots a time series as lineplot.
    """

    data = pd.DataFrame(time_series).reset_index()
    ax = sns.lineplot(data, x="index", y=0)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective Value")
    _save_or_show(save)


def plot_cluster_examples(time_series: np.ndarray, clustering: np.ndarray, save_dir: str | Path, n: int = 10):
    """
    Plots n examples from each cluster individually in the corresponding cluster sub-folder.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    assert len(clustering) == len(time_series)
    n_clusters = np.max(clustering)
    for cluster in range(n_clusters+1):
        cluster_path = save_dir / f"{cluster}"
        print(f"Plotting examples for cluster {cluster} in {cluster_path}.")
        cluster_path.mkdir(exist_ok=True)
        time_series_in_cluster = time_series[clustering == cluster][:n]
        for i, series in enumerate(tqdm(time_series_in_cluster, leave=False, desc="Plotting time series.")):
            plot_time_series(series, save=cluster_path / f"curve_{i}.pdf")


def plot_cluster_examples_grid(time_series: np.ndarray, clustering: np.ndarray, save_dir: str | Path, n: int = 15):
    """
    Plots n examples from each cluster in a grid for the corresponding cluster image.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    assert len(clustering) == len(time_series)
    n_clusters = np.max(clustering)
    for cluster in range(n_clusters + 1):
        cluster_path = save_dir / f"{cluster}"
        print(f"Plotting examples for cluster {cluster} in {cluster_path}.")
        cluster_path.mkdir(exist_ok=True)
        time_series_in_cluster = time_series[clustering == cluster][:n]
        fig, axs = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle(f"Cluster {cluster}")
        for series, ax in zip(time_series_in_cluster, axs.flat):
            data = pd.DataFrame(series).reset_index()
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Objective Value")
            sns.lineplot(data, x="index", y=0, ax=ax)
        _save_or_show(save=cluster_path)

