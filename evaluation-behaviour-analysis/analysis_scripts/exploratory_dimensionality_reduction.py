from pathlib import Path

import click

import dimensionality_reduction
import utils


@click.command()
@click.option('-s', '--seed', type=click.INT, default=42)
@click.option('-a', '--algorithm', type=click.STRING, default='RS')
@click.option('-d', '--dataset', type=click.STRING, default='RS_d02_p10')
@click.option('-e', '--samples_per_plot', type=click.INT, default=600)
def main(seed: int, algorithm: str, dataset: str, samples_per_plot: int) -> None:
    base_path = Path(__file__).parent
    save_directory = (Path(base_path / '..' / 'data' / 'exploratory')).resolve()

    print(__file__)
    print(save_directory)

    components = [2, 3, 4]
    perplexities = [2, 5, 10, 30, 50, 100]
    neighbors = [5, 15, 30]
    distances = [0.05, 0.1, 0.3, 0.5]
    hues = ['Function', 'Instance per Function', 'Function per Group', 'Groups']
    granularities = [10, 50, 100]
    additional_labels = []

    if algorithm == 'GA':
        hues += utils.GA_split_labels
        additional_labels += utils.GA_labels
    elif algorithm == 'PSO':
        hues += utils.PSO_labels
        additional_labels += utils.PSO_labels
    elif algorithm == 'DE':
        hues += utils.DE_split_labels
        additional_labels += utils.DE_labels
    elif algorithm == 'ES':
        hues += utils.ES_split_labels
        additional_labels += utils.ES_labels

    for granularity in granularities:
        training_sets = [
        f'{dataset}_ts_{granularity}',
        f'{dataset}_ts_mean_{granularity}',
        f'{dataset}_ts_aocc_{granularity}',
        f'{dataset}_ts_pwd_{granularity}',
        f'{dataset}_ts_pwd_mean_{granularity}',
        ]

        for training_set in training_sets:
            dimensionality_reduction.comparison(algorithm, seed, training_set, components, perplexities, neighbors, distances, samples_per_plot, hues, additional_labels, save_directory)


if __name__ == '__main__':
    main()
