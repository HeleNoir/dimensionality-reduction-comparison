pub mod genetic_algorithm;

use crate::genetic_algorithm::behaviour_ga;

use mahf::{prelude::*, configuration::Configuration, Random,
           lens::common::{BestObjectiveValueLens, ObjectiveValuesLens, PopulationLens},
           components::measures::{diversity::{NormalizedDiversityLens, DimensionWiseDiversity, PairwiseDistanceDiversity, MinimumIndividualDistance, RadiusDiversity},
                                  improvement::{FitnessImprovement, TotalImprovementLens},
                                  stepsize::{EuclideanStepSize, IndividualStepSizeLens, MeanStepSizeLens, StepSizeVarianceLens}},
};
use mahf_coco::{Instance, AcceleratedEvaluator, Suite, Context, Options, backends::C, Name::Bbob};

use std::{
    fs::{self},
    path::PathBuf,
    sync::{Arc},
};
use once_cell::sync::Lazy;
use clap::Parser;
use itertools::iproduct;
use mahf::components::utils::Noop;
use mahf::lens::common::BestSolutionLens;
use mahf::problems::LimitedVectorProblem;
use rayon::prelude::*;

static CONTEXT: Lazy<Context<C>> = Lazy::new(Context::default);

#[derive(Parser)]
#[clap(version, about)]
struct Args {
    /// Number of BBOB function
    #[arg(long, default_value_t = 1)]
    function: usize,

    /// Dimensions of BBOB function
    #[arg(long, default_value_t = 2)]
    dimensions: usize,

    /// Population size of algorithm
    #[arg(long, default_value_t = 10)]
    population_size: u32,

    /// Selection operator of algorithm
    //  let selections = vec!["tournament", "roulette", "linrank"];
    #[arg(long, default_value_t = String::from("tournament"))]
    selection: String,

}


fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let pop_size = args.population_size;
    let functions = args.function;
    let dimensions: usize = args.dimensions;
    let selection = args.selection;

    let folder = format!("data/exploratory/GA/d{:?}_p{:?}/{}", dimensions, pop_size, selection);

    // set number of runs per instance
    let runs = [1, 2, 3, 4, 5];
    // set number of evaluations and iterations
    let evaluations: u32 = (10000 * dimensions) as u32;
    let iterations = (evaluations - pop_size)/pop_size;


    // set GA parameter
    let pcs = [0.5, 0.8, 1.0];
    let no_pc = [0.0];
    let sigmas = [0.01, 0.1, 0.5];
    let elitists: usize = 1;
    let tournament_size = 2;
    let crossovers = vec!["uniform", "arithmetic"];
    let no_crossover = vec!["none"];
    let mutations = vec!["gaussian", "uniform"];
    let mutation_params = vec!["first", "second", "third"];

    let mut configs_no_crossover: Vec<(f64, &str, &str, &str)> = iproduct!(no_pc, no_crossover, mutations.clone(), mutation_params.clone()).collect::<Vec<_>>();
    let mut configs: Vec<_> = iproduct!(pcs, crossovers, mutations, mutation_params).collect();
    configs.append(&mut configs_no_crossover);

    // set the benchmark problems
    let instance_indices = 1..6;
    let index: Vec<usize> = instance_indices.clone().collect();

    let n = runs.len() as u64;
    let m = index.len() as u64;

    let seeds: Vec<Vec<u64>> = (0..n)
        .map(|i| ((i * m + 1)..((i + 1) * m + 1)).collect())
        .collect();

    let options = Options::new()
        .with_dimensions([dimensions])
        .with_function_indices([functions])
        .with_instance_indices(instance_indices);
    let mut suite = Suite::with_options(Bbob, None, Some(&options)).unwrap();

    let mut problems = Vec::new();
    let mut evaluators = Vec::new();

    while let Some(instance) = suite.next() {
        let evaluator = AcceleratedEvaluator::new(&CONTEXT, &mut suite, &instance);
        problems.push(instance);
        evaluators.push(evaluator);
    }

    runs.into_par_iter()
        .zip(std::iter::repeat(evaluators).take(runs.len()).collect::<Vec<_>>())
        .for_each(|(run, evaluator)| {

            for config in &configs {

                for (i, (instance, eval)) in problems.iter().zip(evaluator.iter()).enumerate() {
                    let evaluator = eval.clone();

                    let seed = seeds[run - 1][i];

                    let s = if selection == "tournament" {
                        selection::Tournament::new(pop_size, tournament_size)
                    } else if selection == "roulette" {
                        selection::RouletteWheel::new(pop_size, 0.01)
                    } else {
                        selection::LinearRank::new(pop_size)
                    };

                    let crossover = if config.1 == "arithmetic" {
                        recombination::ArithmeticCrossover::new(config.0, true)
                    } else if config.1 == "uniform" {
                        recombination::UniformCrossover::new(config.0, true)
                    } else {
                        Noop::new()
                    };

                    let mut_param = if config.2 == "uniform" && config.3 == "first" {
                        instance.domain()[0].end.clone()
                    } else if config.2 == "uniform" && config.3 == "second" {
                        instance.domain()[0].end.clone() / 5.0
                    } else if config.2 == "uniform" && config.3 == "third" {
                        instance.domain()[0].end.clone() / 10.0
                    } else if config.2 == "gaussian" && config.3 == "first"{
                        sigmas[0]
                    } else if config.2 == "gaussian" && config.3 == "second" {
                        sigmas[1]
                    } else {
                        sigmas[2]
                    };

                    let mutation = if config.2 == "gaussian" {
                        mutation::NormalMutation::new(mut_param, 1.0)
                    } else {
                        mutation::UniformMutation::new(mut_param, 1.0)
                    };

                    let conf: Configuration<Instance> = behaviour_ga(
                        evaluations,
                        pop_size,
                        elitists,
                        s,
                        crossover,
                        mutation,
                    );

                    let output = format!("{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
                                         run,
                                         "_",
                                         instance.name(),
                                         "_",
                                         pop_size,
                                         "_",
                                         selection, // Selection
                                         "_",
                                         config.1, // Crossover
                                         "_",
                                         config.0, // Pc
                                         "_",
                                         config.2, // Mutation
                                         "_",
                                         mut_param, // Mutation Distribution
                    );

                    let data_dir = Arc::new(PathBuf::from(&folder));
                    fs::create_dir_all(data_dir.as_ref()).expect("TODO: panic message");

                    let experiment_desc = output;
                    let log_file = data_dir.join(format!("{}.cbor", experiment_desc));

                    let setup = conf.optimize_with(&instance, |state: &mut State<_>| -> ExecResult<()> {
                        state.insert_evaluator(evaluator);
                        state.insert(Random::new(seed));
                        state.configure_log(|con| {
                            con
                                .with_many(
                                    conditions::EveryN::iterations(1),
                                    [
                                        ValueOf::<common::Evaluations>::entry(),
                                        BestObjectiveValueLens::entry(),
                                        ObjectiveValuesLens::entry(),
                                        NormalizedDiversityLens::<DimensionWiseDiversity>::entry(),
                                        NormalizedDiversityLens::<PairwiseDistanceDiversity>::entry(),
                                        NormalizedDiversityLens::<MinimumIndividualDistance>::entry(),
                                        NormalizedDiversityLens::<RadiusDiversity>::entry(),
                                    ],
                                )
                            ;
                            Ok(())
                        })
                    });
                    setup.unwrap().log().to_cbor(log_file).expect("TODO: panic message");
                }
            }
        });
    Ok(())
}