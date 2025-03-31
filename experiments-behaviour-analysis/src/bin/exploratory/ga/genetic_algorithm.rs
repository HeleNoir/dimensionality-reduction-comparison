use mahf::{prelude::*,
           components::{initialization, replacement,
                        archive::{IntermediateArchiveUpdate, ElitistArchiveIntoPopulation, ElitistArchiveUpdate},
                        measures::{diversity::{DimensionWiseDiversity, PairwiseDistanceDiversity, MinimumIndividualDistance, RadiusDiversity},
                                   improvement::{FitnessImprovement},
                                   stepsize::{EuclideanStepSize},}},
           configuration::Configuration, logging::Logger,
           problems::{LimitedVectorProblem, SingleObjectiveProblem, KnownOptimumProblem}};

pub fn behaviour_ga<P>(
    evaluations: u32,
    population_size: u32,
    elitists: usize,
    selection: Box<dyn Component<P>>,
    crossover: Box<dyn Component<P>>,
    mutation: Box<dyn Component<P>>,
) -> Configuration<P>
where P: SingleObjectiveProblem + LimitedVectorProblem<Element = f64> + KnownOptimumProblem,
{
    Configuration::builder()
        .do_(initialization::RandomSpread::new(population_size))
        .evaluate()
        .update_best_individual()
        .do_(DimensionWiseDiversity::new())
        .do_(PairwiseDistanceDiversity::new())
        .do_(MinimumIndividualDistance::new())
        .do_(RadiusDiversity::new())
        .do_(Logger::new())
        .while_(
            conditions::LessThanN::evaluations(evaluations),
            |builder| {
                builder
                    .do_(Box::from(ElitistArchiveIntoPopulation))
                    .do_(selection)
                    .do_(crossover)
                    .do_(mutation)
                    .do_(boundary::Saturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(Box::from(ElitistArchiveUpdate::new(elitists)))
                    .do_(replacement::Generational::new(population_size))
                    .do_(DimensionWiseDiversity::new())
                    .do_(PairwiseDistanceDiversity::new())
                    .do_(MinimumIndividualDistance::new())
                    .do_(RadiusDiversity::new())
                    .do_(Logger::new())
            },
        )
        .build()
}