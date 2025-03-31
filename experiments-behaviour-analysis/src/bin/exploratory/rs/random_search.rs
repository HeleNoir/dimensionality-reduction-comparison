use mahf::{prelude::*,
           components::{initialization, mutation, replacement, selection,
                        archive::{IntermediateArchiveUpdate},
                        measures::{diversity::{DimensionWiseDiversity, PairwiseDistanceDiversity, MinimumIndividualDistance, RadiusDiversity},
                                   improvement::{FitnessImprovement},
                                   stepsize::{EuclideanStepSize},}},
           configuration::Configuration, logging::Logger,
           problems::{LimitedVectorProblem, SingleObjectiveProblem, KnownOptimumProblem}};


pub fn behaviour_rs<P>(
    evaluations: u32,
    population_size: u32,
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
                    .do_(selection::All::new())
                    .do_(mutation::PartialRandomSpread::new_full())
                    .evaluate()
                    .update_best_individual()
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