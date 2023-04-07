//! The stochastic simulation algorithm ([SSA](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm))
//! with a Monte-Carlo generating method.
//!
//! TODO example
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::Context;
use rand::Rng;
use rand_distr::Open01;

/// Number of individuals present in the system.
pub type NbIndividuals = u64;

/// The next reaction sampled by the SSA.
#[derive(Debug)]
pub struct NextReaction<Reaction>
where
    Reaction: std::fmt::Debug,
{
    /// The relative time at which this next reaction took place.
    pub time: f32,
    /// The event corresponding to the next iteration found by the
    /// [`AdvanceStep::next_reaction`].
    pub event: Reaction,
}

/// Whether to stop or continue the simulation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SimState {
    /// A simulation is stopped whether one of those conditions are met (see
    /// [`StopReason`]):
    ///
    /// 1. the maximal number of individuals has been reached,
    /// 2. the total population size is 0 (all lineages have died out),
    /// 3. the maximal number of iterations has been reached.
    ///
    Stop(StopReason),
    Continue,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StopReason {
    /// No individual left, all lineages have died out.
    NoIndividualsLeft,
    /// The maximal number of individual has been reached.
    MaxIndividualsReached,
    /// The maximal number of iterations has been reached.
    MaxItersReached,
    /// Absorbing state has been reached
    AbsorbingStateReached,
}

pub struct Options {
    pub max_iter: usize,
    pub init_iter: usize,
    pub max_cells: NbIndividuals,
    pub verbosity: u8,
}

/// The main loop running one realisation of a stochastic process with
/// `NB_REACTIONS` possible `REACTION`s.
pub fn simulate<P, REACTION, const NB_REACTIONS: usize>(
    state: &mut CurrentState<NB_REACTIONS>,
    rates: &ReactionRates<{ NB_REACTIONS }>,
    possible_reactions: &[REACTION; NB_REACTIONS],
    bd_process: &mut P,
    options: &Options,
    rng: &mut impl Rng,
) -> StopReason
where
    P: AdvanceStep<NB_REACTIONS, Reaction = REACTION>,
    REACTION: std::fmt::Debug,
{
    let mut iter = options.init_iter;
    loop {
        if state.population[0] + state.population[1] >= options.max_cells {
            return StopReason::MaxIndividualsReached;
        }

        let (sim_state, reaction) = bd_process.next_reaction(
            state,
            rates,
            possible_reactions,
            iter,
            options.max_iter - 1,
            rng,
        );

        if options.verbosity > 1 {
            println!("State: {:#?}, reaction: {:#?}", state, reaction);
        }

        match sim_state {
            SimState::Continue => {
                // unwrap is safe since SimState::Continue returns always
                // something (i.e. not None).
                let reaction = reaction.unwrap();

                // update the process according to the reaction
                bd_process.advance_step(reaction, rng);

                // update the state according to the process
                bd_process.update_state(state);
                iter += 1;
            }
            SimState::Stop(reason) => return reason,
        }

        // the absorbing state is when there are no NPlus cells and only NMinus
        // cells.
        if state.population[1] == 0 {
            return StopReason::AbsorbingStateReached;
        }
    }
}

/// The current state of a Markov process.
#[derive(Debug, Clone)]
pub struct CurrentState<const NB_REACTIONS: usize> {
    /// The number of individuals for all reactions.
    pub population: [NbIndividuals; NB_REACTIONS],
}

/// Perform an iteration of the SSA.
pub trait AdvanceStep<const NB_REACTIONS: usize> {
    type Reaction: std::fmt::Debug + Copy;

    fn next_reaction(
        &self,
        state: &CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<{ NB_REACTIONS }>,
        possible_reactions: &[Self::Reaction; NB_REACTIONS],
        iter: usize,
        max_iter: usize,
        rng: &mut impl Rng,
    ) -> (SimState, Option<NextReaction<Self::Reaction>>) {
        //! Find the next reaction in the system according to a
        //! [Monte-Carlo generating method](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm).
        //! ## Returns
        //! `None` if the maximal number of iterations have been reached or
        //! there aren't any individuals left in the total population, see
        //! [`SimState`].
        // StopIteration appears when there are no cells anymore (due to
        // cell death), when the iteration has reached the max number of
        // iterations nb_iter >= self.max_iter or maximal number of cells
        // i.e. when the iteration has generated a tumor of max_cells size
        if state.population.iter().sum::<u64>() == 0u64 {
            return (SimState::Stop(StopReason::NoIndividualsLeft), None);
        };
        if iter >= max_iter {
            return (SimState::Stop(StopReason::MaxItersReached), None);
        };

        let mut selected_event = 0_usize;
        let times = rates.compute_times_events(&state.population, rng);
        let mut smaller_waiting_time = times[0];
        for (idx, &waiting_time) in times.iter().enumerate() {
            if waiting_time <= smaller_waiting_time {
                smaller_waiting_time = waiting_time;
                selected_event = idx;
            }
        }
        (
            SimState::Continue,
            Some(NextReaction {
                time: smaller_waiting_time,
                event: possible_reactions[selected_event],
            }),
        )
    }

    /// Updates the process by stepping one step forward in the sumulation
    /// according to the [`NextReaction`] generated by
    /// [`AdvanceStep::next_reaction`].
    fn advance_step(&mut self, reaction: NextReaction<Self::Reaction>, rng: &mut impl Rng);

    /// Once the process has been updated by [`AdvanceStep::advance_step`],
    /// update back the [`CurrentState`] according to the process' rules.
    fn update_state(&self, state: &mut CurrentState<NB_REACTIONS>);
}

/// The rate of a reaction is the average number of occurrence of that reaction
/// in a time-unit.
#[derive(Debug, Clone)]
pub struct ReactionRates<const N: usize>(
    /// All the `N` rates for the `N` reactions present in the system.
    pub [f32; N],
);

impl<const N: usize> ReactionRates<N> {
    pub fn compute_times_events(
        &self,
        population: &[NbIndividuals; N],
        rng: &mut impl Rng,
    ) -> [f32; N] {
        //! Compute the Gillepsie-time for all reactions.
        //! The Gillespie-time is
        //! [defined](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm)
        //! as:
        //!
        //! `-ln(1 - r) / (population[i] * rates[i]) for i 0..N`
        //!
        //! where `r` is a random number and rates is `self.0`.
        //!
        let mut times = self.0;

        for i in 0..N {
            times[i] = exprand(times[i] * population[i] as f32, rng);
        }
        times
    }
}

pub fn exprand(lambda: f32, rng: &mut impl Rng) -> f32 {
    //! Generates a random waiting time using the exponential waiting time with
    //! parameter `lambda` of Poisson StochasticProcess.
    if (lambda - 0_f32).abs() < f32::EPSILON {
        f32::INFINITY
    } else {
        // random number between (0, 1)
        let val: f32 = rng.sample(Open01);
        -(1. - val).ln() / lambda
    }
}

pub fn write2file<T: std::fmt::Display>(
    data: &[T],
    path: &Path,
    header: Option<&str>,
    endline: bool,
) -> anyhow::Result<()> {
    //! Write vector of float into new file with a precision of 4 decimals.
    //! Write NAN if the slice to write to file is empty.
    fs::create_dir_all(path.parent().unwrap()).expect("Cannot create dir");
    let f = fs::OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(path)
        .with_context(|| "Cannot open stream")?;

    let mut buffer = BufWriter::new(f);

    if !data.is_empty() {
        if let Some(h) = header {
            writeln!(buffer, "{}", h)?;
        }

        for ele in data.iter() {
            write!(buffer, "{:.4},", ele)?;
        }

        if endline {
            writeln!(buffer)?;
        }
    } else {
        write!(buffer, "{},", f32::NAN)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck_macros::quickcheck;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::num::NonZeroU16;

    #[quickcheck]
    fn exprand_same_seed_test(lambda: f32, seed: u64) -> bool {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        if lambda == 0f32 {
            exprand(lambda, &mut rng).is_infinite()
        } else if lambda.is_nan() {
            exprand(lambda, &mut rng).is_nan()
        } else {
            let exp1 = exprand(lambda, &mut rng);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let exp2 = exprand(lambda, &mut rng);
            (exp1 - exp2).abs() < f32::EPSILON
        }
    }

    #[test]
    fn exprand_test() {
        let mut rng = ChaCha8Rng::seed_from_u64(1u64);
        let lambda = 0_f32;
        let first = exprand(lambda, &mut rng);
        assert!(first.is_infinite());
        let lambda = f32::INFINITY;
        let first = exprand(lambda, &mut rng);
        assert!((0f32 - first).abs() < f32::EPSILON);
    }

    struct TestNextReaction {
        population: [u64; 4],
    }
    impl AdvanceStep<4> for TestNextReaction {
        type Reaction = usize;
        fn advance_step(&mut self, reaction: NextReaction<Self::Reaction>, _rng: &mut impl Rng) {
            self.population[reaction.event] += 1;
        }
        fn update_state(&self, state: &mut CurrentState<4>) {
            state.population = self.population;
        }
    }

    #[quickcheck]
    fn advance_step_trait_where_only_second_reaction_is_possible_because_state_test(
        pop: NonZeroU16,
        seed: u64,
    ) -> bool {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let population = [0, pop.get() as NbIndividuals, 0, 0];
        let mut expected_population = population;
        expected_population[1] = population[1] + 1;
        let mut state = CurrentState { population };
        let rates = &ReactionRates([1., 1., 1., 1.]);
        let possible_reactions = &[0usize, 1usize, 2usize, 3usize];
        let mut process = TestNextReaction { population };
        let (sim_state, next_reaction) =
            process.next_reaction(&state, rates, possible_reactions, 0, 10, &mut rng);
        process.advance_step(next_reaction.unwrap(), &mut rng);
        assert_eq!(state.population, population);
        process.update_state(&mut state);

        state.population == expected_population
            && sim_state == SimState::Continue
            && process.population == expected_population
    }

    #[quickcheck]
    fn advance_step_trait_where_only_third_reaction_is_possible_because_rates_test(
        pop: NonZeroU16,
        seed: u64,
    ) -> bool {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let population = [
            pop.get() as NbIndividuals,
            pop.get() as NbIndividuals,
            pop.get() as NbIndividuals,
            pop.get() as NbIndividuals,
        ];
        let mut expected_population = population;
        expected_population[2] = population[2] + 1;
        let mut state = CurrentState { population };
        let rates = &ReactionRates([0., 0., 1., 0.]);
        let possible_reactions = &[0usize, 1usize, 2usize, 3usize];
        let mut process = TestNextReaction { population };
        let (sim_state, next_reaction) =
            process.next_reaction(&state, rates, possible_reactions, 0, 10, &mut rng);
        process.advance_step(next_reaction.unwrap(), &mut rng);
        assert_eq!(state.population, population);
        process.update_state(&mut state);

        state.population == expected_population
            && sim_state == SimState::Continue
            && process.population == expected_population
    }
}
