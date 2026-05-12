//! The stochastic simulation algorithm ([SSA](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm))
//! with a Monte-Carlo generating method.
//!
//! `sosa` runs the SSA over agents that carry individual properties
//! evolving over time. A typical use case: human cells that reproduce
//! asexually and acquire new point mutations on each division. With `sosa`
//! you can track the per-cell mutation count while letting different
//! sub-populations reproduce at different rates.
//!
//! If you only need to track aggregate counts and don't care about
//! per-individual state, prefer [`rebop`](https://crates.io/crates/rebop).
//!
//! # Overview
//!
//! To run a simulation, pass the following to [`simulate`]:
//!
//! - a type implementing [`AdvanceStep`] that owns your per-individual
//!   state and describes how the process advances on each reaction,
//! - an initial [`CurrentState`] holding the per-reaction population
//!   counts,
//! - a [`ReactionRates`] array giving the per-individual rate of each
//!   reaction,
//! - an array of reaction identifiers (`[REACTION; NB_REACTIONS]`)
//!   matching the order of `rates`,
//! - an [`Options`] value bounding the run (max iterations, max simulated
//!   time, max population),
//! - a random number generator implementing [`rand_core::Rng`].
//!
//! See [`simulate`] for a runnable example.

#![warn(missing_docs)]

use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::Context;
use log::{debug, trace};
use rand_core::Rng;
use thiserror::Error;

/// Number of individuals present in the system.
pub type NbIndividuals = u64;

/// The next reaction sampled by the SSA — the outcome of
/// [`AdvanceStep::next_reaction`].
#[derive(Debug)]
pub struct NextReaction<Reaction>
where
    Reaction: std::fmt::Debug,
{
    /// Waiting time until this reaction takes place, drawn from the SSA's
    /// exponential distribution.
    pub time: f32,
    /// Identifier of the reaction that takes place.
    pub event: Reaction,
}

/// Whether the simulation should stop or continue after the current step.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SimState {
    /// The simulation has reached a terminating condition; the carried
    /// [`StopReason`] explains which.
    Stop(StopReason),
    /// The simulation should advance another step.
    Continue,
}

/// Why the simulation terminated. Returned by [`simulate`] and carried
/// inside [`SimState::Stop`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum StopReason {
    /// All lineages have died out — the total population reached zero.
    NoIndividualsLeft,
    /// The population reached [`Options::max_cells`].
    MaxIndividualsReached,
    /// The iteration counter reached [`IterTime::iter`] of
    /// [`Options::max_iter_time`].
    MaxItersReached,
    /// The process entered an absorbing state from which no further
    /// reactions are possible.
    AbsorbingStateReached,
    /// Simulated time reached [`IterTime::time`] of
    /// [`Options::max_iter_time`].
    MaxTimeReached,
}

/// A pair of (iteration count, simulated time) used to bound or report the
/// simulation's progress.
#[derive(Clone, Debug)]
pub struct IterTime {
    /// Number of SSA steps taken.
    pub iter: usize,
    /// Cumulative simulated time, in the units implied by [`ReactionRates`].
    pub time: f32,
}

/// Run-configuration passed to [`simulate`].
#[derive(Clone, Debug)]
pub struct Options {
    /// Starting value of the iteration counter.
    pub init_iter: usize,
    /// Upper bounds on iterations and simulated time. Hitting either
    /// terminates the run with [`StopReason::MaxItersReached`] or
    /// [`StopReason::MaxTimeReached`].
    pub max_iter_time: IterTime,
    /// Population cap; when total population reaches this value the run
    /// ends with [`StopReason::MaxIndividualsReached`].
    pub max_cells: NbIndividuals,
}

/// Run one realisation of a stochastic process with `NB_REACTIONS`
/// possible reactions.
///
/// Drives the SSA loop: on each iteration,
/// [`AdvanceStep::next_reaction`] samples the next reaction,
/// [`AdvanceStep::advance_step`] applies it to the user's process, and
/// [`AdvanceStep::update_state`] writes the result back into the shared
/// [`CurrentState`]. The loop terminates when the trait reports
/// [`SimState::Stop`] or when the population cap in
/// [`Options::max_cells`] is reached.
///
/// # Type parameters
///
/// - `NB_REACTIONS` — number of distinct reactions in the system.
/// - `REACTION` — user-defined identifier for one reaction (often
///   `usize` or a small enum).
/// - `P` — the user's process type implementing [`AdvanceStep`].
///
/// # Returns
///
/// The [`StopReason`] that terminated the run.
///
/// # Example
///
/// A 3-type cell population evolving under a 2-reaction process. The
/// process tracks all three types internally but exposes only two
/// aggregated counts to the simulator's [`CurrentState`].
///
/// ```rust
/// # use rand::SeedableRng;
/// # use rand::rngs::StdRng;
/// # use rand_core::Rng;
/// # use sosa::CurrentState;
/// # use sosa::AdvanceStep;
/// # use sosa::NextReaction;
/// struct HealthyCells {
///     // a population of 3 types of cells
///     population: [u64; 3],
/// }
/// impl AdvanceStep<2> for HealthyCells {
///     type Reaction = usize;
///     fn advance_step(&mut self, reaction: NextReaction<Self::Reaction>, _rng: &mut impl Rng) {
///         // a new cell is born!
///         self.population[reaction.event] += 1;
///     }
///     fn update_state(&self, state: &mut CurrentState<2>) {
///         // provide some flexibility: `state` and the `population` don't need to be
///         // the same. Here for example the state has only 2 types
///         state.population = [self.population[0], self.population[1] + self.population[2]];
///     }
/// }
///
/// # let seed = 26;
/// # let mut rng = StdRng::seed_from_u64(seed);
/// let population = [0, 0, 0];
/// let mut process = HealthyCells { population };
/// // type of the cell that will proliferate
/// let type_proliferating = 1;
/// let mut state = CurrentState { population: [population[0], population[1]] };
/// let next_reaction = NextReaction {event: type_proliferating, time: 0.2};
/// let mut expected_population = population;
/// expected_population[type_proliferating] = population[type_proliferating] + 1;
///
/// // a new cell of type 1 is born
/// process.advance_step(next_reaction, &mut rng);
/// assert_eq!(process.population, expected_population);
/// // but the state hasnt changed yet
/// assert_eq!(state.population, population[..2]);
///
/// // update the state now
/// process.update_state(&mut state);
/// assert_eq!(state.population, expected_population[..2]);
/// ```
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
    let mut time = 0f32;
    loop {
        if state.population.iter().sum::<u64>() >= options.max_cells {
            return StopReason::MaxIndividualsReached;
        }

        let (sim_state, reaction) = bd_process.next_reaction(
            state,
            rates,
            possible_reactions,
            IterTime {
                iter: options.max_iter_time.iter - 1,
                time: options.max_iter_time.time,
            },
            IterTime { iter, time },
            rng,
        );

        debug!("time: {time} iter: {iter} and reaction: {reaction:#?}");
        trace!("State: {state:#?}");

        match sim_state {
            SimState::Continue => {
                // unwrap is safe since SimState::Continue returns always
                // something (i.e. not None).
                let reaction = reaction.unwrap();
                let reaction_time = reaction.time;

                // update the process according to the reaction
                bd_process.advance_step(reaction, rng);

                // update the state according to the process
                bd_process.update_state(state);
                iter += 1;
                time += reaction_time;
            }
            SimState::Stop(reason) => return reason,
        }
    }
}

/// The simulator's view of the population — the slice of state shared
/// between [`simulate`] and the user's [`AdvanceStep`] implementation.
///
/// Updated on each step by [`AdvanceStep::update_state`].
#[derive(Debug, Clone)]
pub struct CurrentState<const NB_REACTIONS: usize> {
    /// Per-reaction population counts. Index `i` holds the population for
    /// reaction `i` in `possible_reactions`.
    pub population: [NbIndividuals; NB_REACTIONS],
}

/// User-supplied hook describing how the process evolves on each SSA step.
///
/// [`simulate`] owns the *when* (sampling reaction times, checking stop
/// conditions); implementors own the *what* (mutating their per-individual
/// state when a reaction takes place). The default
/// [`next_reaction`](AdvanceStep::next_reaction) implements the SSA
/// Monte-Carlo sampling — most users only need to implement
/// [`advance_step`](AdvanceStep::advance_step) and
/// [`update_state`](AdvanceStep::update_state).
///
/// Per-step call order: `next_reaction` → `advance_step` → `update_state`.
pub trait AdvanceStep<const NB_REACTIONS: usize> {
    /// Identifier of one of the `NB_REACTIONS` reactions, often `usize` or
    /// a small `enum`.
    type Reaction: std::fmt::Debug + Copy;

    /// Find the next reaction in the system using the
    /// [Monte-Carlo generating method](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm).
    ///
    /// # Returns
    ///
    /// `(SimState::Continue, Some(_))` with the sampled reaction when the
    /// run should advance, or `(SimState::Stop(_), None)` with the reason
    /// when a stopping condition is met (no individuals left, max
    /// iterations, max time).
    ///
    /// # Panics
    ///
    /// If the rate computation produces only non-normal waiting times
    /// (e.g. all rates zero or infinite with a non-empty population).
    fn next_reaction(
        &self,
        state: &CurrentState<NB_REACTIONS>,
        rates: &ReactionRates<{ NB_REACTIONS }>,
        possible_reactions: &[Self::Reaction; NB_REACTIONS],
        max_iter_and_time: IterTime,
        iter_and_time: IterTime,
        rng: &mut impl Rng,
    ) -> (SimState, Option<NextReaction<Self::Reaction>>) {
        // StopIteration appears when there are no cells anymore (due to
        // cell death), when the iteration has reached the max number of
        // iterations nb_iter >= self.max_iter or maximal number of cells
        // i.e. when the iteration has generated a tumor of max_cells size
        if state.population.iter().sum::<u64>() == 0u64 {
            return (SimState::Stop(StopReason::NoIndividualsLeft), None);
        };
        if iter_and_time.iter >= max_iter_and_time.iter {
            return (SimState::Stop(StopReason::MaxItersReached), None);
        };
        if iter_and_time.time >= max_iter_and_time.time {
            return (SimState::Stop(StopReason::MaxTimeReached), None);
        };

        let mut selected_event = 0_usize;
        let times = rates.compute_times_events(&state.population, rng).unwrap();
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

    /// Apply the sampled `reaction` to the process — mutate per-individual
    /// state to reflect one SSA step.
    ///
    /// Called by [`simulate`] immediately after
    /// [`next_reaction`](AdvanceStep::next_reaction).
    fn advance_step(&mut self, reaction: NextReaction<Self::Reaction>, rng: &mut impl Rng);

    /// Sync the simulator's [`CurrentState`] from the process after
    /// [`advance_step`](AdvanceStep::advance_step) has run.
    ///
    /// The process' internal representation can be richer than `state`;
    /// this method projects it down to the per-reaction counts that
    /// [`simulate`] consumes.
    fn update_state(&self, state: &mut CurrentState<NB_REACTIONS>);
}

/// Errors produced when computing reaction times via
/// [`ReactionRates::compute_times_events`] is incompatible with the current
/// state.
#[derive(Error, Debug, PartialEq)]
enum ReactionRatesError {
    /// The total population is zero — no reactions can take place.
    #[error("no individuals left")]
    NoIndividualsLeft,
    /// All computed waiting times were zero or infinite, typically due to
    /// rates that are zero or infinite.
    #[error("the computed times are 0 or inf, probably 0 or inf rates")]
    ComputeTimesAllInfinite,
}

/// Per-reaction rates: the average number of occurrences of each reaction
/// per unit of simulated time, per individual.
///
/// Index `i` corresponds to reaction `i` in the `possible_reactions` array
/// passed to [`simulate`].
#[derive(Debug, Clone)]
pub struct ReactionRates<const N: usize>(
    /// Rates for the `N` reactions, in the same order as
    /// `possible_reactions`.
    pub [f32; N],
);

impl<const N: usize> ReactionRates<N> {
    /// Compute the Gillespie waiting time for all reactions.
    ///
    /// The Gillespie time is
    /// [defined](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm)
    /// as `-ln(1 - r) / (population[i] * rates[i])` for `i` in `0..N`,
    /// where `r` is a uniform random number and `rates` is `self.0`.
    ///
    /// # Returns
    ///
    /// - the array of computed times on success,
    /// - [`ReactionRatesError::NoIndividualsLeft`] if every population is
    ///   zero,
    /// - [`ReactionRatesError::ComputeTimesAllInfinite`] if all computed
    ///   times are zero or infinite.
    fn compute_times_events(
        &self,
        population: &[NbIndividuals; N],
        rng: &mut impl Rng,
    ) -> Result<[f32; N], ReactionRatesError> {
        if population.iter().all(|&pop| pop == 0) {
            return Err(ReactionRatesError::NoIndividualsLeft);
        }
        let mut times = self.0;

        for i in 0..N {
            times[i] = exprand(times[i] * population[i] as f32, rng);
        }
        if times.iter().any(|time| time.is_normal()) {
            return Ok(times);
        }
        Err(ReactionRatesError::ComputeTimesAllInfinite)
    }
}

/// Number of mantissa-aligned bits used when sampling a uniform f32 in the
/// open interval (0, 1) — same construction as `rand_distr::Open01`.
const OPEN01_PRECISION_BITS: u32 = 24;
/// Inverse of `2^OPEN01_PRECISION_BITS`, the step between adjacent samples.
const OPEN01_SCALE: f32 = 1.0 / (1u32 << OPEN01_PRECISION_BITS) as f32;

/// Sample an exponential waiting time with rate `lambda`, as used by the
/// Gillespie SSA Monte-Carlo step.
///
/// # Returns
///
/// - `0` if `lambda` is infinity,
/// - a positive exponential waiting time if `lambda` is
///   [`f32::is_normal`],
/// - infinity otherwise.
///
/// ```
/// use rand::{rngs::StdRng, SeedableRng};
/// # use sosa::exprand;
///
/// let mut rng = StdRng::seed_from_u64(1u64);
///
/// let lambda_gr_than_zero = 0.1_f32;
/// assert!(exprand(lambda_gr_than_zero, &mut rng).is_sign_positive());
///
/// let lambda_zero = 0_f32;
/// assert!(exprand(lambda_zero, &mut rng).is_infinite());
///
/// let lambda_inf = f32::INFINITY;
/// assert!((exprand(lambda_inf, &mut rng) - 0.).abs() < f32::EPSILON);
/// ```
///
/// # Panics
///
/// When `lambda` is negative:
///
/// ```should_panic
/// use rand::{rngs::StdRng, SeedableRng};
/// # use sosa::exprand;
///
/// let mut rng = StdRng::seed_from_u64(1u64);
///
/// let lambda_neg = -0.1_f32;
/// exprand(lambda_neg, &mut rng);
/// ```
pub fn exprand(lambda: f32, rng: &mut impl Rng) -> f32 {
    assert!(!lambda.is_sign_negative());
    if lambda.is_normal() {
        // Uniform f32 in the open interval (0, 1): 24 bits of entropy with
        // a half-step offset, so values are strictly > 0 and strictly < 1.
        // >> throws away the lowest 8 bits.
        let bits = rng.next_u32() >> (32 - OPEN01_PRECISION_BITS);
        // add 0.5 to avoid -inf when bits == 0
        let val = OPEN01_SCALE * (bits as f32 + 0.5);
        return -val.ln() / lambda;
    } else if lambda.is_infinite() {
        return 0.;
    }
    f32::INFINITY
}

/// Append a slice of `Display` values to a file as comma-separated values.
///
/// Each element is formatted with `{:.4}` precision (relevant for floats;
/// integer-like types print their normal representation). When `data` is
/// empty, writes `NaN,` instead. The file is opened in append mode and
/// created if missing; parent directories are created if missing.
///
/// # Errors
///
/// Returns an error if opening or writing to the file fails. Panics if the
/// parent directory cannot be created.
pub fn write2file<T: std::fmt::Display>(
    data: &[T],
    path: &Path,
    header: Option<&str>,
    endline: bool,
) -> anyhow::Result<()> {
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
            writeln!(buffer, "{h}")?;
        }

        for ele in data.iter() {
            write!(buffer, "{ele:.4},")?;
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
    use rand::{SeedableRng, rngs::StdRng};
    use std::num::{NonZeroU8, NonZeroU16};

    #[quickcheck]
    fn exprand_same_seed_test(lambda: u8, seed: u64) -> bool {
        let mut rng = StdRng::seed_from_u64(seed);
        let lambda = lambda as f32 / 10.;
        if lambda.is_normal() {
            let exp1 = exprand(lambda, &mut rng);
            let mut rng = StdRng::seed_from_u64(seed);
            let exp2 = exprand(lambda, &mut rng);
            return (exp1 - exp2).abs() < f32::EPSILON;
        }
        exprand(lambda, &mut rng).is_infinite()
    }

    #[test]
    #[should_panic]
    fn exprand_test_neg_lambda() {
        let mut rng = StdRng::seed_from_u64(1u64);
        let lambda = -0_f32;
        exprand(lambda, &mut rng);
    }

    #[test]
    fn exprand_test() {
        let mut rng = StdRng::seed_from_u64(1u64);
        let lambda = 0_f32;
        let first = exprand(lambda, &mut rng);
        assert!(first.is_infinite());
        let lambda = f32::INFINITY;
        let first = exprand(lambda, &mut rng);
        assert!((0f32 - first).abs() < f32::EPSILON);
        let lambda = 1_f32;
        let first = exprand(lambda, &mut rng);
        assert!(first.is_sign_positive());
    }

    #[test]
    fn exprand_rate_one_returns_1_test() {
        let mut rng = StdRng::seed_from_u64(1u64);
        let lambda_one = 1f32;
        let samples = 1000000;
        let sum: f32 = (0..)
            .map(|_| exprand(lambda_one, &mut rng))
            .take(samples)
            .sum();
        let mean = sum / (samples as f32);
        // should be one but too much variation
        assert!((mean - 1.).abs() < 0.001)
    }

    #[quickcheck]
    fn exprand_test_is_positive(lambda: NonZeroU8, seed: u64) -> bool {
        let mut rng = StdRng::seed_from_u64(seed);
        exprand(lambda.get() as f32 / 10., &mut rng).is_sign_positive()
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
        let mut rng = StdRng::seed_from_u64(seed);
        let population = [0, pop.get() as NbIndividuals, 0, 0];
        let mut expected_population = population;
        expected_population[1] = population[1] + 1;
        let mut state = CurrentState { population };
        let rates = &ReactionRates([1., 1., 1., 1.]);
        let possible_reactions = &[0usize, 1usize, 2usize, 3usize];
        let mut process = TestNextReaction { population };
        let (sim_state, next_reaction) = process.next_reaction(
            &state,
            rates,
            possible_reactions,
            IterTime {
                iter: 10,
                time: 10.,
            },
            IterTime { iter: 0, time: 0. },
            &mut rng,
        );
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
        let mut rng = StdRng::seed_from_u64(seed);
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
        let (sim_state, next_reaction) = process.next_reaction(
            &state,
            rates,
            possible_reactions,
            IterTime {
                iter: 10,
                time: 10.,
            },
            IterTime { iter: 0, time: 0. },
            &mut rng,
        );
        process.advance_step(next_reaction.unwrap(), &mut rng);
        assert_eq!(state.population, population);
        process.update_state(&mut state);

        state.population == expected_population
            && sim_state == SimState::Continue
            && process.population == expected_population
    }

    #[test]
    fn reaction_rates() {
        let mut rng = StdRng::seed_from_u64(1u64);

        let rates = ReactionRates([0.1, 0.1]);
        let population = [10, 10];
        assert!(
            rates
                .compute_times_events(&population, &mut rng)
                .unwrap()
                .iter()
                .all(|time| time.is_normal())
        );

        let rates_zeros = ReactionRates([0., 0.]);
        let population = [10, 10];
        assert_eq!(
            rates_zeros.compute_times_events(&population, &mut rng),
            Err(ReactionRatesError::ComputeTimesAllInfinite)
        );

        let rates = ReactionRates([0.1, 0.1]);
        let population_zero = [0, 0];
        assert_eq!(
            rates.compute_times_events(&population_zero, &mut rng),
            Err(ReactionRatesError::NoIndividualsLeft)
        );
    }
}
