## SOSA
This crate implements the [stochastic simulation algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm#Algorithm) (aka Gillespie mehtod) with the Monte Carlo method.

This provides the building block for simulating agents evolving over time (see [docs](https://docs.rs/sosa/3.0.5/sosa/index.html)).
The user only needs to define how the simulated system evolves upon a single iteration by implementing the trait [`AdvanceStep`](https://docs.rs/sosa/3.0.5/sosa/trait.AdvanceStep.html).
Once this trait is implemented, a simulation loop can be called with [`simulate`](https://docs.rs/sosa/3.0.5/sosa/fn.simulate.html), see the example at
