# Changelog
## 5.1.0
- migrate to Rust 2024 edition
- bump MSRV to 1.85 (declared via `rust-version` in Cargo.toml)
- refresh CI: replace archived `actions-rs/*` with `dtolnay/rust-toolchain` + direct cargo invocations

## 5.0.0
- replace `rand` and `rand_distr` runtime deps with `rand_core` only; public API now takes `&mut impl rand_core::Rng` so consumers can upgrade `rand` independently of `sosa`
- inline the `Open01` sampler in `exprand` (no more `rand_distr` dependency)

## 4.0.0
- get rid of verbosity, use `log` instead

## 3.0.5
- added example and more doc

## 3.0.4
- replace `ChaChaRng` with `SmallRng`
- update `thiserror` to version 2

## 3.0.3
### BugFix
- Fix stop condition when max individuals is reached

## 3.0.2
- `Options` derives `Debug` and `Clone`

## 3.0.1
### BugFix
- Fix stop condition when max time is reached.

## 3.0.0
- Add option to condition upon time (not only population size) by stopping loop when `max_time` has been reached

## 2.0.1
- increase the required verbosity to print the state of the system

## 2.0.0
## Changes
- remove absorbing state condition
- `exprand` generates exp random number only if `lambda.is_normal()`, else either `0` or `f32::INFINITY`,
- raise error if all the waiting times are infinity in `compute_times_events`
- remove `compute_times_events` from the public API

## 1.0.1
Update link to repo

## 1.0.0
Publish crate on creates.io.
