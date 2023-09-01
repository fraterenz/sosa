# Changelog
## 3.0.2
- `Options` derives `Debug` and `Clone`

## 3.0.1
### BugFix
- Fix stop condition when max time has reached.

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
