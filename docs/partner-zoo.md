# Partner zoo

NanoCoop evaluates a focal policy against a small public partner zoo.

## Why a partner zoo

Cross-play is the point. A focal policy that only coordinates with itself is not competitive here.

The public partner zoo is intentionally simple enough to be inspectable and strong enough to expose weak conventions.

## Included smoke partners

- `courier`
  - shares private info early
  - fetches dish aggressively
  - tends to serve if the soup is plated

- `potter`
  - prepares the pot as early as possible
  - expects the focal policy to complement, not duplicate

- `handoff`
  - waits for enough evidence, then complements the missing stage
  - more robust but slower

- `noisy`
  - occasionally delays or repeats
  - stress-tests recovery and overfitting to a single clean convention

## Benchmark advice

Strong methods usually improve one or more of:

- partner diversity in the collected data
- adaptation after the partner's first informative action
- robustness to redundant work
- recovery from stochastic failures
- handling of held-out layouts with slightly different timing pressure
