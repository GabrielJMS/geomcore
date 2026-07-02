# Validation

Numerical geometry code fails quietly: an evaluator can look right on simple
inputs and be wrong in the fourth decimal on a rotated frame, a periodic
knot vector, or a pole-crossing circle. `geomcore`'s answer is **golden-fixture
validation**.

## How it works

- The repository ships JSON fixtures (`tests/fixtures/*.json`) containing
  thousands of expected results — curve and surface evaluations, derivatives,
  parameter inversions, and curve-on-surface parametrizations — over
  deliberately awkward configurations: tilted frames, periodic B-splines,
  rational weights, seam- and pole-crossing geometry.
- The expected values were generated independently of this codebase with
  established, industry-proven CAD tooling, so agreement is meaningful — the
  library is checked against external ground truth, not against itself.
- Rust integration tests replay every fixture and require agreement within
  `1e-7 · max(1, |expected|)` — a relative tolerance at CAD-kernel grade.

## What's covered

- Every public curve and surface type, including rational and periodic
  B-splines (where subtle indexing bugs traditionally hide — one such latent
  bug was caught precisely because a fixture disagreed).
- Every analytic curve-on-surface pair, verified through the defining
  identity `surface(pcurve(t)) == curve(t)` at fixed test parameters.
- On top of the fixtures: unit tests for construction and edge cases,
  doctests on every public item, and a behavioral pytest suite for the
  Python bindings.

## What this buys you

If `geomcore` gives you a number, it has been checked against an independent
implementation to seven decimal places. When you find a case where that's
not true, that's a bug — please
[open an issue](https://github.com/GabrielJMS/geomcore/issues) with the
inputs, and it becomes a fixture.
