# Discretization deviations are proven bounds

Every chord deviation and angle deviation a discretization reports is an upper bound geomcore has proven from the curve's definition: closed form for elementary curves, and for B-splines a bound from the control points of each span's Bézier form (the largest second difference for the chord, the derivative's control points and a lower bound on the speed for the angle). geomcore never returns a deviation it only estimated. Point-in-polygon classification and the pruning of numeric seeds will rely on these bounds: if a segment's widened box or band does not enclose the curve, a point is classified inside instead of outside, or a hit is skipped, and nothing reports it. Where no angle bound exists (the curve's speed reaches zero), the segment reports π rather than a guess.

## Considered options

- **Estimated deviations** (midpoint tests, sampled maxima, a numeric search per segment), as many kernels use for display: rejected. A prototype on eight test curves found the estimates almost always within the requested deviation, but not always, and the proven a priori method was 10–25 times faster per point than every adaptive method, proven or estimated, so an estimated mode would buy no speed.
- **An estimated mode alongside the proven one**: rejected for the same reason, and because two modes would make every consumer check which one it was given.
- **Convex-hull subdivision** (proven, adaptive): kept as a possible later refinement for curves whose speed varies within a span, where it gives fewer points, but not the default. Its point count is only known after running it.
