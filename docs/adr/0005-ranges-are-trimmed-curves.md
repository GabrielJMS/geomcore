# Parameter ranges are given by trimmed curves

A parameter range on a curve is expressed in one way only: a trimmed curve, whose `bounds()` are the range. Algorithms take no `first, last` arguments; they work over `bounds()`, so every generic numeric algorithm (intersection, extrema, projection, discretization) restricts itself to a trimmed curve's range without knowing trimming exists, and the range is validated once, at construction (seam crossings, ranges longer than a period, ranges outside the basis). The curve traits are implemented for references, so a trimmed curve can borrow its basis: a kernel whose edges share one B-spline builds a `TrimmedCurve3D<&BSplineCurve3D>` per edge, a pointer and two numbers, instead of cloning the curve. Operations that need a finite range, such as discretization, are not offered on unbounded curves (lines, parabolas, hyperbolas), so the compiler rejects them until they are trimmed.

## Considered options

- **`first, last` arguments on every range-aware method**: rejected. A symmetric intersection would take two ranges, each method would re-validate them, and it would add a second way to say what a trimmed curve already says.
- **Dropping trimmed curves for range arguments**: rejected for the same reasons, and it would undo the range-restricted numeric algorithms of 0.28 and their Python API.
- **Trimmed curves that own their basis only**: rejected. It forces a kernel to clone a shared basis curve per edge.
