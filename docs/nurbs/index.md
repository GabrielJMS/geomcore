# NURBS

B-splines — rational ones are NURBS — are the freeform half of the
kernel: `BSplineCurve3D` and `BSplineSurface`, built from a degree,
control poles, a knot vector and optional weights. Where
[Elementary](../geometry/elementary.md) covers the ten closed forms,
this module covers everything else. The math foundations below are
adapted from
[The mathematics behind free-form geometry](https://avenirgeometry.com/articles/free-form-geometry/);
the API pages are [Curves](curves.md), [Surfaces](surfaces.md) and
[Editing & fitting](editing-fitting.md).

## From Bézier to B-spline

A Bézier curve stores control points instead of polynomial
coefficients, and evaluates them with the de Casteljau algorithm —
repeated linear interpolation, one level per degree.

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/03-de-casteljau-at-two-fifths.svg" alt="De Casteljau evaluation">
<figcaption>De Casteljau at two fifths: the point is found by interpolating the polygon down to one point.</figcaption>
</figure>

The blending behind it is the Bernstein basis — one hump per control
point:

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/05-quadratic-basis-functions.svg" alt="Quadratic basis functions">
<figcaption>The quadratic basis: each function weighs one control point, and they sum to one everywhere.</figcaption>
</figure>

A B-spline generalises this to several spans: piecewise polynomials
joined at knots. Knot multiplicity steers continuity — a simple knot is
smooth, a full-multiplicity knot is a hard corner and a place the curve
can be split:

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/07-knot-multiplicity-and-continuity.svg" alt="Knot multiplicity and continuity">
<figcaption>Knot multiplicity and continuity: repeating a knot sharpens the joint up to a corner.</figcaption>
</figure>

Moving one control point only moves the spans it touches — editing is
local:

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/06-moving-one-control-point.svg" alt="Moving one control point">
<figcaption>Local control: one pole moves its neighbourhood, the rest of the curve keeps its shape bit for bit.</figcaption>
</figure>

The same holds on a surface: a pole of a bicubic patch moves only the
cells it supports — here 3 × 2 of the 4 × 3 spans — and the rest stays flat.

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/13-a-surface-and-one-raised-point.svg" alt="A surface and one raised point">
<figcaption>Raising P₄,₄ moves only u in [¼, 1) by v in [⅓, 1); the rest of the patch keeps its shape.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/nurbs-raised-pole.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="A bicubic B-spline surface with one raised control point and its control net"></iframe>

## Rational curves

Give each pole a weight and the curve is pulled toward the heavy ones.
With the right weights a quadratic reproduces the circle exactly —
this is why the conics of the previous chapter convert losslessly with
`to_bspline`, and why NURBS, not plain B-splines, are the interchange
workhorse.

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/08-weights-make-conics.svg" alt="Weights make conics">
<figcaption>Weights make conics: a rational quadratic is an exact circle arc, not an approximation.</figcaption>
</figure>

Inserting a knot refines the control polygon without changing the
shape — the operation behind refinement, splitting and extraction in
[Editing & fitting](editing-fitting.md):

<figure class="geom-fig">
<img src="../assets/figures/free-form-geometry/11-inserting-a-knot.svg" alt="Inserting a knot">
<figcaption>Inserting a knot: more poles, same curve — the shape is untouched, only its description grows.</figcaption>
</figure>

## In geomcore

- Poles are stored Cartesian, once, with a separate weight per pole
  when rational — no second homogeneous copy (ADR 0008). Evaluation,
  editing and projection all read them in place.
- Every curve or surface is either elementary or a B-spline — there is
  no third representation, and exact conversions bridge the two.
- The running example below is the geomdl tutorial curve as a clamped
  degree-2 B-spline; the viewer draws it sampled (native B-spline
  protocol types are
  [viewer#5](https://github.com/geomcore/geomcore-viewer/issues/5)).

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/nurbs-curve.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Quadratic B-spline with control polygon"></iframe>

## This module

- [Curves](curves.md) — create, evaluate, sample with proven bounds.
- [Surfaces](surfaces.md) — patches, evaluation, fitting grids.
- [Editing & fitting](editing-fitting.md) — knots, degree, splitting,
  interpolation, approximation, exact conversions, measure and bound.
