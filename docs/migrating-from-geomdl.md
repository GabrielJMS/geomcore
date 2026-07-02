# Migrating from NURBS-Python (geomdl)

[NURBS-Python (geomdl)](https://nurbs-python.readthedocs.io/) is a
well-established pure-Python NURBS library. If your workload is
**constructing and evaluating** B-spline/NURBS geometry, `geomcore` covers
the same ground with a compiled core, upfront validation, and bulk
evaluation — this page maps the concepts. If you rely on geomdl's wider
toolkit (see [the gaps](#what-geomcore-does-not-do-yet)), keep geomdl for
those parts — the two libraries coexist happily.

## Concept map

| geomdl | geomcore |
|---|---|
| `crv = BSpline.Curve()` then assign `crv.degree`, `crv.ctrlpts`, `crv.knotvector` | one validated call: `BSplineCurve3D.new(degree, poles, knots, mults, periodic)` |
| `NURBS.Curve()` + `ctrlptsw` (weighted) | `BSplineCurve3D.new_rational(..., weights, ...)` — weights stay separate |
| `crv.evaluate_single(t)` | `curve.eval_point(t)` |
| `crv.evaluate_list(ts)` / `crv.evalpts` | `curve.eval_points(ts)` |
| `crv.derivatives(t, order=n)` | `curve.eval_derivative(t, n)` |
| `BSpline.Surface()` + `ctrlpts2d` | `BSplineSurface.new(u_deg, v_deg, pole_grid, ...)` |
| `surf.evaluate_single((u, v))` | `surface.eval_point(u, v)` |
| knot vector domain (`crv.domain`) | `curve.bounds()` |
| `crv.delta` / `sample_size` | no implicit sampling — pass the exact parameter list to `eval_points` |

## Knot vectors: flat list → knots + multiplicities

geomdl uses flat knot vectors with repeated values; `geomcore` takes unique
knots plus multiplicities (which is also how you avoid floating-point
"equal" knots drifting apart). Converting is mechanical:

```python
import math

def split_knot_vector(knot_vector):
    """geomdl-style flat knot vector -> (unique knots, multiplicities)."""
    knots, mults = [], []
    for k in knot_vector:
        if knots and math.isclose(k, knots[-1]):
            mults[-1] += 1
        else:
            knots.append(k)
            mults.append(1)
    return knots, mults

print(split_knot_vector([0, 0, 0, 0, 1, 2, 3, 3, 3, 3]))
# ([0, 1, 2, 3], [4, 1, 1, 4])
```

## Differences worth knowing

- **Objects are immutable and validated.** There is no
  build-then-`evaluate()` lifecycle: the constructor checks degree/pole/knot
  consistency and raises `ValueError` immediately, and a constructed curve
  can't be mutated into an invalid state.
- **Geometry is 3D** (`Point3` poles). For planar data, set `z = 0.0`; the 2D
  types (`Line2D`, `Circle2D`, `Point2`) appear in surface parameter space.
- **Bulk evaluation is the fast path.** `eval_points` makes one native call
  for the whole batch instead of a Python-loop per point.
- **Beyond NURBS**: geomdl is NURBS-only; `geomcore` also gives you exact
  analytic primitives (circles, ellipses, spheres, tori, cones...), rigid
  transforms, and [curve-on-surface
  parametrization](tutorials/curve-on-surface.md).
- **Validation story**: every numeric path is pinned by independently
  generated golden fixtures at 1e-7 — see [Validation](validation.md).

## What geomcore does *not* do (yet)

geomdl features with no `geomcore` equivalent today: knot insertion and
refinement, curve/surface splitting and Bézier decomposition,
interpolation/approximation (fitting), tessellation, visualization modules,
and exchange formats. Fitting and the knot toolkit are on the
[roadmap](https://github.com/GabrielJMS/geomcore/blob/main/ROADMAP.md); if
one of these gaps is what's blocking you, say so in a GitHub issue — it
helps us order the work.
