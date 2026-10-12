# geomcore

A geometry-only kernel: evaluation, transformations, projection, extrema and intersection of curves and surfaces. Topology (faces, edges, solids) belongs to the caller.

## Shapes

**Elementary curve**:
A line, circle, ellipse, parabola or hyperbola: a curve with a closed-form definition.
_Avoid_: analytic curve, conic (a line is not a conic), primitive

**Elementary surface**:
A plane, cylinder, cone, sphere or torus: a surface with a closed-form definition.
_Avoid_: analytic surface, quadric (a torus is not a quadric), primitive

**B-spline**:
A curve or surface defined by poles, knots and optional weights. Every curve or surface is either elementary or a B-spline.

## Intersection

**Exact intersection**:
An intersection computed in closed form between elementary shapes, whose result is itself made of elementary curves and points.
_Avoid_: analytic intersection

**Numeric intersection**:
An intersection traced or refined iteratively, whose curves are fitted approximations. Never chosen implicitly in place of an exact intersection.
_Avoid_: fallback, approximate intersection

**Not analytic**:
The outcome when an exact intersection has no closed form for the given configuration. The caller decides whether to run a numeric intersection.

**Coincident**:
The outcome when two shapes overlap entirely, so their intersection is a whole shape rather than curves or points.

**Contact point**:
An isolated point where two surfaces meet, either tangentially or at a singular point such as a cone apex. It has a position but no surface parameters, which a singular point does not define.

**Hit**:
A point where a curve meets another curve or a surface, with its parameter on each curve and its multiplicity: 1 for a crossing, 2 or more for a tangency. Surface parameters are not part of a hit; the caller projects the point when it needs them.
_Avoid_: intersection point, contact (reserved for surfaces)
