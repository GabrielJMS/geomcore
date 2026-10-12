# geomcore

A geometry-only kernel: evaluation, transformations, projection, extrema and intersection of curves and surfaces. Topology (faces, edges, solids) belongs to the caller.

## Shapes

**Elementary curve**:
A line, circle, ellipse, parabola or hyperbola: a curve with a closed-form definition.
_Avoid_: analytic curve, conic (a line is not a conic), primitive

**Elementary surface**:
A plane, cylinder, cone, sphere or torus: a surface with a closed-form definition.
_Avoid_: analytic surface, quadric (a torus is not a quadric), primitive

**Conic**:
A circle, ellipse, parabola or hyperbola: an elementary curve other than the line.

**Quadric**:
A plane, sphere, cylinder or cone: an elementary surface whose implicit equation has degree at most two, the plane being the degree-one case.

**Lies on**:
A point lies on a curve or surface when it is within `tol.confusion` of it, tested by `contains`.
_Avoid_: inside (reserved for polygons), containment for anything other than this test

**B-spline**:
A curve or surface defined by poles, knots and optional weights. Every curve or surface is either elementary or a B-spline.

## Precision

**Model size**:
The side of the cube, centred on the origin, that models are assumed to fit in: 1e5 model units. It bounds the model, not its carriers: an infinite line or plane matters only inside it. Within it, `f64` coordinates are exact to far below the confusion tolerance.

**Confusion tolerance**:
The distance below which two points coincide (`tol.confusion`). Every geometric decision is a distance decision against it.
_Avoid_: precision, resolution, epsilon

**Parallel**:
Two directions are parallel for a configuration when turning one onto the other moves the geometry by at most the confusion tolerance: an angle below the confusion tolerance over the geometry's extent, or over the model size for unbounded geometry (`tol.angular_for_extent`).

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

## Discretization

**Discretization**:
Points on a curve, in order of parameter, joined by straight segments. Each point keeps its parameter; each segment keeps its chord deviation and angle deviation.
_Avoid_: polyline, polygon (reserved for a closed 2D outline), tessellation (reserved for surfaces), sampling

**Chord deviation**:
The largest distance from a piece of curve to its chord, the straight segment joining the piece's ends. Each segment of a discretized curve carries a proven upper bound on it, never an estimate.
_Avoid_: sag, deflection, chordal error, tolerance (reserved for `Tolerance`)

**Angle deviation**:
The largest angle between a curve's tangent and the chord, over one segment. Proven like the chord deviation; where the curve's speed reaches zero no bound exists, and the segment reports π.
_Avoid_: angular deflection, turning angle

## Polygons

**Polygon**:
A closed 2D outline of straight segments. It encloses a region.

**Point-in-polygon classification**:
Deciding whether a point is inside the region a closed polygon encloses, outside it, or on its boundary: one of Inside, Outside, OnBoundary. A polygon never offers a `contains` method.
_Avoid_: containment, contains (reserved for a point that lies on a curve or surface), point on polygon
