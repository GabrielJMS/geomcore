//! The hull-pruned searches on pairs holding a B-spline: two B-splines as
//! they are, and an elementary shape against a B-spline once converted to
//! its exact rational B-spline (see [`curve_surface`],
//! [`curve_curve_meetings`] and [`curve_curve_seeds`]).
//!
//! A hit lies on the B-spline, so within the box of its control points.
//! An unbounded elementary shape (a line, parabola, hyperbola, plane,
//! cylinder or cone) is converted over the parameters of its points inside
//! that box grown by a margin, which hold every hit; a trimmed line,
//! parabola or hyperbola over the part of its range among them. Inside its
//! pieces, the B-spline of a circle, ellipse, hyperbola or revolved surface
//! does not follow the shape's own parameter, so what the search finds is
//! carried back to it.

use std::borrow::Cow;
use std::f64::consts::{FRAC_PI_2, TAU};

use super::curve_curve::ParamDomain;
use super::curve_curve_hull::{bspline_curve_curve, bspline_minima};
use super::curve_surface_hull::{CurveSurfaceHits, Range, bspline_curve_surface};
use super::results::sort_by_known_parameter;
use super::{Completeness, CurveCurveExtremum, CurveCurveRegion};
use crate::curves::{BSplineCurve3D, Circle3D, ElementaryCurve, ParametricCurve3D, TrimmedCurve3D};
use crate::math::boxtree::Aabb;
use crate::surfaces::{BSplineSurface, ElementarySurface, ParametricSurface};
use crate::{Point3D, Tolerance, Vector3D};

/// The box of the control points of `curve`, which holds the curve.
fn curve_box(curve: &BSplineCurve3D) -> Aabb {
    let empty = [
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    curve.poles().iter().fold(empty, |mut b, p| {
        for (c, x) in [p.x, p.y, p.z].into_iter().enumerate() {
            b[c] = b[c].min(x);
            b[c + 3] = b[c + 3].max(x);
        }
        b
    })
}

/// `b` grown on every side by `tol.confusion` and a sixteenth of its
/// diagonal: a point within `tol.confusion` of a shape inside `b` lies
/// inside, away from its faces.
fn grown(b: &Aabb, tol: Tolerance) -> Aabb {
    let diagonal = (0..3)
        .map(|c| (b[c + 3] - b[c]).powi(2))
        .sum::<f64>()
        .sqrt();
    let margin = tol.confusion + diagonal / 16.0;
    std::array::from_fn(|i| if i < 3 { b[i] - margin } else { b[i] + margin })
}

/// The range of `(p - origin) . direction` over the points `p` of `b`.
fn reach(b: &Aabb, origin: Point3D, direction: Vector3D) -> (f64, f64) {
    let (o, d) = (
        [origin.x, origin.y, origin.z],
        [direction.x, direction.y, direction.z],
    );
    (0..3).fold((0.0, 0.0), |(lo, hi), c| {
        let (x0, x1) = ((b[c] - o[c]) * d[c], (b[c + 3] - o[c]) * d[c]);
        (lo + x0.min(x1), hi + x0.max(x1))
    })
}

/// The range of `t` over which `origin + t direction` lies inside `b`,
/// with `lo > hi` when it misses `b`.
fn clip(b: &Aabb, origin: Point3D, direction: Vector3D) -> (f64, f64) {
    let (o, d) = (
        [origin.x, origin.y, origin.z],
        [direction.x, direction.y, direction.z],
    );
    let missed = (f64::INFINITY, f64::NEG_INFINITY);
    (0..3).fold((f64::NEG_INFINITY, f64::INFINITY), |(lo, hi), c| {
        if d[c] == 0.0 {
            return if (b[c]..=b[c + 3]).contains(&o[c]) {
                (lo, hi)
            } else {
                missed
            };
        }
        let (t0, t1) = ((b[c] - o[c]) / d[c], (b[c + 3] - o[c]) / d[c]);
        (lo.max(t0.min(t1)), hi.min(t0.max(t1)))
    })
}

/// The representative of the angle `angle` nearest `x`.
fn nearest(angle: f64, x: f64) -> f64 {
    let d = angle - x;
    x + d - TAU * (d / TAU).round()
}

/// `x` brought into `range`: wrapped into a periodic one, else clamped.
fn settle(range: Range, x: f64) -> f64 {
    range.admit(x).unwrap_or(x.clamp(range.lo, range.hi))
}

/// A curve as a B-spline over a parameter range: a B-spline, or trimmed
/// B-spline, as it is, or an elementary curve converted.
struct HullCurve<'a> {
    bspline: Cow<'a, BSplineCurve3D>,
    range: Range,
    /// The elementary curve converted, if any.
    source: Option<ElementaryCurve>,
}

impl HullCurve<'_> {
    /// The curve's own parameter at the B-spline's `t`; an angle is taken
    /// at its representative nearest `t`, not wrapped into the range.
    fn carried(&self, t: f64) -> f64 {
        let point = || self.bspline.eval_point(t);
        match self.source {
            None | Some(ElementaryCurve::Line(_) | ElementaryCurve::Parabola(_)) => t,
            Some(ElementaryCurve::Circle(c)) => nearest(c.parameter_of(point()), t),
            Some(ElementaryCurve::Ellipse(e)) => nearest(e.parameter_of(point()), t),
            Some(ElementaryCurve::Hyperbola(h)) => h.parameter_of(point()),
        }
    }

    /// The curve's own parameter at the B-spline's `t`, within the range.
    fn parameter(&self, t: f64) -> f64 {
        match self.source {
            None => t,
            Some(_) => settle(self.range, self.carried(t)),
        }
    }

    fn carried_range(&self, (lo, hi): (f64, f64)) -> (f64, f64) {
        (self.carried(lo), self.carried(hi))
    }
}

/// A curve ready for the searches, or a restricted elementary curve
/// whose range holds no point of the box it was restricted to.
enum Hull<'a> {
    Curve(Box<HullCurve<'a>>),
    Apart,
}

/// A range of parameters of the line, parabola or hyperbola `curve`
/// holding those of its points inside the box `b`; `None` for a circle or
/// ellipse.
fn inside(curve: ElementaryCurve, b: &Aabb) -> Option<(f64, f64)> {
    match curve {
        ElementaryCurve::Line(l) => Some(clip(b, l.origin(), l.direction())),
        ElementaryCurve::Parabola(p) => {
            let frame = p.frame();
            Some(reach(b, frame.origin(), frame.y_direction()))
        }
        ElementaryCurve::Hyperbola(h) => {
            let frame = h.frame();
            let (y0, y1) = reach(b, frame.origin(), frame.y_direction());
            let minor = h.minor_radius();
            Some(((y0 / minor).asinh(), (y1 / minor).asinh()))
        }
        ElementaryCurve::Circle(_) | ElementaryCurve::Ellipse(_) => None,
    }
}

/// `curve` as a [`HullCurve`]: a B-spline over its range, or an
/// elementary curve over its range, a line, parabola or hyperbola
/// restricted to the parameters of its points inside `cover` (see
/// [`inside`]). `None` for any other curve, an unbounded one without
/// `cover`, or a range whose B-spline does not fit in `f64`.
fn hull_curve<'a, C: ParametricCurve3D>(curve: &'a C, cover: Option<&Aabb>) -> Option<Hull<'a>> {
    if let Some(bspline) = curve.as_bspline() {
        return Some(Hull::Curve(Box::new(HullCurve {
            bspline: Cow::Borrowed(bspline),
            range: ParamDomain::of(curve).range(),
            source: None,
        })));
    }
    let source = curve.as_elementary()?;
    let (mut lo, mut hi) = curve.bounds();
    if let Some((r0, r1)) = cover.and_then(|b| inside(source, b)) {
        (lo, hi) = (lo.max(r0), hi.min(r1));
        if lo >= hi {
            return Some(Hull::Apart);
        }
    }
    let periodic = curve.period().is_some();
    let bspline = if periodic {
        source.to_bspline().ok()?
    } else {
        TrimmedCurve3D::new(source, lo, hi)
            .ok()?
            .to_bspline()
            .ok()?
    };
    Some(Hull::Curve(Box::new(HullCurve {
        bspline: Cow::Owned(bspline),
        range: Range { lo, hi, periodic },
        source: Some(source),
    })))
}

/// How the parameter of a B-spline patch along one direction maps to the
/// elementary surface's.
enum Direction {
    /// The same parameter.
    Straight,
    /// The angle of the unit circle's B-spline `arc` over the same range,
    /// built from the same pieces as the patch's.
    Circular {
        circle: Circle3D,
        arc: Box<BSplineCurve3D>,
    },
}

impl Direction {
    fn circular((lo, hi): (f64, f64)) -> Option<Direction> {
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).ok()?;
        let arc = TrimmedCurve3D::new(circle, lo, hi)
            .ok()?
            .to_bspline()
            .ok()?;
        Some(Direction::Circular {
            circle,
            arc: Box::new(arc),
        })
    }

    /// The surface's parameter at the patch's `x`; an angle is taken at
    /// its representative nearest `x`.
    fn carried(&self, x: f64) -> f64 {
        match self {
            Direction::Straight => x,
            Direction::Circular { circle, arc } => {
                nearest(circle.parameter_of(arc.eval_point(x)), x)
            }
        }
    }
}

/// An elementary surface as B-spline patches holding its points inside a
/// box: one patch, or a cone's two nappes when the box holds its apex.
struct HullSurface {
    patches: Vec<BSplineSurface>,
    directions: [Direction; 2],
    /// The surface's parameter ranges the patches cover.
    ranges: [Range; 2],
}

impl HullSurface {
    /// The patches of `surface` holding its points inside `cover`: a
    /// sphere or torus whole, a plane over the parameters of those
    /// points, a cylinder or cone over a whole turn and the heights of
    /// those points. `None` when a patch does not fit in `f64`.
    fn new(surface: ElementarySurface, cover: &Aabb) -> Option<HullSurface> {
        let turn = Range {
            lo: 0.0,
            hi: TAU,
            periodic: true,
        };
        let interval = |(lo, hi): (f64, f64)| Range {
            lo,
            hi,
            periodic: false,
        };
        let (patches, directions, ranges) = match surface {
            ElementarySurface::Plane(p) => {
                let frame = p.frame();
                let u = reach(cover, frame.origin(), frame.x_direction());
                let v = reach(cover, frame.origin(), frame.y_direction());
                (
                    vec![p.to_bspline_patch(u, v).ok()?],
                    [Direction::Straight, Direction::Straight],
                    [interval(u), interval(v)],
                )
            }
            ElementarySurface::Cylinder(c) => {
                let frame = c.frame();
                let v = reach(cover, frame.origin(), frame.z_direction());
                (
                    vec![c.to_bspline_patch((0.0, TAU), v).ok()?],
                    [Direction::circular((0.0, TAU))?, Direction::Straight],
                    [turn, interval(v)],
                )
            }
            ElementarySurface::Cone(c) => {
                let frame = c.frame();
                let (h0, h1) = reach(cover, frame.origin(), frame.z_direction());
                let (sin, cos) = c.semi_angle().sin_cos();
                let (v0, v1) = (h0 / cos, h1 / cos);
                // Each nappe ends on the apex as a collapsed boundary.
                let apex = -c.ref_radius() / sin;
                let nappes = if v0 < apex && apex < v1 {
                    vec![(v0, apex), (apex, v1)]
                } else {
                    vec![(v0, v1)]
                };
                let patches = nappes
                    .into_iter()
                    .map(|v| c.to_bspline_patch((0.0, TAU), v).ok())
                    .collect::<Option<Vec<_>>>()?;
                (
                    patches,
                    [Direction::circular((0.0, TAU))?, Direction::Straight],
                    [turn, interval((v0, v1))],
                )
            }
            ElementarySurface::Sphere(s) => {
                let latitudes = (-FRAC_PI_2, FRAC_PI_2);
                (
                    vec![s.to_bspline_patch((0.0, TAU), latitudes).ok()?],
                    [
                        Direction::circular((0.0, TAU))?,
                        Direction::circular(latitudes)?,
                    ],
                    [turn, interval(latitudes)],
                )
            }
            ElementarySurface::Torus(t) => (
                vec![t.to_bspline_patch((0.0, TAU), (0.0, TAU)).ok()?],
                [
                    Direction::circular((0.0, TAU))?,
                    Direction::circular((0.0, TAU))?,
                ],
                [turn, turn],
            ),
        };
        Some(HullSurface {
            patches,
            directions,
            ranges,
        })
    }

    /// The surface's own parameters at the patches' `(u, v)`, within its
    /// ranges.
    fn params(&self, (u, v): (f64, f64)) -> (f64, f64) {
        (
            settle(self.ranges[0], self.directions[0].carried(u)),
            settle(self.ranges[1], self.directions[1].carried(v)),
        )
    }

    fn carried_range(&self, k: usize, (lo, hi): (f64, f64)) -> (f64, f64) {
        (
            self.directions[k].carried(lo),
            self.directions[k].carried(hi),
        )
    }
}

/// The hits of the hull-pruned search (see [`bspline_curve_surface`]) of
/// `curve` with `surface` when one is a B-spline (or a trimmed one) and
/// the other a B-spline or an elementary shape, with their parameters on
/// the shapes themselves; `None` for any other pair, or when the
/// elementary shape's B-spline does not fit in `f64`.
pub(super) fn curve_surface<C, S>(
    curve: &C,
    surface: &S,
    tol: Tolerance,
) -> Option<CurveSurfaceHits>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    let (bounds, periodic) = (curve.bounds(), curve.period().is_some());
    match (curve.as_bspline(), surface.as_bspline()) {
        (Some(bspline), Some(patches)) => Some(bspline_curve_surface(
            bspline, bounds, periodic, patches, tol,
        )),
        (Some(bspline), None) => {
            let cover = grown(&curve_box(bspline), tol);
            let hull = HullSurface::new(surface.as_elementary()?, &cover)?;
            let mut hits = hull
                .patches
                .iter()
                .map(|patch| bspline_curve_surface(bspline, bounds, periodic, patch, tol))
                .reduce(|a, b| a.join(b, tol))?;
            for hit in &mut hits.hits {
                hit.surface_parameters = hit.surface_parameters.map(|uv| hull.params(uv));
            }
            if let Completeness::Incomplete { regions, .. } = &mut hits.completeness {
                for region in regions {
                    region.u_range = hull.carried_range(0, region.u_range);
                    region.v_range = hull.carried_range(1, region.v_range);
                }
            }
            Some(hits)
        }
        (None, Some(patches)) => {
            let cover = grown(patches.patch_boxes().bounds(), tol);
            let Hull::Curve(hull) = hull_curve(curve, Some(&cover))? else {
                return Some(CurveSurfaceHits::complete(Vec::new()));
            };
            let range = (hull.range.lo, hull.range.hi);
            let mut hits =
                bspline_curve_surface(&hull.bspline, range, hull.range.periodic, patches, tol);
            for hit in &mut hits.hits {
                if let Some(parameter) = hit.curve_parameter {
                    let parameter = hull.parameter(parameter);
                    hit.curve_parameter = Some(parameter);
                    hit.point = curve.eval_point(parameter);
                }
            }
            sort_by_known_parameter(&mut hits.hits, |hit| hit.curve_parameter);
            if let Completeness::Incomplete { regions, .. } = &mut hits.completeness {
                for region in regions {
                    region.curve_range = hull.carried_range(region.curve_range);
                }
            }
            Some(hits)
        }
        (None, None) => None,
    }
}

/// `curve`, paired with the B-spline `other`, as a [`Hull`] restricted
/// to its points inside the box of `other`'s control points, grown, which
/// hold its meetings with `other`.
fn restricted<'a, C: ParametricCurve3D>(
    curve: &'a C,
    other: &BSplineCurve3D,
    tol: Tolerance,
) -> Option<Hull<'a>> {
    hull_curve(curve, Some(&grown(&curve_box(other), tol)))
}

/// `a` and `b` as [`Hull`]s when one is a B-spline (or a trimmed one) and
/// the other a B-spline or an elementary curve, the elementary one
/// [`restricted`] if `restrict`.
fn hull_pair<'a, A, B>(
    a: &'a A,
    b: &'a B,
    restrict: bool,
    tol: Tolerance,
) -> Option<(Hull<'a>, Hull<'a>)>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    match (a.as_bspline(), b.as_bspline()) {
        (Some(_), Some(_)) => Some((hull_curve(a, None)?, hull_curve(b, None)?)),
        (Some(sa), None) if restrict => Some((hull_curve(a, None)?, restricted(b, sa, tol)?)),
        (None, Some(sb)) if restrict => Some((restricted(a, sb, tol)?, hull_curve(b, None)?)),
        (Some(_), None) | (None, Some(_)) => Some((hull_curve(a, None)?, hull_curve(b, None)?)),
        (None, None) => None,
    }
}

/// `completeness` with the ranges of its regions carried from the
/// B-splines to the curves `ha` and `hb` were made of.
fn carried(
    completeness: Completeness<CurveCurveRegion>,
    ha: &HullCurve,
    hb: &HullCurve,
) -> Completeness<CurveCurveRegion> {
    match completeness {
        Completeness::Incomplete { reason, regions } => Completeness::Incomplete {
            reason,
            regions: regions
                .into_iter()
                .map(|r| CurveCurveRegion {
                    first_range: ha.carried_range(r.first_range),
                    second_range: hb.carried_range(r.second_range),
                    ..r
                })
                .collect(),
        },
        other => other,
    }
}

/// The meetings of `a` and `b` found by the hull-pruned search (see
/// [`bspline_curve_curve`]) when one is a B-spline (or a trimmed one) and
/// the other a B-spline or an elementary curve, as extrema on the curves
/// themselves, closest first, with their completeness; `None` for any
/// other pair, or when the elementary curve's B-spline does not fit in
/// `f64`.
pub(super) fn curve_curve_meetings<A, B>(
    a: &A,
    b: &B,
    tol: Tolerance,
) -> Option<(Vec<CurveCurveExtremum>, Completeness<CurveCurveRegion>)>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    let (Hull::Curve(ha), Hull::Curve(hb)) = hull_pair(a, b, true, tol)? else {
        return Some((Vec::new(), Completeness::Complete));
    };
    let (found, completeness) =
        bspline_curve_curve(&ha.bspline, ha.range, &hb.bspline, hb.range, tol);
    let mut found: Vec<CurveCurveExtremum> = found
        .into_iter()
        .map(|e| {
            let (s, t) = (
                ha.parameter(e.first_parameter),
                hb.parameter(e.second_parameter),
            );
            CurveCurveExtremum {
                first_parameter: s,
                second_parameter: t,
                distance: a.eval_point(s).distance(b.eval_point(t)),
            }
        })
        .collect();
    found.sort_by(|p, q| p.distance.total_cmp(&q.distance));
    Some((found, carried(completeness, &ha, &hb)))
}

/// Whether the minima search beats the grid on `curve`: a B-spline or a
/// segment does. The loose bounds of the rational pieces of a curved
/// elementary curve slow the search down (a closed one adds a maximum of
/// the distance for every point of the other curve), and a full line
/// reduces exactly to the other curve alone.
fn searched<C: ParametricCurve3D>(curve: &C) -> bool {
    match curve.as_elementary() {
        None => true,
        Some(ElementaryCurve::Line(_)) => curve.bounds().0.is_finite(),
        Some(_) => false,
    }
}

/// Seeds for the local minima of the distance between `a` and `b` from
/// the hull-pruned search (see [`bspline_minima`]) when one is a B-spline
/// (or a trimmed one) and the other a B-spline or a segment, as
/// parameters on the curves themselves, with their completeness; `None`
/// for any other pair (see [`searched`]).
pub(super) fn curve_curve_seeds<A, B>(
    a: &A,
    b: &B,
    tol: Tolerance,
) -> Option<(Vec<[f64; 2]>, Completeness<CurveCurveRegion>)>
where
    A: ParametricCurve3D,
    B: ParametricCurve3D,
{
    if !(searched(a) && searched(b)) {
        return None;
    }
    let (Hull::Curve(ha), Hull::Curve(hb)) = hull_pair(a, b, false, tol)? else {
        return None;
    };
    let (seeds, completeness) = bspline_minima(&ha.bspline, ha.range, &hb.bspline, hb.range, tol);
    let seeds = seeds
        .into_iter()
        .map(|[s, t]| [ha.parameter(s), hb.parameter(t)])
        .collect();
    Some((seeds, carried(completeness, &ha, &hb)))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_4, PI};

    use super::*;
    use crate::curves::{Curve3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D};
    use crate::intersect::curve_curve::{curve_curve_minima, sampled_curve_curve};
    use crate::intersect::curve_surface::sampled_curve_surface;
    use crate::intersect::{
        CurveCurveHit, CurveSurfaceHit, curve_curve_extrema, search_curve_curve,
        search_curve_surface,
    };
    use crate::surfaces::{Cone, Cylinder, Plane, Sphere, Surface, Torus};

    const TOL: Tolerance = Tolerance::DEFAULT;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn p(x: f64, y: f64, z: f64) -> Point3D {
        Point3D::new(x, y, z)
    }

    fn v(x: f64, y: f64, z: f64) -> Vector3D {
        Vector3D::new(x, y, z)
    }

    /// A clamped B-spline of `degree` on uniform knots through `poles`.
    fn bspline(degree: usize, poles: &[[f64; 3]]) -> BSplineCurve3D {
        let poles = poles
            .iter()
            .map(|&[x, y, z]| p(x, y, z))
            .collect::<Vec<_>>();
        let spans = poles.len() - degree;
        let knots = (0..=spans).map(|k| k as f64).collect();
        let mut mults = vec![1; spans + 1];
        let end = u32::try_from(degree).unwrap() + 1;
        (mults[0], mults[spans]) = (end, end);
        BSplineCurve3D::new(degree, poles, knots, mults, false).unwrap()
    }

    fn segment(from: Point3D, to: Point3D) -> BSplineCurve3D {
        bspline(1, &[[from.x, from.y, from.z], [to.x, to.y, to.z]])
    }

    /// A bicubic 6×6 net over `[0, 5]^2` moved by `offset`, with heights
    /// `sin(0.7 i j)`.
    fn wavy(offset: Vector3D) -> BSplineSurface {
        net(offset, |i, j| ((i * j) as f64 * 0.7).sin())
    }

    fn net(offset: Vector3D, height: impl Fn(usize, usize) -> f64) -> BSplineSurface {
        let poles = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| p(i as f64, j as f64, height(i, j)) + offset)
                    .collect()
            })
            .collect();
        let (knots, mults) = (vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4]);
        BSplineSurface::new(
            3,
            3,
            poles,
            knots.clone(),
            mults.clone(),
            knots,
            mults,
            false,
            false,
        )
        .unwrap()
    }

    /// A cubic wandering through `[-3, 3]^2 × [-2, 2]`.
    fn wander() -> BSplineCurve3D {
        bspline(
            3,
            &[
                [-3.0, -1.0, -2.0],
                [-1.0, 2.5, -1.0],
                [0.5, -2.5, 0.5],
                [1.5, 2.0, 1.0],
                [3.0, -0.5, 2.0],
            ],
        )
    }

    /// A cubic in the plane `z = 0` swinging across the origin.
    fn swing() -> BSplineCurve3D {
        bspline(
            3,
            &[
                [-3.0, -0.5, 0.0],
                [-1.0, 2.5, 0.0],
                [1.0, -2.5, 0.0],
                [3.0, 0.5, 0.0],
            ],
        )
    }

    /// A cubic swinging over the plane `z = 0`, never meeting it.
    fn skew() -> BSplineCurve3D {
        bspline(
            3,
            &[
                [-3.0, -1.0, 1.0],
                [-1.0, 2.0, 0.5],
                [1.0, -2.0, 1.5],
                [3.0, 1.0, 1.0],
            ],
        )
    }

    /// Whether `t` is within the bounds of `curve`: the sampled searches
    /// can stray past those of a trimmed curve.
    fn within<C: ParametricCurve3D>(curve: &C, t: f64) -> bool {
        let (lo, hi) = curve.bounds();
        (lo - 1e-9..=hi + 1e-9).contains(&t)
    }

    /// Asserts every hit lies on both shapes, with parameters within the
    /// shapes' bounds, ordered along the curve.
    #[track_caller]
    fn assert_on_both<C: ParametricCurve3D, S: ParametricSurface>(
        hits: &[CurveSurfaceHit],
        curve: &C,
        surface: &S,
    ) {
        let (lo, hi) = curve.bounds();
        let ((u0, u1), (v0, v1)) = (surface.u_bounds(), surface.v_bounds());
        for hit in hits {
            let (u, v) = hit.surface_parameters.unwrap();
            assert!((lo..=hi).contains(&hit.curve_parameter.unwrap()), "{hit:?}");
            assert!((u0..=u1).contains(&u) && (v0..=v1).contains(&v), "{hit:?}");
            assert!(
                hit.point
                    .distance(curve.eval_point(hit.curve_parameter.unwrap()))
                    < 1e-12
            );
            assert!(hit.point.distance(surface.eval_point(u, v)) <= TOL.confusion);
        }
        for pair in hits.windows(2) {
            assert!(
                pair[0].curve_parameter < pair[1].curve_parameter,
                "{hits:#?}"
            );
        }
    }

    /// Asserts the hits are complete, lie on both shapes and are those of
    /// the sampled search, and returns them.
    #[track_caller]
    fn assert_surface_hits<C: ParametricCurve3D, S: ParametricSurface>(
        curve: &C,
        surface: &S,
    ) -> Vec<CurveSurfaceHit> {
        let hits = search_curve_surface(curve, surface, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_on_both(&hits, curve, surface);
        let sampled: Vec<_> = sampled_curve_surface(curve, surface, TOL)
            .into_iter()
            .filter(|h| within(curve, h.curve_parameter.unwrap()))
            .collect();
        assert_eq!(hits.len(), sampled.len(), "{hits:#?}\n{sampled:#?}");
        for want in &sampled {
            assert!(
                hits.iter().any(|h| h.point.distance(want.point) < 1e-6
                    && (h.curve_parameter.unwrap() - want.curve_parameter.unwrap()).abs() < 1e-6),
                "{want:?} missing from {hits:#?}"
            );
        }
        hits
    }

    /// Asserts the hits are complete, lie on both curves and are those of
    /// the sampled search, multiplicities included, and returns them.
    #[track_caller]
    fn assert_curve_hits<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveHit> {
        let hits = search_curve_curve(a, b, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_curve_hits_on_both(&hits, a, b);
        let sampled: Vec<_> = sampled_curve_curve(a, b, TOL)
            .into_iter()
            .filter(|h| {
                within(a, h.first_parameter.unwrap()) && within(b, h.second_parameter.unwrap())
            })
            .collect();
        assert_eq!(hits.len(), sampled.len(), "{hits:#?}\n{sampled:#?}");
        for want in &sampled {
            assert!(
                hits.iter()
                    .any(|h| h.point.distance(want.point) < 1e-6
                        && h.multiplicity == want.multiplicity),
                "{want:?} missing from {hits:#?}"
            );
        }
        hits
    }

    #[track_caller]
    fn assert_curve_hits_on_both<A: ParametricCurve3D, B: ParametricCurve3D>(
        hits: &[CurveCurveHit],
        a: &A,
        b: &B,
    ) {
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for hit in hits {
            assert!((a0..=a1).contains(&hit.first_parameter.unwrap()), "{hit:?}");
            assert!(
                (b0..=b1).contains(&hit.second_parameter.unwrap()),
                "{hit:?}"
            );
            assert!(
                hit.point
                    .distance(a.eval_point(hit.first_parameter.unwrap()))
                    < 1e-12
            );
            assert!(
                hit.point
                    .distance(b.eval_point(hit.second_parameter.unwrap()))
                    <= TOL.confusion
            );
        }
        for pair in hits.windows(2) {
            assert!(
                pair[0].first_parameter < pair[1].first_parameter,
                "{hits:#?}"
            );
        }
    }

    /// Asserts the minima are complete, are local minima of the distance
    /// within the curves' bounds, and include those of the sampled search,
    /// and returns them.
    #[track_caller]
    fn assert_extrema<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveExtremum> {
        let extrema = curve_curve_extrema(a, b, TOL);
        assert_eq!(extrema.completeness(), &Completeness::Complete);
        let found = extrema.into_minima();
        assert!(!found.is_empty());
        let ((a0, a1), (b0, b1)) = (a.bounds(), b.bounds());
        for e in &found {
            let (s, t) = (e.first_parameter, e.second_parameter);
            assert!((a0..=a1).contains(&s) && (b0..=b1).contains(&t), "{e:?}");
            assert!((a.eval_point(s).distance(b.eval_point(t)) - e.distance).abs() < 1e-12);
            for (ds, dt) in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)] {
                for h in [-1e-4, 1e-4] {
                    let s = (s + h * ds).clamp(a0, a1);
                    let t = (t + h * dt).clamp(b0, b1);
                    let d = a.eval_point(s).distance(b.eval_point(t));
                    assert!(d >= e.distance - 1e-12, "{e:?} not a minimum");
                }
            }
        }
        for want in curve_curve_minima(a, b, TOL, false).into_minima() {
            let (pa, pb) = (
                a.eval_point(want.first_parameter),
                b.eval_point(want.second_parameter),
            );
            assert!(
                found
                    .iter()
                    .any(|e| (e.distance - want.distance).abs() < 1e-9
                        && a.eval_point(e.first_parameter).distance(pa) < 1e-6
                        && b.eval_point(e.second_parameter).distance(pb) < 1e-6),
                "{want:?} missing from {found:#?}"
            );
        }
        found
    }

    #[test]
    fn test_elementary_curves_against_bspline_surface() {
        let surface = wavy(Vector3D::ZERO);
        let circle = Circle3D::new(p(2.5, 2.5, 0.0), Vector3D::X, 2.0).unwrap();
        let ellipse =
            Ellipse3D::new(p(2.5, 2.5, 0.2), v(1.0, 1.0, 0.0), Vector3D::Z, 2.0, 1.0).unwrap();
        let vertical = Line3D::new(p(2.0, 2.0, -1.0), Vector3D::Z).unwrap();
        let slanted = Line3D::new(p(0.2, 0.3, -1.0), v(1.0, 0.8, 0.5)).unwrap();
        let parabola = Parabola3D::new(p(2.5, 2.5, -1.0), Vector3D::X, Vector3D::Z, 0.5).unwrap();
        let hyperbola =
            Hyperbola3D::new(p(2.5, 2.5, -2.0), Vector3D::X, Vector3D::Z, 1.0, 1.0).unwrap();
        assert_eq!(assert_surface_hits(&circle, &surface).len(), 2);
        assert_eq!(assert_surface_hits(&ellipse, &surface).len(), 2);
        assert_eq!(assert_surface_hits(&vertical, &surface).len(), 1);
        assert!(!assert_surface_hits(&slanted, &surface).is_empty());
        assert_eq!(assert_surface_hits(&parabola, &surface).len(), 2);
        assert_eq!(assert_surface_hits(&hyperbola, &surface).len(), 2);
        let curve = Curve3D::Circle(circle);
        let hits = assert_surface_hits(&curve, &Surface::BSpline(surface.clone()));
        assert_eq!(
            hits,
            search_curve_surface(&circle, &surface, TOL).into_hits()
        );
    }

    #[test]
    fn test_trimmed_elementary_curves_keep_their_hits_in_range() {
        let surface = wavy(Vector3D::ZERO);
        let circle = Circle3D::new(p(2.5, 2.5, 0.0), Vector3D::X, 2.0).unwrap();
        let whole = assert_surface_hits(&circle, &surface);
        for (lo, hi) in [(0.5, 3.5), (3.0, 6.0), (2.0, 2.5)] {
            let arc = TrimmedCurve3D::new(circle, lo, hi).unwrap();
            let hits = assert_surface_hits(&arc, &surface);
            let inside = whole
                .iter()
                .filter(|h| (lo..=hi).contains(&h.curve_parameter.unwrap()));
            assert_eq!(hits.len(), inside.clone().count());
            for (hit, want) in hits.iter().zip(inside) {
                assert!(
                    (hit.curve_parameter.unwrap() - want.curve_parameter.unwrap()).abs() < 1e-9
                );
            }
        }
        let line = Line3D::new(p(2.0, 2.0, -1.0), Vector3D::Z).unwrap();
        let hit = assert_surface_hits(&line, &surface)[0];
        let trimmed = TrimmedCurve3D::new(line, hit.curve_parameter.unwrap() - 0.5, 10.0).unwrap();
        let hits = assert_surface_hits(&trimmed, &surface);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].curve_parameter.unwrap() - hit.curve_parameter.unwrap()).abs() < 1e-9);
        let missed = TrimmedCurve3D::new(line, hit.curve_parameter.unwrap() + 0.5, 10.0).unwrap();
        assert!(assert_surface_hits(&missed, &surface).is_empty());
        let parabola = Parabola3D::new(p(2.5, 2.5, -1.0), Vector3D::X, Vector3D::Z, 0.5).unwrap();
        let half = TrimmedCurve3D::new(parabola, 0.0, 5.0).unwrap();
        assert_eq!(assert_surface_hits(&half, &surface).len(), 1);
    }

    /// Asserts the hits hold one at `at`, and that they are complete or
    /// incomplete with a region holding it.
    #[track_caller]
    fn assert_contact<C: ParametricCurve3D, S: ParametricSurface>(
        curve: &C,
        surface: &S,
        at: Point3D,
    ) {
        let hits = search_curve_surface(curve, surface, TOL);
        let hit = *hits
            .iter()
            .find(|h| h.point.distance(at) < 1e-6)
            .unwrap_or_else(|| panic!("no hit at {at:?} in {hits:#?}"));
        let (u, v) = hit.surface_parameters.unwrap();
        assert!(surface.eval_point(u, v).distance(at) < 1e-6);
        match hits.completeness() {
            Completeness::Complete => {}
            Completeness::Incomplete { regions, .. } => assert!(
                regions.iter().any(|r| {
                    let (lo, hi) = r.curve_range;
                    (lo - 1e-9..=hi + 1e-9).contains(&hit.curve_parameter.unwrap())
                }),
                "{hit:?} outside {regions:#?}"
            ),
            Completeness::Sampled => panic!("a mixed pair is never sampled"),
        }
    }

    #[test]
    fn test_tangent_contacts_are_found() {
        let flat = net(Vector3D::ZERO, |_, _| 0.0);
        let resting = Circle3D::new(p(2.5, 2.2, 1.0), Vector3D::X, 1.0).unwrap();
        assert_contact(&resting, &flat, p(2.5, 2.2, 0.0));
        let touching = segment(p(0.0, -2.0, 1.0), p(0.0, 2.0, 1.0));
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::X, 1.0).unwrap();
        assert_contact(&touching, &cylinder, p(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_bspline_curve_against_elementary_surfaces() {
        let curve = wander();
        // A torus whose tube the curve runs through at its middle.
        let tangent = curve.eval_derivative(1.0, 1).normalized().unwrap();
        let across = tangent.cross(Vector3D::X).normalized().unwrap();
        let core = curve.eval_point(1.0) - across * 2.0;
        let surfaces = [
            Surface::Plane(Plane::new(p(0.0, 0.0, 0.3), v(0.2, 0.1, 1.0)).unwrap()),
            Surface::Cylinder(Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap()),
            Surface::Cone(Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.5, 1.0).unwrap()),
            Surface::Sphere(Sphere::new(p(0.2, 0.1, 0.0), 2.0).unwrap()),
            Surface::Torus(Torus::new(core, tangent, 2.0, 0.6).unwrap()),
        ];
        for surface in &surfaces {
            let hits = assert_surface_hits(&curve, surface);
            assert!(!hits.is_empty(), "{surface:?}");
            let elementary = surface.as_elementary().unwrap();
            let again = assert_surface_hits(&curve, &elementary);
            assert_eq!(hits, again);
        }
        let circle = Circle3D::new(p(0.3, 0.0, 0.2), v(0.3, 1.0, 0.2), 2.5).unwrap();
        let rational = circle.to_bspline();
        for surface in &surfaces {
            assert_surface_hits(&rational, surface);
        }
    }

    #[test]
    fn test_hits_at_sphere_poles_and_cone_apex() {
        let sphere = Sphere::new(p(1.0, 2.0, 3.0), 2.0).unwrap();
        let through = segment(p(1.0, 2.0, 0.0), p(1.0, 2.0, 6.0));
        let hits = search_curve_surface(&through, &sphere, TOL);
        assert_ne!(hits.completeness(), &Completeness::Sampled);
        let hits = hits.into_hits();
        assert_on_both(&hits, &through, &sphere);
        assert_eq!(hits.len(), 2, "{hits:#?}");
        assert!(hits[0].point.distance(p(1.0, 2.0, 1.0)) < 1e-9);
        assert!(hits[1].point.distance(p(1.0, 2.0, 5.0)) < 1e-9);
        assert!((hits[0].surface_parameters.unwrap().1 + FRAC_PI_2).abs() < 1e-6);
        assert!((hits[1].surface_parameters.unwrap().1 - FRAC_PI_2).abs() < 1e-6);

        // The apex is at height -1; the axis meets the cone there alone.
        let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, FRAC_PI_4, 1.0).unwrap();
        let axis = segment(p(0.0, 0.0, -3.0), p(0.0, 0.0, 1.0));
        let hits = search_curve_surface(&axis, &cone, TOL);
        assert_ne!(hits.completeness(), &Completeness::Sampled);
        let hits = hits.into_hits();
        assert_on_both(&hits, &axis, &cone);
        assert_eq!(hits.len(), 1, "{hits:#?}");
        assert!(hits[0].point.distance(p(0.0, 0.0, -1.0)) < 1e-9);

        // Across both nappes, at heights 0.5 and -2.5.
        let across = segment(p(1.5, 0.0, -4.0), p(1.5, 0.0, 3.0));
        let hits = assert_surface_hits(&across, &cone);
        assert_eq!(hits.len(), 2, "{hits:#?}");
        assert!(hits[0].point.distance(p(1.5, 0.0, -2.5)) < 1e-9);
        assert!(hits[1].point.distance(p(1.5, 0.0, 0.5)) < 1e-9);
        let apex = -1.0 / FRAC_PI_4.sin();
        assert!(
            hits[0].surface_parameters.unwrap().1 < apex
                && apex < hits[1].surface_parameters.unwrap().1
        );
    }

    #[test]
    fn test_unbounded_shapes_hit_far_from_origin() {
        let offset = v(1000.0, -2000.0, 500.0);
        let surface = wavy(offset);
        let target = surface.eval_point(1.3, 1.7);
        let line = Line3D::new(
            Point3D::ORIGIN,
            target - Point3D::ORIGIN + v(0.0, 0.0, 300.0),
        )
        .unwrap();
        let line = Line3D::new(target - line.direction() * 2500.0, line.direction()).unwrap();
        let hits = search_curve_surface(&line, &surface, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_on_both(&hits, &line, &surface);
        assert!(
            hits.iter().any(|h| h.point.distance(target) < 1e-9),
            "{hits:#?}"
        );
        let piece = segment(line.eval_point(2400.0), line.eval_point(2600.0));
        let along = search_curve_surface(&piece, &surface, TOL).into_hits();
        assert_eq!(hits.len(), along.len());
        for (hit, want) in hits.iter().zip(&along) {
            assert!(hit.point.distance(want.point) < 1e-9);
        }

        let far = bspline(
            3,
            &[
                [4990.0, 0.0, 3000.0],
                [5005.0, 2.0, 3001.0],
                [4995.0, -1.0, 3002.0],
                [5010.0, 1.0, 3003.0],
            ],
        );
        let plane = Plane::new(p(5000.0, 0.0, 0.0), Vector3D::X).unwrap();
        let hits = search_curve_surface(&far, &plane, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_on_both(&hits, &far, &plane);
        assert_eq!(hits.len(), 1, "{hits:#?}");
        let cylinder = Cylinder::new(p(4998.0, 0.0, 0.0), Vector3D::Z, 5.0).unwrap();
        let hits = search_curve_surface(&far, &cylinder, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_on_both(&hits, &far, &cylinder);
        assert!(!hits.is_empty());
        assert!(
            hits.iter()
                .all(|h| h.surface_parameters.unwrap().1 > 2999.0)
        );
        let cone = Cone::new(p(5000.0, 0.0, 2000.0), Vector3D::Z, 0.004, 1.0).unwrap();
        let hits = search_curve_surface(&far, &cone, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_on_both(&hits, &far, &cone);
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_restriction_keeps_every_hit() {
        let surface = wavy(Vector3D::ZERO);
        let cover = grown(surface.patch_boxes().bounds(), TOL);
        let mut rng = Lcg(5);
        for _ in 0..40 {
            let (u, w) = (3.0 * rng.next(), 3.0 * rng.next());
            let target = surface.eval_point(u, w);
            let direction = v(rng.next() - 0.5, rng.next() - 0.5, rng.next() + 0.2);
            let reach = 1e4 * (rng.next() - 0.5);
            let line = Line3D::new(Point3D::ORIGIN, direction).unwrap();
            let line = Line3D::new(target - line.direction() * reach, direction).unwrap();
            let Some(Hull::Curve(hull)) = hull_curve(&line, Some(&cover)) else {
                panic!("{line:?} misses the box of {target:?}");
            };
            assert!((hull.range.lo..=hull.range.hi).contains(&reach));
            let hits = search_curve_surface(&line, &surface, TOL);
            assert_eq!(hits.completeness(), &Completeness::Complete);
            assert_on_both(&hits, &line, &surface);
            assert!(
                hits.iter().any(|h| h.point.distance(target) < 1e-7),
                "{target:?} missing from {hits:#?}"
            );
        }
        let beside = Line3D::new(p(7.0, 0.0, 0.0), Vector3D::Z).unwrap();
        assert!(matches!(
            hull_curve(&beside, Some(&cover)),
            Some(Hull::Apart)
        ));
        let hits = search_curve_surface(&beside, &surface, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        assert!(hits.is_empty());

        let curve = wander();
        let cover = grown(&curve_box(&curve), TOL);
        let cone = Cone::new(p(0.0, 0.0, 0.5), v(0.2, 0.1, 1.0), 0.6, 0.8).unwrap();
        let hull = HullSurface::new(ElementarySurface::Cone(cone), &cover).unwrap();
        assert_eq!(hull.patches.len(), 2);
        for t in 0..=40 {
            let at = curve.eval_point(f64::from(t) / 20.0);
            for (k, x) in [at.x, at.y, at.z].into_iter().enumerate() {
                assert!(cover[k] + TOL.confusion < x && x < cover[k + 3] - TOL.confusion);
            }
        }
        let hits = assert_surface_hits(&curve, &cone);
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_elementary_curves_against_bspline_curve() {
        let curve = swing();
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap();
        let ellipse =
            Ellipse3D::new(p(0.2, 0.0, 0.0), Vector3D::Z, v(1.0, 0.2, 0.0), 2.5, 1.0).unwrap();
        let line = Line3D::new(p(-4.0, 0.3, 0.0), Vector3D::X).unwrap();
        let parabola = Parabola3D::new(p(0.0, -1.0, 0.0), Vector3D::Z, Vector3D::Y, 0.5).unwrap();
        let hyperbola =
            Hyperbola3D::new(p(0.0, -2.0, 0.0), Vector3D::Z, Vector3D::Y, 1.0, 1.0).unwrap();
        assert_eq!(assert_curve_hits(&circle, &curve).len(), 2);
        assert_eq!(assert_curve_hits(&curve, &circle).len(), 2);
        assert!(!assert_curve_hits(&ellipse, &curve).is_empty());
        assert_eq!(assert_curve_hits(&line, &curve).len(), 3);
        assert!(!assert_curve_hits(&curve, &parabola).is_empty());
        assert!(!assert_curve_hits(&hyperbola, &curve).is_empty());
        let (a, b) = (Curve3D::Circle(circle), Curve3D::BSpline(curve.clone()));
        let hits = assert_curve_hits(&a, &b);
        assert_eq!(hits, search_curve_curve(&circle, &curve, TOL).into_hits());

        let whole = assert_curve_hits(&circle, &curve);
        for (lo, hi) in [(0.0, PI), (2.0, 5.5)] {
            let arc = TrimmedCurve3D::new(circle, lo, hi).unwrap();
            let hits = assert_curve_hits(&arc, &curve);
            let inside = whole
                .iter()
                .filter(|h| (lo..=hi).contains(&h.first_parameter.unwrap()));
            assert_eq!(hits.len(), inside.clone().count());
            for (hit, want) in hits.iter().zip(inside) {
                assert!(hit.point.distance(want.point) < 1e-9);
            }
        }
        let short = TrimmedCurve3D::new(line, 1.0, 5.0).unwrap();
        assert!(!assert_curve_hits(&short, &curve).is_empty());
    }

    #[test]
    fn test_tangent_contacts_match_sampled_and_stay_unsettled() {
        // `y = 1 + x^2 / 2` touches the unit circle and the line `y = 1`
        // at `(0, 1)`.
        let cup = bspline(2, &[[-1.0, 1.5, 0.0], [0.0, 0.5, 0.0], [1.0, 1.5, 0.0]]);
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let line = Line3D::new(p(-3.0, 1.0, 0.0), Vector3D::X).unwrap();
        let ellipse =
            Ellipse3D::new(p(0.0, -1.0, 0.0), Vector3D::Z, Vector3D::Y, 2.0, 1.0).unwrap();
        assert_tangent(&circle, &cup);
        assert_tangent(&cup, &line);
        assert_tangent(&ellipse, &cup);
    }

    /// Asserts `a` and `b` touch once at `(0, 1, 0)`, as the sampled
    /// search finds, multiplicity included, and that the search leaves
    /// the contact unsettled within a region holding it.
    #[track_caller]
    fn assert_tangent<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B) {
        let hits = search_curve_curve(a, b, TOL);
        let Completeness::Incomplete { reason, regions } = hits.completeness().clone() else {
            panic!("{:?}", hits.completeness());
        };
        assert_eq!(reason, super::super::IncompleteReason::Unseparated);
        let hits = hits.into_hits();
        assert_curve_hits_on_both(&hits, a, b);
        assert_eq!(hits.len(), 1, "{hits:#?}");
        let hit = hits[0];
        assert!(hit.point.distance(p(0.0, 1.0, 0.0)) < TOL.confusion.sqrt());
        let sampled = sampled_curve_curve(a, b, TOL);
        assert_eq!(sampled.len(), 1);
        assert!(sampled[0].point.distance(hit.point) < 1e-6);
        assert_eq!(sampled[0].multiplicity, hit.multiplicity);
        let holds = |(lo, hi): (f64, f64), x: f64| (lo - 1e-9..=hi + 1e-9).contains(&x);
        assert!(
            regions
                .iter()
                .any(|r| holds(r.first_range, hit.first_parameter.unwrap())
                    && holds(r.second_range, hit.second_parameter.unwrap())),
            "{hit:?} outside {regions:#?}"
        );
    }

    #[test]
    fn test_line_meets_bspline_curve_far_from_origin() {
        let curve = swing();
        let line = Line3D::new(p(-1e4, 0.3, 0.0), Vector3D::X).unwrap();
        let hits = assert_curve_hits_far(&line, &curve);
        assert_eq!(hits.len(), 3);
        assert!(hits.iter().all(|h| h.first_parameter.unwrap() > 9990.0));
        let beside = Line3D::new(p(0.0, 0.0, 5.0), Vector3D::X).unwrap();
        let hits = search_curve_curve(&beside, &curve, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        assert!(hits.is_empty());
    }

    /// The hits of `a` and `b`, which must be complete and on both curves.
    #[track_caller]
    fn assert_curve_hits_far<A: ParametricCurve3D, B: ParametricCurve3D>(
        a: &A,
        b: &B,
    ) -> Vec<CurveCurveHit> {
        let hits = search_curve_curve(a, b, TOL);
        assert_eq!(hits.completeness(), &Completeness::Complete);
        let hits = hits.into_hits();
        assert_curve_hits_on_both(&hits, a, b);
        hits
    }

    #[test]
    fn test_extrema_of_segments_and_bspline_curve() {
        let curve = skew();
        let line = Line3D::new(p(0.0, 0.0, -1.0), v(1.0, 0.3, 0.2)).unwrap();
        let far = Line3D::new(line.eval_point(-5e3), line.direction()).unwrap();
        let segment = TrimmedCurve3D::new(line, -3.0, 3.0).unwrap();
        assert_extrema(&curve, &segment);
        let near = assert_extrema(&segment, &curve);
        let shifted = assert_extrema(&TrimmedCurve3D::new(far, 4997.0, 5003.0).unwrap(), &curve);
        assert_eq!(near.len(), shifted.len());
        for (a, b) in near.iter().zip(&shifted) {
            assert!((a.distance - b.distance).abs() < 1e-9);
            assert!((a.first_parameter + 5e3 - b.first_parameter).abs() < 1e-6);
        }
        let short = TrimmedCurve3D::new(line, -2.0, 1.0).unwrap();
        assert_extrema(&curve, &short);
        assert_sampled_extrema(&line, &curve);
        assert_sampled_extrema(&Curve3D::Line(line), &Curve3D::BSpline(curve.clone()));
    }

    /// Asserts the minima of `a` and `b` are the grid's, sampled.
    #[track_caller]
    fn assert_sampled_extrema<A: ParametricCurve3D, B: ParametricCurve3D>(a: &A, b: &B) {
        let extrema = curve_curve_extrema(a, b, TOL);
        assert_eq!(extrema.completeness(), &Completeness::Sampled);
        let grid = curve_curve_minima(a, b, TOL, false);
        assert!(!grid.is_empty());
        assert_eq!(extrema.into_minima(), grid.into_minima());
    }

    #[test]
    fn test_extrema_of_curved_elementary_curves_stay_sampled() {
        let curve = skew();
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap();
        let ellipse =
            Ellipse3D::new(p(0.2, 0.0, 0.0), Vector3D::Z, v(1.0, 0.2, 0.0), 2.5, 1.0).unwrap();
        let parabola = Parabola3D::new(p(0.0, -1.0, 0.0), Vector3D::Z, Vector3D::Y, 0.5).unwrap();
        assert_sampled_extrema(&circle, &curve);
        assert_sampled_extrema(&curve, &circle);
        assert_sampled_extrema(&ellipse, &curve);
        assert_sampled_extrema(&TrimmedCurve3D::new(circle, 1.0, 4.0).unwrap(), &curve);
        assert_sampled_extrema(&TrimmedCurve3D::new(ellipse, 0.0, PI).unwrap(), &curve);
        assert_sampled_extrema(&parabola, &curve);
        assert_sampled_extrema(&TrimmedCurve3D::new(parabola, -3.0, 3.0).unwrap(), &curve);
        let (a, b) = (Curve3D::Ellipse(ellipse), Curve3D::BSpline(curve.clone()));
        assert_sampled_extrema(&a, &b);
        assert_sampled_extrema(&Curve3D::Circle(circle), &curve);
    }
}
