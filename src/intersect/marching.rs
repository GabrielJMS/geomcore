//! Numeric surface-surface intersection by marching: seeds from the
//! exact intersections of iso-curves with the other surface, from Bézier
//! patch pairs on B-splines, or from a parameter grid, traced along the
//! intersection and fitted.

use super::routing::route_curve_surface;
use super::{CurveSurfaceIntersection, pieces, surface_surface};
use crate::curves::{BSplineCurve3D, ElementaryCurve, ParametricCurve3D, PointParametrization};
use crate::math::boxtree::{Aabb, boxes_apart};
use crate::math::{solve_2x2, solve_3x3, solve_4x4};
use crate::surfaces::{BSplineSurface, ElementarySurface, ParametricSurface, Surface};
use crate::{Bound3D, BoundingBox3D, Point3D, Tolerance, Vector3D};

/// Marching surface-surface intersection for any two parametric surfaces.
///
/// When either surface is elementary (and the other elementary or a
/// B-spline), seeds are the points where its iso-curves meet the other
/// surface, found by the routed curve-surface intersection (exact for
/// every elementary pair but a circle against a torus):
/// - cylinder and cone: 24 rulings (iso-`u` lines) over the period;
/// - sphere: 24 iso-`u` great circles over half the period (each is a
///   whole meridian circle);
/// - torus: 24 iso-`u` circles over the period;
/// - plane: 24 lines each way across the other surface's bounding box,
///   grown by 10 %; against a surface without one (a plane, cylinder or
///   cone), its two iso-lines through its origin.
///
/// Both sides seed when both are elementary. A line is intersected whole,
/// so an unbounded direction is searched whole: an intersection far from
/// the parameter origin is found.
///
/// Otherwise seeds come from both parameter grids (unbounded directions
/// sampled over `[-10, 10]`): the samples where the distance to the other
/// surface is a local minimum are converged onto the intersection by
/// Newton's method on `S1(u1, v1) = S2(u2, v2)`, so curves running between
/// grid lines are found too. Each seed traces both ways
/// along the `N1 x N2` tangent; every step is corrected by Newton's method
/// on the same system plus the plane through the predicted point normal to
/// the tangent, and the step length follows the curvature (a chord
/// deviation of about `1000 * confusion`, at most 0.1). A trace ends when
/// it closes a loop, stalls, or reaches a parameter bound, its last point
/// then lying on the bound. Traces fit [`BSplineCurve3D`] interpolants
/// (degree 3, closed loops by repeated end points).
///
/// Two B-spline surfaces (see [`ParametricSurface::as_bspline`]) are
/// seeded from pairs of their Bézier patches instead. The pairs whose
/// control points' boxes overlap are halved until the hulls part, or
/// until both pieces are flat and their normals provably never parallel:
/// then no closed loop of the intersection hides within the pair, and a
/// seed converged from its centre, unless a traced curve already crosses
/// it, finds the curves through it. Pairs whose normals may be parallel
/// (a tangency, or a small loop) are halved down to `1e4 * confusion`
/// across. Loops smaller than the grid spacing are found this way; a
/// search that exhausts its budget (overlapping surfaces) traces the grid
/// seeds too.
///
/// The fits are APPROXIMATIONS (interpolation error, typically ~1e-4 at
/// default tolerance — assert residuals, not containment): exact
/// classification stays with the exact intersection
/// ([`IntersectExact::intersect_exact`](crate::IntersectExact::intersect_exact)). Likewise
/// no tangency classification (grazing pairs may trace partially or not
/// at all), no marching through singular points (poles, apexes), and a
/// fixed number of seeding iso-curves or grid samples (a loop that no
/// seeding iso-curve crosses, or that leaves no distance minimum on either
/// grid, is missed) — v1 heuristics, documented so callers can judge fit.
///
/// # Examples
///
/// ```
/// use geomcore::{Plane, Point3D, Tolerance, Vector3D, marching_intersection};
/// let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let curves = marching_intersection(&xy, &zy, Tolerance::DEFAULT);
/// assert_eq!(curves.len(), 1);
/// ```
pub fn marching_intersection<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<BSplineCurve3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let mut traced = Traced::default();
    if let (Some(sa), Some(sb)) = (a.as_bspline(), b.as_bspline()) {
        let pairs = surface_surface::patch_pairs(sa, sb, tol);
        for leaf in &pairs.leaves {
            if leaf.open && traced.crosses(&leaf.overlap, tol) {
                continue;
            }
            let [ra, rb] = leaf.rects;
            let centres = [
                0.5 * (ra[0] + ra[1]),
                0.5 * (ra[2] + ra[3]),
                0.5 * (rb[0] + rb[1]),
                0.5 * (rb[2] + rb[3]),
            ];
            if let Some(seed) = meet(a, b, centres, None, tol) {
                traced.follow(a, b, seed, tol);
            }
        }
        if pairs.complete {
            return traced.curves;
        }
    }
    match iso_curve_seeds(a, b, tol) {
        Some(seeds) => {
            for seed in seeds {
                if let Some(seed) = meet(a, b, seed, None, tol) {
                    traced.follow(a, b, seed, tol);
                }
            }
        }
        None => trace_grid_seeds(a, b, tol, &mut traced),
    }
    traced.curves
}

/// Iso-curves seeding from one elementary side.
const ISO_CURVE_COUNT: usize = 24;

/// Which parameter an iso-curve holds fixed.
#[derive(Clone, Copy)]
enum IsoDirection {
    /// `u` fixed, the curve running along `v`.
    U,
    /// `v` fixed, the curve running along `u`.
    V,
}

/// A surface the routed curve-surface intersection takes: the one an
/// iso-curve of the other side meets.
#[derive(Clone, Copy)]
enum SeedTarget<'a> {
    Elementary(ElementarySurface),
    BSpline(&'a BSplineSurface),
}

impl<'a> SeedTarget<'a> {
    /// `surface` as a target, or `None` when it is neither elementary nor
    /// a B-spline.
    fn of<S: ParametricSurface>(surface: &'a S) -> Option<SeedTarget<'a>> {
        surface
            .as_elementary()
            .map(SeedTarget::Elementary)
            .or_else(|| surface.as_bspline().map(SeedTarget::BSpline))
    }

    fn intersect(self, curve: &ElementaryCurve, tol: Tolerance) -> CurveSurfaceIntersection {
        match self {
            SeedTarget::Elementary(surface) => route_curve_surface(curve, &surface, tol),
            SeedTarget::BSpline(surface) => route_curve_surface(curve, surface, tol),
        }
    }

    fn parameters_of(self, point: Point3D, tol: Tolerance) -> (f64, f64) {
        match self {
            SeedTarget::Elementary(surface) => surface.parameters_of(point),
            SeedTarget::BSpline(surface) => {
                let projection = surface.project_point(point, tol);
                (projection.u, projection.v)
            }
        }
    }

    fn bounding_box(self) -> Option<BoundingBox3D> {
        match self {
            SeedTarget::Elementary(surface) => surface.bounding_box(),
            SeedTarget::BSpline(surface) => surface.bounding_box(),
        }
    }
}

/// Parameters `[a_u, a_v, b_u, b_v]` of the points where the seeding
/// iso-curves of each elementary side meet the other side (see
/// [`marching_intersection`]), or `None` when neither side is elementary
/// or a side is neither elementary nor a B-spline.
fn iso_curve_seeds<A, B>(a: &A, b: &B, tol: Tolerance) -> Option<Vec<[f64; 4]>>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let (first, second) = (SeedTarget::of(a)?, SeedTarget::of(b)?);
    let mut seeds = Vec::new();
    if let SeedTarget::Elementary(own) = first {
        seeds.extend(family_seeds(own, second, tol));
    }
    if let SeedTarget::Elementary(own) = second {
        seeds.extend(
            family_seeds(own, first, tol)
                .into_iter()
                .map(|x| [x[2], x[3], x[0], x[1]]),
        );
    }
    let any_elementary =
        matches!(first, SeedTarget::Elementary(_)) || matches!(second, SeedTarget::Elementary(_));
    any_elementary.then_some(seeds)
}

/// Parameters `[own_u, own_v, other_u, other_v]` of the points where the
/// seeding iso-curves of `own` meet `other`: every hit, and a point of
/// every stretch of an iso-curve lying on `other`.
fn family_seeds(own: ElementarySurface, other: SeedTarget, tol: Tolerance) -> Vec<[f64; 4]> {
    let surface = Surface::from(own);
    let mut seeds = Vec::new();
    for (direction, value) in seeding_iso_values(own, other) {
        let curve = match direction {
            IsoDirection::U => surface.u_iso(value),
            IsoDirection::V => surface.v_iso(value),
        };
        let Some(curve) = curve.ok().and_then(|curve| curve.as_elementary()) else {
            continue;
        };
        let result = other.intersect(&curve, tol);
        let hits = result.hits().iter().map(|hit| {
            let parameter = hit
                .curve_parameter
                .unwrap_or_else(|| curve.parameter_of(hit.point));
            (parameter, hit.point, hit.surface_parameters)
        });
        let overlaps = result.overlaps().iter().map(|&range| {
            let parameter = point_of_range(range);
            (parameter, curve.eval_point(parameter), None)
        });
        for (parameter, point, other_parameters) in hits.chain(overlaps) {
            let (u, v) = own_parameters(own, direction, value, parameter, point);
            let (other_u, other_v) =
                other_parameters.unwrap_or_else(|| other.parameters_of(point, tol));
            seeds.push([u, v, other_u, other_v]);
        }
    }
    seeds
}

/// The iso-curves `own` seeds from, as `(direction, fixed value)`.
fn seeding_iso_values(own: ElementarySurface, other: SeedTarget) -> Vec<(IsoDirection, f64)> {
    let spread = |lo: f64, hi: f64| {
        (0..ISO_CURVE_COUNT).map(move |i| lo + (hi - lo) * i as f64 / ISO_CURVE_COUNT as f64)
    };
    let tau = std::f64::consts::TAU;
    match own {
        ElementarySurface::Plane(plane) => match other.bounding_box() {
            Some(bounds) => {
                // The other surface's box, in the plane's parameters.
                let (min, max) = (bounds.min(), bounds.max());
                let corners = (0..8).map(|k| {
                    plane.parameters_of(Point3D::new(
                        if k & 1 == 0 { min.x } else { max.x },
                        if k & 2 == 0 { min.y } else { max.y },
                        if k & 4 == 0 { min.z } else { max.z },
                    ))
                });
                let (mut low, mut high) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
                for (u, v) in corners {
                    (low[0], high[0]) = (low[0].min(u), high[0].max(u));
                    (low[1], high[1]) = (low[1].min(v), high[1].max(v));
                }
                // Grown by 10 %, lines at both ends.
                let across = |k: usize| {
                    let margin = 0.05 * (high[k] - low[k]);
                    let (lo, hi) = (low[k] - margin, high[k] + margin);
                    (0..ISO_CURVE_COUNT)
                        .map(move |i| lo + (hi - lo) * i as f64 / (ISO_CURVE_COUNT - 1) as f64)
                };
                across(0)
                    .map(|u| (IsoDirection::U, u))
                    .chain(across(1).map(|v| (IsoDirection::V, v)))
                    .collect()
            }
            // An unbounded other surface: two crossing lines meet any plane
            // that is not parallel; a cylinder or cone seeds from its own
            // rulings.
            None => vec![(IsoDirection::U, 0.0), (IsoDirection::V, 0.0)],
        },
        // A meridian circle is whole: the one at `u + PI` is the same.
        ElementarySurface::Sphere(_) => spread(0.0, 0.5 * tau)
            .map(|u| (IsoDirection::U, u))
            .collect(),
        ElementarySurface::Cylinder(_)
        | ElementarySurface::Cone(_)
        | ElementarySurface::Torus(_) => spread(0.0, tau).map(|u| (IsoDirection::U, u)).collect(),
    }
}

/// Parameters on `own` of the point at `parameter` on its iso-curve at
/// `value`: the iso-curve runs along the free direction with the surface's
/// own parametrization, so they are `(value, parameter)` for an iso-`u`
/// curve and `(parameter, value)` for an iso-`v` one. Where that leaves a
/// bounded parameter's range (the half of a sphere's meridian circle that
/// lies on the meridian at `value + PI`), the closed form recovers them.
fn own_parameters(
    own: ElementarySurface,
    direction: IsoDirection,
    value: f64,
    parameter: f64,
    point: Point3D,
) -> (f64, f64) {
    let (u, v) = match direction {
        IsoDirection::U => (value, parameter),
        IsoDirection::V => (parameter, value),
    };
    let within = |x: f64, (lo, hi): (f64, f64), period: Option<f64>| {
        period.is_some() || (lo..=hi).contains(&x)
    };
    if within(u, own.u_bounds(), own.u_period()) && within(v, own.v_bounds(), own.v_period()) {
        (u, v)
    } else {
        own.parameters_of(point)
    }
}

/// A finite parameter in `range`: its middle, or its finite end.
fn point_of_range((first, last): (f64, f64)) -> f64 {
    match (first.is_finite(), last.is_finite()) {
        (true, true) => 0.5 * (first + last),
        (true, false) => first,
        (false, true) => last,
        (false, false) => 0.0,
    }
}

/// [`marching_intersection`] seeded from the parameter grids alone.
#[cfg(test)]
pub(super) fn grid_marching<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<BSplineCurve3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let mut traced = Traced::default();
    trace_grid_seeds(a, b, tol, &mut traced);
    traced.curves
}

/// Traces from the seeds of both parameter grids (see [`grid_seeds`]).
fn trace_grid_seeds<A, B>(a: &A, b: &B, tol: Tolerance, traced: &mut Traced)
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let mut seeds = grid_seeds(a, b, tol);
    seeds.extend(
        grid_seeds(b, a, tol)
            .into_iter()
            .map(|x| [x[2], x[3], x[0], x[1]]),
    );
    for seed in seeds {
        traced.follow(a, b, seed, tol);
    }
}

/// The curves traced so far, fitted, and their traced polylines: the
/// box around each, the boxes around its runs of `RUN` segments, and its
/// points.
#[derive(Default)]
struct Traced {
    curves: Vec<BSplineCurve3D>,
    traces: Vec<(Aabb, Vec<Aabb>, Vec<Point3D>)>,
}

impl Traced {
    /// Traces the curve through `seed` unless it was traced already.
    fn follow<A, B>(&mut self, a: &A, b: &B, seed: [f64; 4], tol: Tolerance)
    where
        A: ParametricSurface,
        B: ParametricSurface,
    {
        // Seeds near a traced polyline are on its curve (the trace's chord
        // deviation exceeds confusion, so tight containment cannot decide
        // duplication). Near-tangent distinct curves (closer than this)
        // are out of scope.
        let p = a.eval_point(seed[0], seed[1]);
        if self.crosses(&[p.x, p.y, p.z, p.x, p.y, p.z], tol) {
            return;
        }
        if let Some(points) = trace(a, b, seed, tol) {
            let degree = 3.min(points.len() - 1);
            if let Ok(fit) =
                BSplineCurve3D::interpolate(&points, degree, PointParametrization::Centripetal)
            {
                let coords: Vec<[f64; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
                let runs = (1..coords.len())
                    .step_by(RUN)
                    .map(|k| pieces::bounding_box(&coords[k - 1..(k + RUN).min(coords.len())]))
                    .collect();
                self.traces
                    .push((pieces::bounding_box(&coords), runs, points));
                self.curves.push(fit);
            }
        }
    }

    /// Whether a traced polyline passes through the box `b`, grown by the
    /// distance its chords may stray from the curve.
    fn crosses(&self, b: &Aabb, tol: Tolerance) -> bool {
        // Steps keep `chord * turn` within 32 chord deviations, so a chord
        // strays at most a quarter of that from the curve.
        let gap = 8.0 * CHORD_DEVIATION * tol.confusion;
        let grown: Aabb = std::array::from_fn(|c| if c < 3 { b[c] - gap } else { b[c] + gap });
        self.traces.iter().any(|(bounds, runs, points)| {
            !boxes_apart(bounds, &grown, 0.0)
                && runs.iter().enumerate().any(|(r, run)| {
                    let first = r * RUN;
                    !boxes_apart(run, &grown, 0.0)
                        && points[first..(first + RUN + 1).min(points.len())]
                            .windows(2)
                            .any(|w| segment_meets_box(w[0], w[1], &grown))
                })
        })
    }
}

/// Segments of a traced polyline per box in [`Traced`].
const RUN: usize = 16;

/// Whether the segment from `p` to `q` meets the box `b`.
fn segment_meets_box(p: Point3D, q: Point3D, b: &Aabb) -> bool {
    let (from, d) = ([p.x, p.y, p.z], [q.x - p.x, q.y - p.y, q.z - p.z]);
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for c in 0..3 {
        if d[c] == 0.0 {
            if from[c] < b[c] || from[c] > b[c + 3] {
                return false;
            }
            continue;
        }
        let (t0, t1) = ((b[c] - from[c]) / d[c], (b[c + 3] - from[c]) / d[c]);
        lo = lo.max(t0.min(t1));
        hi = hi.min(t0.max(t1));
        if lo > hi {
            return false;
        }
    }
    true
}

/// Parameters `[s_u, s_v, other_u, other_v]` of intersection points
/// converged from the samples of a grid on `s` where the distance to
/// `other` is a local minimum. Unbounded directions sample a documented
/// default window.
fn grid_seeds<S, T>(s: &S, other: &T, tol: Tolerance) -> Vec<[f64; 4]>
where
    S: ParametricSurface,
    T: ParametricSurface,
{
    const N: usize = 24;
    let (u0, u1) = seed_window(s.u_bounds());
    let (v0, v1) = seed_window(s.v_bounds());
    let mut samples: Vec<([f64; 4], f64)> = Vec::with_capacity((N + 1) * (N + 1));
    for j in 0..=N {
        for i in 0..=N {
            let (u, v) = (
                u0 + (u1 - u0) * i as f64 / N as f64,
                v0 + (v1 - v0) * j as f64 / N as f64,
            );
            let q = other.project_point(s.eval_point(u, v), tol);
            samples.push(([u, v, q.u, q.v], q.distance));
        }
    }
    let distance = |i: usize, j: usize| samples[j * (N + 1) + i].1;
    let mut seeds = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            let d = distance(i, j);
            let neighbours = [
                (i > 0).then(|| distance(i - 1, j)),
                (i < N).then(|| distance(i + 1, j)),
                (j > 0).then(|| distance(i, j - 1)),
                (j < N).then(|| distance(i, j + 1)),
            ];
            if neighbours.into_iter().flatten().all(|e| d <= e)
                && let Some(x) = meet(s, other, samples[j * (N + 1) + i].0, None, tol)
            {
                seeds.push(x);
            }
        }
    }
    seeds
}

/// Chord deviation allowed per traced step, in multiples of
/// `tol.confusion`.
const CHORD_DEVIATION: f64 = 1000.0;

/// Trace the intersection curve through `seed` both ways. Returns its
/// points in order (a closed loop ends on its first point), or `None`
/// when fewer than four were traced.
fn trace<A, B>(a: &A, b: &B, seed: [f64; 4], tol: Tolerance) -> Option<Vec<Point3D>>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    // Chord deviation allowed per step, and the longest step.
    let chord_deviation = CHORD_DEVIATION * tol.confusion;
    const H_MAX: f64 = 0.1;
    // Steps each way: at most 1000 units of length at the longest step, so
    // a loop of a few hundred units (two cylinders of radius 50) closes,
    // and an unbounded curve stops.
    const STEPS_PER_DIRECTION: usize = 10_000;
    // Turn always allowed in one step. Beyond it, a step may turn by at
    // most `CURVATURE_GROWTH` times the previous step's curvature times
    // its chord: on a smooth curve turn/chord tends to the curvature as
    // the step is halved, so the step is eventually accepted, while at a
    // crossing of two branches (where the surfaces touch) the turn onto
    // the other branch stays finite, and the trace ends there.
    const FREE_TURN: f64 = 0.05;
    const CURVATURE_GROWTH: f64 = 4.0;
    let start = a.eval_point(seed[0], seed[1]);
    let t0 = tangent(a, b, seed, tol)?;
    let mut halves: [Vec<Point3D>; 2] = [Vec::new(), Vec::new()];
    let mut closed = false;
    for (half, dir) in halves.iter_mut().zip([1.0, -1.0]) {
        if closed {
            break;
        }
        let (mut x, mut p, mut t) = (seed, start, t0 * dir);
        let mut h = (8.0 * chord_deviation).sqrt().min(H_MAX);
        // Curvature of the last accepted step (none before the first).
        let mut last_curvature = 0.0;
        for _ in 0..STEPS_PER_DIRECTION {
            // Predict along the tangent, correct onto both surfaces in the
            // plane normal to it; halve the step until the result stays on
            // this branch with a small chord deviation.
            let mut step = None;
            let mut edge = None;
            while h >= tol.confusion {
                let target = p + t * h;
                let guess = predict(a, b, x, t * h);
                if let Some((pin, bound)) = leaving(a, b, x, guess) {
                    // The step leaves a parameter domain: end on its edge.
                    if let Some(y) = land(a, b, guess, pin, bound, tol) {
                        let q = a.eval_point(y[0], y[1]);
                        if (q - p).dot(t) >= 0.0 && q.distance(p) <= 2.0 * h {
                            edge = Some(q);
                            break;
                        }
                    }
                } else if let Some(y) = meet(a, b, guess, Some((target, t)), tol)
                    && let Some(ty) = tangent(a, b, y, tol).map(|ty| ty * dir)
                {
                    let q = a.eval_point(y[0], y[1]);
                    let chord = q.distance(p);
                    let turn = t.cross(ty).magnitude().atan2(t.dot(ty));
                    let smooth = turn <= FREE_TURN.max(CURVATURE_GROWTH * last_curvature * chord);
                    if ty.dot(t) > 0.0
                        && chord <= 2.0 * h
                        && chord * turn <= 32.0 * chord_deviation
                        && smooth
                    {
                        step = Some((y, q, ty, turn / chord.max(f64::MIN_POSITIVE)));
                        break;
                    }
                }
                h *= 0.5;
            }
            if let Some(q) = edge {
                if q.distance(p) > tol.confusion.max(crate::tol::CONFUSION) {
                    half.push(q);
                }
                break;
            }
            let Some((y, q, ty, curvature)) = step else {
                break;
            };
            // Back at the seed: close the loop on it.
            let along = (start - p).dot(q - p) / (q - p).dot(q - p).max(f64::MIN_POSITIVE);
            if half.len() >= 3
                && (0.0..=1.0).contains(&along)
                && (p + (q - p) * along).distance(start)
                    <= 4.0 * chord_deviation + 10.0 * tol.confusion
            {
                if p.distance(start) < 0.1 * h {
                    half.pop();
                }
                half.push(start);
                closed = true;
                break;
            }
            half.push(q);
            (x, p, t) = (y, q, ty);
            last_curvature = curvature;
            h = (2.0 * h)
                .min(H_MAX)
                .min((8.0 * chord_deviation / curvature.max(f64::MIN_POSITIVE)).sqrt());
        }
    }
    let [forward, backward] = halves;
    let mut points: Vec<Point3D> = backward.into_iter().rev().collect();
    points.push(start);
    points.extend(forward);
    (points.len() >= 4).then_some(points)
}

/// Unit tangent `N1 x N2` of the intersection at parameters `x`, or `None`
/// where the surfaces are tangent or a normal degenerates.
fn tangent<A, B>(a: &A, b: &B, x: [f64; 4], tol: Tolerance) -> Option<Vector3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let (_, [au, av]) = a.eval_d1(x[0], x[1]);
    let (_, [bu, bv]) = b.eval_d1(x[2], x[3]);
    let (na, nb) = (au.cross(av), bu.cross(bv));
    let t = na.cross(nb);
    let scale = na.magnitude() * nb.magnitude();
    if t.magnitude() <= tol.angular * scale.max(f64::MIN_POSITIVE) {
        None
    } else {
        Some(t * (1.0 / t.magnitude()))
    }
}

/// Parameters `x` moved to first order by the displacement `d` on both
/// surfaces (unchanged on a surface whose tangent plane degenerates).
fn predict<A, B>(a: &A, b: &B, x: [f64; 4], d: Vector3D) -> [f64; 4]
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let shift = |s: &dyn ParametricSurface, u: f64, v: f64| {
        let (_, [su, sv]) = s.eval_d1(u, v);
        let m = [[su.dot(su), su.dot(sv)], [su.dot(sv), sv.dot(sv)]];
        solve_2x2(m, [su.dot(d), sv.dot(d)], 1e-12).unwrap_or([0.0, 0.0])
    };
    let [du1, dv1] = shift(a, x[0], x[1]);
    let [du2, dv2] = shift(b, x[2], x[3]);
    [x[0] + du1, x[1] + dv1, x[2] + du2, x[3] + dv2]
}

/// Newton's method for a point on both surfaces from parameters
/// `x = [u1, v1, u2, v2]`: solves `S1(u1, v1) = S2(u2, v2)`, together with
/// `(S1 - origin) . normal = 0` when `plane` is given; without it the
/// minimum-norm step heads for the nearest intersection point. Steps are
/// halved until the residual drops, and non-periodic parameters stay in
/// their bounds. `None` when it does not converge to `confusion`.
fn meet<A, B>(
    a: &A,
    b: &B,
    x: [f64; 4],
    plane: Option<(Point3D, Vector3D)>,
    tol: Tolerance,
) -> Option<[f64; 4]>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let bounded = |x: [f64; 4]| within_bounds(a, b, x);
    let residual = |x: [f64; 4]| {
        let pa = a.eval_point(x[0], x[1]);
        let f = pa - b.eval_point(x[2], x[3]);
        let r = plane.map_or(0.0, |(origin, normal)| (pa - origin).dot(normal));
        (f, r, f.dot(f) + r * r)
    };
    let mut x = bounded(x);
    let (mut f, mut r, mut merit) = residual(x);
    for _ in 0..32 {
        if f.magnitude() <= 0.1 * tol.confusion && r.abs() <= 0.1 * tol.confusion {
            return Some(x);
        }
        let (_, [au, av]) = a.eval_d1(x[0], x[1]);
        let (_, [bu, bv]) = b.eval_d1(x[2], x[3]);
        let cols = [au, av, -bu, -bv];
        let delta = match plane {
            Some((_, normal)) => {
                let m = [
                    cols.map(|c| c.x),
                    cols.map(|c| c.y),
                    cols.map(|c| c.z),
                    [cols[0].dot(normal), cols[1].dot(normal), 0.0, 0.0],
                ];
                solve_4x4(m, [-f.x, -f.y, -f.z, -r], 1e-12)?
            }
            None => {
                // Minimum-norm step `J^T y` with `J J^T y = -f`.
                let rows = [cols.map(|c| c.x), cols.map(|c| c.y), cols.map(|c| c.z)];
                let dot = |p: [f64; 4], q: [f64; 4]| (0..4).map(|k| p[k] * q[k]).sum::<f64>();
                let m = rows.map(|p| rows.map(|q| dot(p, q)));
                let y = solve_3x3(m, [-f.x, -f.y, -f.z], 1e-12)?;
                cols.map(|c| c.dot(Vector3D::new(y[0], y[1], y[2])))
            }
        };
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..12 {
            let trial = bounded([0, 1, 2, 3].map(|k| x[k] + scale * delta[k]));
            let (tf, tr, tm) = residual(trial);
            if tm < merit {
                (x, f, r, merit) = (trial, tf, tr, tm);
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (f.magnitude() <= tol.confusion && r.abs() <= tol.confusion).then_some(x)
}

/// The bounds `(lo, hi)` of parameter `k` of `x = [u1, v1, u2, v2]`, or
/// `None` where it is periodic.
fn param_bounds<A, B>(a: &A, b: &B, k: usize) -> Option<(f64, f64)>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let s: &dyn ParametricSurface = if k < 2 { a } else { b };
    if k.is_multiple_of(2) {
        s.u_period().is_none().then(|| s.u_bounds())
    } else {
        s.v_period().is_none().then(|| s.v_bounds())
    }
}

/// Parameters `x` with each non-periodic one clamped into its bounds.
fn within_bounds<A, B>(a: &A, b: &B, x: [f64; 4]) -> [f64; 4]
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    std::array::from_fn(|k| param_bounds(a, b, k).map_or(x[k], |(lo, hi)| x[k].clamp(lo, hi)))
}

/// The parameter that the straight move from `x` to `y` takes out of its
/// bounds first, and that bound, or `None` when `y` is within them.
fn leaving<A, B>(a: &A, b: &B, x: [f64; 4], y: [f64; 4]) -> Option<(usize, f64)>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let mut first: Option<(usize, f64, f64)> = None;
    for k in 0..4 {
        let Some((lo, hi)) = param_bounds(a, b, k) else {
            continue;
        };
        let bound = if y[k] > hi {
            hi
        } else if y[k] < lo {
            lo
        } else {
            continue;
        };
        let fraction = (bound - x[k]) / (y[k] - x[k]);
        if first.is_none_or(|(_, _, f)| fraction < f) {
            first = Some((k, bound, fraction));
        }
    }
    first.map(|(k, bound, _)| (k, bound))
}

/// Newton's method for a point on both surfaces with parameter `pin` of
/// `x = [u1, v1, u2, v2]` held at `value`, from `x`: solves `S1(u1, v1) =
/// S2(u2, v2)` in the three others, which stay in their bounds. `None`
/// when it does not converge to `confusion`.
fn land<A, B>(a: &A, b: &B, x: [f64; 4], pin: usize, value: f64, tol: Tolerance) -> Option<[f64; 4]>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let free: [usize; 3] = std::array::from_fn(|i| if i < pin { i } else { i + 1 });
    let residual = |x: [f64; 4]| a.eval_point(x[0], x[1]) - b.eval_point(x[2], x[3]);
    let mut x = within_bounds(a, b, x);
    x[pin] = value;
    let mut f = residual(x);
    for _ in 0..32 {
        if f.magnitude() <= 0.1 * tol.confusion {
            return Some(x);
        }
        let (_, [au, av]) = a.eval_d1(x[0], x[1]);
        let (_, [bu, bv]) = b.eval_d1(x[2], x[3]);
        let cols = [au, av, -bu, -bv].map(|c| [c.x, c.y, c.z]);
        let m = std::array::from_fn(|r| free.map(|k| cols[k][r]));
        let delta = solve_3x3(m, [-f.x, -f.y, -f.z], 1e-12)?;
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..12 {
            let mut trial = x;
            for (i, &k) in free.iter().enumerate() {
                trial[k] += scale * delta[i];
            }
            let trial = within_bounds(a, b, trial);
            let tf = residual(trial);
            if tf.square_magnitude() < f.square_magnitude() {
                (x, f) = (trial, tf);
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (f.magnitude() <= tol.confusion).then_some(x)
}

/// Clamp a parameter bound pair to a finite seed window.
fn seed_window((lo, hi): (f64, f64)) -> (f64, f64) {
    const WINDOW: f64 = 10.0;
    (
        if lo.is_finite() { lo } else { -WINDOW },
        if hi.is_finite() { hi } else { WINDOW },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marching_plane_plane_line() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let curves = marching_intersection(&xy, &zy, tol);
        assert_eq!(curves.len(), 1);
        // The traced line runs along Y through the origin.
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = curves[0].eval_point(t);
            assert!(p.x.abs() < 1e-6, "{p:?}");
            assert!(p.z.abs() < 1e-6, "{p:?}");
        }
    }

    #[test]
    fn test_marching_plane_sphere_circle() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let curves = marching_intersection(&plane, &sphere, tol);
        assert_eq!(curves.len(), 1);
        // Approximate fit: residuals within marching scale (not confusion).
        for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let p = curves[0].eval_point(t);
            assert!(plane.project_point(p, tol).distance < 1e-3, "{p:?}");
            assert!(sphere.project_point(p, tol).distance < 1e-3, "{p:?}");
        }
    }

    #[test]
    fn test_marching_loop_is_traced_once() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let curves = marching_intersection(&plane, &sphere, tol);
        let curve = assert_one_curve_on_both(&curves, &plane, &sphere);
        let (t0, t1) = curve.bounds();
        let n = 2000;
        let length: f64 = (0..n)
            .map(|k| {
                let at = |k: usize| curve.eval_point(t0 + (t1 - t0) * k as f64 / n as f64);
                at(k).distance(at(k + 1))
            })
            .sum();
        let circumference = 4.0 * std::f64::consts::PI;
        assert!(
            (length - circumference).abs() < 1e-3 * circumference,
            "{length}"
        );
    }

    /// Asserts one traced curve whose samples lie on both surfaces.
    fn assert_one_curve_on_both<A, B>(curves: &[BSplineCurve3D], a: &A, b: &B) -> BSplineCurve3D
    where
        A: ParametricSurface,
        B: ParametricSurface,
    {
        let tol = Tolerance::DEFAULT;
        assert_eq!(curves.len(), 1);
        let curve = curves[0].clone();
        let (t0, t1) = curve.bounds();
        for k in 0..=40 {
            let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
            assert!(a.project_point(p, tol).distance < 1e-3, "{p:?}");
            assert!(b.project_point(p, tol).distance < 1e-3, "{p:?}");
        }
        curve
    }

    #[test]
    fn test_marching_plane_sphere_off_grid_heights() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        // No grid row of either surface lands on these circles.
        for h in [0.3, 0.8, -0.45, 0.123] {
            let plane = Plane::new(Point3D::new(0.0, 0.0, h), Vector3D::Z).unwrap();
            let curves = marching_intersection(&plane, &sphere, tol);
            let curve = assert_one_curve_on_both(&curves, &plane, &sphere);
            let r = (1.0 - h * h).sqrt();
            for t in [0.0, 0.3, 0.6, 0.9] {
                let p = curve.eval_point(t);
                assert!((p.x.hypot(p.y) - r).abs() < 1e-3, "h = {h}: {p:?}");
            }
        }
    }

    #[test]
    fn test_marching_cylinder_offset_sphere() {
        use crate::{Cylinder, Sphere};
        let tol = Tolerance::DEFAULT;
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let sphere = Sphere::new(Point3D::new(0.8, 0.0, 0.0), 1.78).unwrap();
        let curves = marching_intersection(&cylinder, &sphere, tol);
        assert_one_curve_on_both(&curves, &cylinder, &sphere);
    }

    #[test]
    fn test_marching_shallow_angle_spheres() {
        use crate::Sphere;
        let tol = Tolerance::DEFAULT;
        // Unit spheres whose centers are `d` apart meet at about `d`
        // radians, on the circle z = d/2.
        let a = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        for d in [0.05, 0.0175] {
            let b = Sphere::new(Point3D::new(0.0, 0.0, d), 1.0).unwrap();
            let curves = marching_intersection(&a, &b, tol);
            let curve = assert_one_curve_on_both(&curves, &a, &b);
            let r = (1.0 - d * d / 4.0).sqrt();
            for t in [0.0, 0.3, 0.6, 0.9] {
                let p = curve.eval_point(t);
                assert!((p.z - d / 2.0).abs() < 1e-3, "d = {d}: {p:?}");
                assert!((p.x.hypot(p.y) - r).abs() < 1e-3, "d = {d}: {p:?}");
            }
        }
    }

    #[test]
    fn test_marching_small_loop() {
        // A loop much smaller than the default step: the step follows the
        // curvature.
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.003), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 0.01).unwrap();
        let curves = marching_intersection(&plane, &sphere, tol);
        let curve = assert_one_curve_on_both(&curves, &plane, &sphere);
        let r = (1e-4f64 - 9e-6).sqrt();
        for t in [0.0, 0.3, 0.6, 0.9] {
            let p = curve.eval_point(t);
            assert!((p.x.hypot(p.y) - r).abs() < 1e-4, "{p:?}");
        }
    }

    #[test]
    fn test_marching_open_trace_is_not_closed() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let xy = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let zy = Plane::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let curves = marching_intersection(&xy, &zy, tol);
        assert_eq!(curves.len(), 1);
        let (t0, t1) = curves[0].bounds();
        let gap = curves[0].eval_point(t0).distance(curves[0].eval_point(t1));
        assert!(gap > 10.0, "open line fitted as a loop: end gap {gap}");
    }

    /// Polyline length of a traced curve.
    fn traced_length(curve: &BSplineCurve3D) -> f64 {
        let (t0, t1) = curve.bounds();
        let n = 4000;
        let at = |k: usize| curve.eval_point(t0 + (t1 - t0) * k as f64 / n as f64);
        (0..n).map(|k| at(k).distance(at(k + 1))).sum()
    }

    /// The torus of radii 3 and 1 about z, and the plane through its
    /// center tilted `asin(1/3) + d` about y.
    fn villarceau_pair(d: f64) -> (crate::Torus, crate::Plane) {
        let torus = crate::Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
        let a = (1.0f64 / 3.0).asin() + d;
        let plane =
            crate::Plane::new(Point3D::ORIGIN, Vector3D::new(-a.sin(), 0.0, a.cos())).unwrap();
        (torus, plane)
    }

    #[test]
    fn test_marching_stops_at_a_self_crossing() {
        // Issue #7: at the Villarceau tilt the section is two circles of
        // radius 3 crossing where the plane touches the tube. A trace used
        // to turn onto the other circle there and cover an arc twice.
        let tol = Tolerance::DEFAULT;
        let (torus, plane) = villarceau_pair(0.0);
        let curves = marching_intersection(&torus, &plane, tol);
        assert!(!curves.is_empty());
        for curve in &curves {
            let (t0, t1) = curve.bounds();
            for k in 0..=40 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
                assert!(torus.project_point(p, tol).distance < 1e-3, "{p:?}");
                assert!(plane.project_point(p, tol).distance < 1e-3, "{p:?}");
            }
        }
        let total: f64 = curves.iter().map(traced_length).sum();
        let expected = 4.0 * std::f64::consts::PI * 3.0;
        assert!(
            (total - expected).abs() < 1e-3 * expected,
            "total length {total}, expected {expected}: {:?}",
            curves.iter().map(traced_length).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_marching_near_the_villarceau_tilt_gives_two_loops() {
        let tol = Tolerance::DEFAULT;
        for d in [-1e-3, 1e-3] {
            let (torus, plane) = villarceau_pair(d);
            let curves = marching_intersection(&torus, &plane, tol);
            assert_eq!(curves.len(), 2, "d = {d}");
            for curve in &curves {
                let (t0, t1) = curve.bounds();
                let gap = curve.eval_point(t0).distance(curve.eval_point(t1));
                assert!(gap < 1e-6, "d = {d}: open curve, end gap {gap}");
            }
        }
    }

    #[test]
    fn test_segment_meets_box() {
        let unit = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let meets = |p: [f64; 3], q: [f64; 3]| {
            segment_meets_box(
                Point3D::new(p[0], p[1], p[2]),
                Point3D::new(q[0], q[1], q[2]),
                &unit,
            )
        };
        assert!(meets([-1.0, 0.5, 0.5], [2.0, 0.5, 0.5]));
        assert!(meets([0.2, 0.3, 0.4], [0.2, 0.3, 0.4]));
        assert!(meets([-1.0, -1.0, 0.5], [0.0, 0.0, 0.5]));
        assert!(!meets([-1.0, 0.5, 0.5], [-0.1, 0.5, 0.5]));
        assert!(!meets([-1.0, 0.0, 0.5], [0.0, -1.0, 0.5]));
        assert!(!meets([0.5, 0.5, 2.0], [0.5, 0.6, 2.0]));
    }

    /// Asserts every traced curve is closed and lies on both surfaces.
    fn assert_closed_on_both<A, B>(curves: &[BSplineCurve3D], a: &A, b: &B)
    where
        A: ParametricSurface,
        B: ParametricSurface,
    {
        let tol = Tolerance::DEFAULT;
        for curve in curves {
            let (t0, t1) = curve.bounds();
            let gap = curve.eval_point(t0).distance(curve.eval_point(t1));
            assert!(gap < 1e-6, "open curve, end gap {gap}");
            for k in 0..=40 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
                assert!(a.project_point(p, tol).distance < 1e-3, "{p:?}");
                assert!(b.project_point(p, tol).distance < 1e-3, "{p:?}");
            }
        }
    }

    #[test]
    fn test_marching_finds_skew_cylinders_far_from_the_origin() {
        // The cylinders meet around x = 500, far beyond the grid window of
        // +-10 along the first cylinder's axis, in one loop about 510 long.
        use crate::Cylinder;
        let tol = Tolerance::DEFAULT;
        let along_x = Cylinder::new(Point3D::ORIGIN, Vector3D::X, 50.0).unwrap();
        let along_y = Cylinder::new(Point3D::new(500.0, 0.0, 30.0), Vector3D::Y, 50.0).unwrap();
        // The same second cylinder, its parameters measured from y = 300:
        // neither grid window holds a point of the intersection.
        let along_y_far =
            Cylinder::new(Point3D::new(500.0, 300.0, 30.0), Vector3D::Y, 50.0).unwrap();
        for curves in [
            marching_intersection(&along_x, &along_y, tol),
            marching_intersection(&along_y, &along_x, tol),
            marching_intersection(&along_x, &along_y_far, tol),
        ] {
            assert!((1..=2).contains(&curves.len()), "{} curves", curves.len());
            assert_closed_on_both(&curves, &along_x, &along_y);
            for curve in &curves {
                let (t0, t1) = curve.bounds();
                for k in 0..=20 {
                    let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 20.0);
                    assert!((p.x - 500.0).abs() <= 50.0 + 1e-6, "{p:?}");
                }
            }
        }
    }

    #[test]
    fn test_marching_steinmetz_cylinders_trace_both_ellipses() {
        // Equal radii, perpendicular axes crossing: two ellipses of
        // semi-axes 1 and sqrt(2), crossing where the cylinders touch, at
        // (0, 0, +-1). Traces stop at a crossing (no marching through
        // tangent points), so each ellipse comes as its two halves.
        use crate::Cylinder;
        let tol = Tolerance::DEFAULT;
        let along_x = Cylinder::new(Point3D::ORIGIN, Vector3D::X, 1.0).unwrap();
        let along_y = Cylinder::new(Point3D::ORIGIN, Vector3D::Y, 1.0).unwrap();
        let curves = marching_intersection(&along_x, &along_y, tol);
        assert_eq!(curves.len(), 4);
        for curve in &curves {
            let (t0, t1) = curve.bounds();
            for end in [t0, t1] {
                let p = curve.eval_point(end);
                assert!((p.z.abs() - 1.0).abs() < 1e-6, "{p:?}");
            }
            for k in 0..=40 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
                assert!(along_x.project_point(p, tol).distance < 1e-3, "{p:?}");
                assert!(along_y.project_point(p, tol).distance < 1e-3, "{p:?}");
            }
        }
        // Two ellipse perimeters (Ramanujan's approximation).
        let (a, b) = (2f64.sqrt(), 1.0);
        let h = ((a - b) / (a + b)).powi(2);
        let perimeter =
            std::f64::consts::PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()));
        let total: f64 = curves.iter().map(traced_length).sum();
        assert!(
            (total - 2.0 * perimeter).abs() < 1e-3 * perimeter,
            "{total}"
        );
    }

    #[test]
    fn test_marching_oblique_torus_plane() {
        use crate::{Plane, Torus};
        let tol = Tolerance::DEFAULT;
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
        let tilt: f64 = 0.6;
        let plane = Plane::new(
            Point3D::new(0.5, 0.0, 0.2),
            Vector3D::new(-tilt.sin(), 0.0, tilt.cos()),
        )
        .unwrap();
        let curves = marching_intersection(&torus, &plane, tol);
        assert!(!curves.is_empty());
        assert_closed_on_both(&curves, &torus, &plane);
    }

    #[test]
    fn test_marching_miss_is_empty() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(marching_intersection(&plane, &sphere, tol).is_empty());
    }
}
