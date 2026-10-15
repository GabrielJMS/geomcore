//! Numeric surface-surface intersection by marching: seeds from the
//! exact intersections of iso-curves with the other surface, from Bézier
//! patch pairs on B-splines, or from a parameter grid, traced along the
//! intersection and fitted.

use super::routing::route_curve_surface;
use super::{
    Completeness, CurveSurfaceIntersection, IncompleteReason, IntersectionCurve,
    SurfaceSurfaceIntersection, SurfaceSurfaceRegion, pieces, surface_surface,
};
use crate::curves::{
    BSplineCurve3D, Curve3D, ElementaryCurve, ParametricCurve3D, PointParametrization,
};
use crate::math::boxtree::{Aabb, boxes_apart};
use crate::math::{solve_2x2, solve_3x3, solve_4x4};
use crate::surfaces::{BSplineSurface, ElementarySurface, ParametricSurface, Surface};
use crate::{Bound3D, BoundingBox3D, Point3D, Tolerance, Vector3D};

/// Marching surface-surface intersection for any two parametric surfaces.
///
/// First, a surface with bounded parameters is checked for lying on the
/// other (see [`lies_on`]): a 7 by 7 grid of its points, every one within
/// confusion of the other surface with the normals parallel there.
/// Marching follows the curves where surfaces cross, and has none to
/// follow on surfaces lying on each other; seeding there would find the
/// whole of each iso-curve lying on the other surface, slowly. So when
/// each lies on the other, the result is coincident, its completeness
/// [`Completeness::Sampled`]; when one lies on the other, which extends
/// beyond it or is unbounded, the result is
/// [`Completeness::Incomplete`] with one
/// [`IncompleteReason::Unseparated`] region covering the one lying on the
/// other. The check stops at the first sample off the other surface, so
/// crossing surfaces pay one or two projections for it.
///
/// Otherwise, when one surface is elementary and the other elementary or a B-spline,
/// seeds are the points where the iso-curves of an elementary side meet
/// the other surface, found by the routed curve-surface intersection
/// (exact against an elementary surface, the Bézier piece search against
/// a B-spline):
/// - cylinder and cone: 24 rulings (iso-`u` lines) over the period, and
///   24 circles (iso-`v`) across the other surface's bounding box grown by
///   10 %, or the circle at `v = 0` against a surface without one;
/// - sphere: 24 iso-`u` great circles over half the period (each is a
///   whole meridian circle);
/// - torus: 24 iso-`u` circles over the period;
/// - plane: 24 lines each way across the other surface's bounding box,
///   grown by 10 %; against a surface without one (a plane, cylinder or
///   cone), its two iso-lines through its origin.
///
/// Circles are left out against a torus, which they meet only through a
/// sampled search. Both sides seed when both are elementary. A B-spline
/// side adds its own parameter grid (below), and so do both sides when
/// both surfaces are bounded (spheres, tori, B-splines), where a small
/// loop may lie between the iso-curves. A line is
/// intersected whole, so an unbounded direction is searched whole: an
/// intersection far from the parameter origin is found. Seeds landing on
/// the same point (a sphere's pole, a cone's apex) are traced from once.
///
/// Otherwise (a torus against a torus, or a surface defined outside the
/// crate) seeds come from both parameter grids (unbounded
/// directions sampled over `[-10, 10]`): the samples where the distance to
/// the other surface is a local minimum are converged onto the
/// intersection by Newton's method on `S1(u1, v1) = S2(u2, v2)`, so curves
/// running between grid lines are found too.
///
/// Where the grid seeds all lie where the surfaces touch rather than
/// cross (see [`grazes`]), the iso-curve seeds are left out: an iso-curve
/// touching a B-spline surface is settled by halving down to the search's
/// smallest pieces, which costs far more than tracing the touch.
///
/// Each seed traces both ways along the `N1 x N2` tangent; every step is
/// corrected by Newton's method on the same system plus the plane through
/// the predicted point normal to the tangent. The step length follows the
/// curvature (a chord deviation of about `1000 * confusion`), with no
/// fixed cap. A step is halved when the intersection at its midpoint,
/// converged onto both surfaces, strays from the chord by more than twice
/// that deviation, or when it crosses more than one knot in a parameter
/// of a B-spline side, so that it never steps over a whole knot span
/// (within a span the control points bound the features). A trace ends
/// when it closes a loop, stalls, reaches a parameter bound, or leaves
/// the cube of side [`Tolerance::MODEL_SIZE`] centred on the origin, its
/// last point then lying on the bound or the cube. Traces fit
/// [`BSplineCurve3D`] interpolants (degree 3, closed loops by repeated end
/// points).
///
/// A seed where the surfaces touch (their normals parallel, or so nearly
/// that moving one surface by confusion makes them touch; see [`grazes`])
/// is traced along the direction in which they keep touching (see
/// [`contact_direction`]), each step corrected by damped least squares
/// (see [`meet_touching`]), and the curve is reported tangent. A touch
/// whose trace stays shorter than a first step is a contact point.
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
/// no marching through singular points (poles, apexes; a touch there is
/// not reported), a coincidence check that sees only whole surfaces (two
/// patches overlapping in part are marched: the seeds where they lie on
/// each other make the result `Incomplete` with an `Unseparated` region
/// around them, but spurious curves may still cross the overlap), and a fixed number of seeding iso-curves or grid samples (a
/// loop that no seeding iso-curve crosses, or that leaves no distance
/// minimum on either grid, is missed) — v1 heuristics, documented so
/// callers can judge fit.
pub(crate) fn marching_intersection<A, B>(
    a: &A,
    b: &B,
    tol: Tolerance,
) -> SurfaceSurfaceIntersection
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    if let Some(coincident) = coincidence(a, b, tol) {
        return coincident;
    }
    let traced = traced_intersection(a, b, tol);
    let completeness = match coincident_region(a, b, &traced.coincident_seeds) {
        Some(region) => Completeness::Incomplete {
            reason: IncompleteReason::Unseparated,
            regions: vec![region],
        },
        None => Completeness::Sampled,
    };
    let curves = traced
        .curves
        .into_iter()
        .map(|(curve, tangent)| IntersectionCurve {
            curve: Curve3D::BSpline(curve),
            tangent,
        })
        .collect();
    SurfaceSurfaceIntersection::numeric(curves, traced.contact_points, completeness)
}

/// Seed grid cells a region around the seeds where the surfaces lie on
/// each other grows by, per parameter (see [`coincident_region`]).
const SEED_GRID_CELLS: f64 = 24.0;

/// The region around `seeds`, where the surfaces lie on each other: the
/// ranges of their parameters, each grown by a cell of a 24-cell grid
/// over its (finite) window and kept within its bounds, and the box
/// around the first surface over its grown ranges. `None` without seeds.
///
/// The seeds sample the stretch where the surfaces lie on each other, as
/// finely as the grids and patch pairs that found them; the growth covers
/// what lies between the outermost seeds and the stretch's end.
fn coincident_region<A, B>(a: &A, b: &B, seeds: &[[f64; 4]]) -> Option<SurfaceSurfaceRegion>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    if seeds.is_empty() {
        return None;
    }
    let range = |k: usize, bounds: (f64, f64)| {
        let low = seeds.iter().map(|x| x[k]).fold(f64::INFINITY, f64::min);
        let high = seeds.iter().map(|x| x[k]).fold(f64::NEG_INFINITY, f64::max);
        let (window_low, window_high) = seed_window(bounds);
        let cell = (window_high - window_low) / SEED_GRID_CELLS;
        ((low - cell).max(bounds.0), (high + cell).min(bounds.1))
    };
    let (first_u_range, first_v_range) = (range(0, a.u_bounds()), range(1, a.v_bounds()));
    let last = (COINCIDENCE_SAMPLES - 1) as f64;
    let at = |(low, high): (f64, f64), i: usize| low + (high - low) * i as f64 / last;
    let bounds = BoundingBox3D::from_points((0..COINCIDENCE_SAMPLES).flat_map(|i| {
        (0..COINCIDENCE_SAMPLES)
            .map(move |j| a.eval_point(at(first_u_range, i), at(first_v_range, j)))
    }))?;
    Some(SurfaceSurfaceRegion {
        first_u_range,
        first_v_range,
        second_u_range: range(2, b.u_bounds()),
        second_v_range: range(3, b.v_bounds()),
        min_corner: bounds.min(),
        max_corner: bounds.max(),
    })
}

/// The curves traced from every seed (see [`marching_intersection`]).
fn traced_intersection<A, B>(a: &A, b: &B, tol: Tolerance) -> Traced
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
        if !pairs.complete {
            trace_grid_seeds(a, b, tol, &mut traced);
        }
        return traced;
    }
    match pair_seeds(a, b, tol) {
        Some(seeds) => {
            for seed in distinct_seeds(a, seeds, tol) {
                traced.follow(a, b, seed, tol);
            }
        }
        None => trace_grid_seeds(a, b, tol, &mut traced),
    }
    traced
}

/// Samples per parameter of the coincidence check (see [`lies_on`]).
const COINCIDENCE_SAMPLES: usize = 7;

/// The result when one surface lies on the other, from samples (see
/// [`lies_on`]), so that marching, which follows crossings, is skipped:
/// - each lies on the other: coincident, with the samples' completeness;
/// - one lies on the other, which extends beyond it or is unbounded:
///   incomplete and unseparated, its region covering the one lying on the
///   other.
///
/// `None` when neither lies on the other.
fn coincidence<A, B>(a: &A, b: &B, tol: Tolerance) -> Option<SurfaceSurfaceIntersection>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let region = if let Some((b_u, b_v)) = lies_on(a, b, tol) {
        if lies_on(b, a, tol).is_some() {
            return Some(SurfaceSurfaceIntersection::numeric_coincident(
                Completeness::Sampled,
            ));
        }
        let (min_corner, max_corner) = surface_corners(a);
        SurfaceSurfaceRegion {
            first_u_range: a.u_bounds(),
            first_v_range: a.v_bounds(),
            second_u_range: b_u,
            second_v_range: b_v,
            min_corner,
            max_corner,
        }
    } else {
        let (a_u, a_v) = lies_on(b, a, tol)?;
        let (min_corner, max_corner) = surface_corners(b);
        SurfaceSurfaceRegion {
            first_u_range: a_u,
            first_v_range: a_v,
            second_u_range: b.u_bounds(),
            second_v_range: b.v_bounds(),
            min_corner,
            max_corner,
        }
    };
    Some(SurfaceSurfaceIntersection::numeric(
        Vec::new(),
        Vec::new(),
        Completeness::Incomplete {
            reason: IncompleteReason::Unseparated,
            regions: vec![region],
        },
    ))
}

/// Whether `s` lies on `other`, from a grid of `COINCIDENCE_SAMPLES` by
/// `COINCIDENCE_SAMPLES` samples over its whole parameter domain, bounds
/// included, and the centres of the grid's cells: every sample projects
/// onto `other` within `tol.confusion`, and the two normals there are
/// parallel within the tilt that distance allows (see below; where
/// neither degenerates, as at a sphere's pole). The ranges of `other`'s
/// parameters at the projections when it does; `None` when it does not,
/// or when a parameter of `s` is unbounded.
///
/// The cell centres sit halfway between grid lines, so a surface that
/// meets `other` only along those lines (as when they fall on knots)
/// still leaves `other` at a centre.
///
/// The check stops at the first sample off `other`, so on crossing
/// surfaces it costs one or two projections. A surface crossing `other`
/// passes through isolated curves, and one touching it does so along
/// curves or at points: neither holds every sample with the normals
/// parallel there.
fn lies_on<S, T>(s: &S, other: &T, tol: Tolerance) -> Option<((f64, f64), (f64, f64))>
where
    S: ParametricSurface,
    T: ParametricSurface,
{
    let (u_bounds, v_bounds) = (s.u_bounds(), s.v_bounds());
    if ![u_bounds.0, u_bounds.1, v_bounds.0, v_bounds.1]
        .iter()
        .all(|x| x.is_finite())
    {
        return None;
    }
    let cells = (COINCIDENCE_SAMPLES - 1) as f64;
    // Grid lines at whole steps, cell centres at half steps.
    let at = |(low, high): (f64, f64), half_steps: usize| {
        low + (high - low) * half_steps as f64 / (2.0 * cells)
    };
    let half_step = |(low, high): (f64, f64)| (high - low) / (2.0 * cells);
    let (u_half_step, v_half_step) = (half_step(u_bounds), half_step(v_bounds));
    let lines = (0..COINCIDENCE_SAMPLES).map(|i| 2 * i);
    let centres = (0..COINCIDENCE_SAMPLES - 1).map(|i| 2 * i + 1);
    let samples = lines
        .clone()
        .flat_map(|i| lines.clone().map(move |j| (i, j)))
        .chain(
            centres
                .clone()
                .flat_map(|i| centres.clone().map(move |j| (i, j))),
        );
    let mut feet = Vec::with_capacity(
        COINCIDENCE_SAMPLES * COINCIDENCE_SAMPLES
            + (COINCIDENCE_SAMPLES - 1) * (COINCIDENCE_SAMPLES - 1),
    );
    for (i, j) in samples {
        let (p, [su, sv]) = s.eval_d1(at(u_bounds, i), at(v_bounds, j));
        let foot = other.project_point(p, tol);
        if foot.distance > tol.confusion {
            return None;
        }
        // Two surfaces within a distance d of each other at samples a
        // length h apart can tilt against each other by about 2d/h there:
        // the gap between them changes by at most 2d over h, plus terms
        // in h times the difference of their curvatures, which copies of
        // one surface share. So a copy off by up to `tol.confusion` (a
        // fit, or a shift) has normals off by up to that bound, which can
        // exceed `tol.angular` by far; h is the distance to the nearest
        // other sample, half a grid step.
        let spacing = (su.magnitude() * u_half_step).min(sv.magnitude() * v_half_step);
        let allowed_tilt = tol.angular.max(2.0 * tol.confusion / spacing);
        let (_, [ou, ov]) = other.eval_d1(foot.u, foot.v);
        if let (Some(n), Some(m)) = (unit_normal(su, sv), unit_normal(ou, ov))
            && n.cross(m).magnitude() > allowed_tilt
        {
            return None;
        }
        feet.push((foot.u, foot.v));
    }
    Some((
        covering_range(feet.iter().map(|f| f.0), other.u_period(), other.u_bounds()),
        covering_range(feet.iter().map(|f| f.1), other.v_period(), other.v_bounds()),
    ))
}

/// The range of a parameter holding `values`: from the least to the
/// greatest, or for a periodic parameter (within `bounds`) the whole
/// period, unless the values leave a gap across the period's seam more
/// than twice as wide as any between them. Samples cannot tell how far
/// between them the values run, so a range that would end inside a gap
/// no wider than theirs takes the whole period.
fn covering_range(
    values: impl Iterator<Item = f64>,
    period: Option<f64>,
    bounds: (f64, f64),
) -> (f64, f64) {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(f64::total_cmp);
    let (first, last) = (values[0], values[values.len() - 1]);
    let Some(period) = period else {
        return (first, last);
    };
    let widest_between = values.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
    if first + period - last > 2.0 * widest_between {
        (first, last)
    } else {
        bounds
    }
}

/// The unit normal `su x sv`, or `None` where it degenerates: where the
/// derivatives are parallel to within `NORMAL_DEGENERACY` of their size,
/// as at a sphere's pole, where `su` vanishes and rounding leaves its
/// direction arbitrary.
fn unit_normal(su: Vector3D, sv: Vector3D) -> Option<Vector3D> {
    let n = su.cross(sv);
    if n.magnitude() <= NORMAL_DEGENERACY * (su.dot(su) + sv.dot(sv)) {
        return None;
    }
    n.normalized()
}

/// The size of `su x sv`, relative to `|su|^2 + |sv|^2`, below which a
/// normal degenerates (see [`unit_normal`]).
const NORMAL_DEGENERACY: f64 = 1e-9;

/// The corners of the box around `s`: its own box when it is elementary
/// or a B-spline, else the box around a grid of its points.
fn surface_corners<S: ParametricSurface>(s: &S) -> (Point3D, Point3D) {
    let own = match (s.as_elementary(), s.as_bspline()) {
        (Some(elementary), _) => elementary.bounding_box(),
        (None, Some(bspline)) => bspline.bounding_box(),
        (None, None) => None,
    };
    let bounds = own.or_else(|| {
        let (u0, u1) = s.u_bounds();
        let (v0, v1) = s.v_bounds();
        let last = (COINCIDENCE_SAMPLES - 1) as f64;
        BoundingBox3D::from_points((0..COINCIDENCE_SAMPLES).flat_map(|i| {
            (0..COINCIDENCE_SAMPLES).map(move |j| {
                s.eval_point(
                    u0 + (u1 - u0) * i as f64 / last,
                    v0 + (v1 - v0) * j as f64 / last,
                )
            })
        }))
    });
    bounds.map_or((Point3D::ORIGIN, Point3D::ORIGIN), |b| (b.min(), b.max()))
}

/// Seeds `[a_u, a_v, b_u, b_v]` on both surfaces when each is elementary
/// or a B-spline, and one is elementary: the union of
/// - the points where the seeding iso-curves of an elementary side meet
///   the other side exactly or through the B-spline search (see
///   [`family_seeds`]);
/// - a B-spline side's own parameter grid (see [`grid_seeds`]), and both
///   grids when both surfaces are bounded (spheres, tori, B-splines).
///
/// `None` otherwise: the caller then seeds from both grids.
fn pair_seeds<A, B>(a: &A, b: &B, tol: Tolerance) -> Option<Vec<[f64; 4]>>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let (first, second) = (SeedTarget::of(a)?, SeedTarget::of(b)?);
    let mut seeds = Vec::new();
    let mut seeded = false;
    // A B-spline side always seeds from its grid; so do both sides when
    // both are bounded (a sphere, torus or B-spline), where the families
    // alone may pass between small loops.
    let both_bounded = first.bounding_box().is_some() && second.bounding_box().is_some();
    if both_bounded || matches!(first, SeedTarget::BSpline(_)) {
        seeded = true;
        seeds.extend(grid_seeds(a, b, tol));
    }
    if both_bounded || matches!(second, SeedTarget::BSpline(_)) {
        seeded = true;
        seeds.extend(
            grid_seeds(b, a, tol)
                .into_iter()
                .map(|x| [x[2], x[3], x[0], x[1]]),
        );
    }
    // Where the grids find the surfaces only touching, the iso-curves
    // touch the other surface too, and their search would settle each
    // touch by halving down to its smallest pieces: the families are left
    // out, and the touches traced from the grid seeds.
    if !seeds.is_empty() && seeds.iter().all(|&x| grazes(a, b, x, tol)) {
        return Some(seeds);
    }
    if let SeedTarget::Elementary(own) = first
        && let Some(family) = family_seeds(own, second, tol)
    {
        seeded = true;
        seeds.extend(family.into_iter().filter_map(|x| meet(a, b, x, None, tol)));
    }
    if let SeedTarget::Elementary(own) = second
        && let Some(family) = family_seeds(own, first, tol)
    {
        seeded = true;
        seeds.extend(
            family
                .into_iter()
                .filter_map(|x| meet(a, b, [x[2], x[3], x[0], x[1]], None, tol)),
        );
    }
    seeded.then_some(seeds)
}

/// `seeds` without those whose point on `a` lies within `tol.confusion`
/// of an earlier one's: the iso-curves through a sphere's pole or a cone's
/// apex, or coinciding lines of a plane, all give the same seed, and a
/// trace failing from it would be retried for each.
fn distinct_seeds<A: ParametricSurface>(
    a: &A,
    seeds: Vec<[f64; 4]>,
    tol: Tolerance,
) -> Vec<[f64; 4]> {
    let mut points: Vec<Point3D> = Vec::with_capacity(seeds.len());
    let mut kept = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let p = a.eval_point(seed[0], seed[1]);
        if points.iter().all(|q| q.distance(p) > tol.confusion) {
            points.push(p);
            kept.push(seed);
        }
    }
    kept
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

/// Parameters `[own_u, own_v, other_u, other_v]` of the points where the
/// seeding iso-curves of `own` meet `other`: every hit, and a point of
/// every stretch of an iso-curve lying on `other`. `None` when `own` has
/// no seeding iso-curves against `other` (see [`seeding_iso_values`]).
fn family_seeds(
    own: ElementarySurface,
    other: SeedTarget,
    tol: Tolerance,
) -> Option<Vec<[f64; 4]>> {
    let surface = Surface::from(own);
    let values = seeding_iso_values(own, other, tol);
    if values.is_empty() {
        return None;
    }
    let mut seeds = Vec::new();
    for (direction, value) in values {
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
    Some(seeds)
}

/// The iso-curves `own` seeds from, as `(direction, fixed value)` (see
/// [`marching_intersection`]). Circles are left out against a torus, where
/// a circle meets it only through the sampled search: a sphere or torus
/// then has none.
fn seeding_iso_values(
    own: ElementarySurface,
    other: SeedTarget,
    tol: Tolerance,
) -> Vec<(IsoDirection, f64)> {
    let periodic =
        |period: f64| (0..ISO_CURVE_COUNT).map(move |i| period * i as f64 / ISO_CURVE_COUNT as f64);
    let tau = std::f64::consts::TAU;
    let circles_meet_exactly =
        !matches!(other, SeedTarget::Elementary(ElementarySurface::Torus(_)));
    // The range of `own`'s parameters at the corners of the other surface's
    // box, grown by 10 %: `ISO_CURVE_COUNT` values across it, ends included
    // (one value when the range is flat).
    let across_box = |bounds: BoundingBox3D| {
        let (min, max) = (bounds.min(), bounds.max());
        let (mut low, mut high) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for k in 0..8 {
            let (u, v) = own.parameters_of(Point3D::new(
                if k & 1 == 0 { min.x } else { max.x },
                if k & 2 == 0 { min.y } else { max.y },
                if k & 4 == 0 { min.z } else { max.z },
            ));
            (low[0], high[0]) = (low[0].min(u), high[0].max(u));
            (low[1], high[1]) = (low[1].min(v), high[1].max(v));
        }
        move |k: usize| -> Vec<f64> {
            if high[k] - low[k] <= tol.confusion {
                return vec![0.5 * (low[k] + high[k])];
            }
            let margin = 0.05 * (high[k] - low[k]);
            let (lo, hi) = (low[k] - margin, high[k] + margin);
            (0..ISO_CURVE_COUNT)
                .map(|i| lo + (hi - lo) * i as f64 / (ISO_CURVE_COUNT - 1) as f64)
                .collect()
        }
    };
    match own {
        ElementarySurface::Plane(_) => match other.bounding_box() {
            Some(bounds) => {
                let across = across_box(bounds);
                let lines = across(0).into_iter().map(|u| (IsoDirection::U, u));
                lines
                    .chain(across(1).into_iter().map(|v| (IsoDirection::V, v)))
                    .collect()
            }
            // An unbounded other surface: two crossing lines meet any plane
            // that is not parallel; a cylinder or cone seeds from its own
            // rulings.
            None => vec![(IsoDirection::U, 0.0), (IsoDirection::V, 0.0)],
        },
        ElementarySurface::Cylinder(_) | ElementarySurface::Cone(_) => {
            let rulings = periodic(tau).map(|u| (IsoDirection::U, u));
            // Circles across the other surface's box, or the one at `v = 0`
            // against an unbounded surface (which a ruling may run
            // alongside, as for two cylinders with parallel axes).
            let circles = match (circles_meet_exactly, other.bounding_box()) {
                (false, _) => Vec::new(),
                (true, Some(bounds)) => across_box(bounds)(1),
                (true, None) => vec![0.0],
            };
            rulings
                .chain(circles.into_iter().map(|v| (IsoDirection::V, v)))
                .collect()
        }
        _ if !circles_meet_exactly => Vec::new(),
        // A meridian circle is whole: the one at `u + PI` is the same.
        ElementarySurface::Sphere(_) => periodic(0.5 * tau).map(|u| (IsoDirection::U, u)).collect(),
        ElementarySurface::Torus(_) => periodic(tau).map(|u| (IsoDirection::U, u)).collect(),
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
    traced.curves.into_iter().map(|(curve, _)| curve).collect()
}

/// The curves of [`marching_intersection`], fitted.
#[cfg(test)]
pub(super) fn marched_curves<A, B>(a: &A, b: &B, tol: Tolerance) -> Vec<BSplineCurve3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    marching_intersection(a, b, tol)
        .curves()
        .iter()
        .filter_map(|piece| match &piece.curve {
            Curve3D::BSpline(curve) => Some(curve.clone()),
            _ => None,
        })
        .collect()
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

/// The curves traced so far, fitted, with whether the surfaces touch
/// along each (see [`grazes`]); the points where they touch without a
/// curve to trace; the seeds where they lie on each other (see
/// [`osculates`]); and the traced polylines: the box around each, the
/// boxes around its runs of `RUN` segments, and its points.
#[derive(Default)]
struct Traced {
    curves: Vec<(BSplineCurve3D, bool)>,
    contact_points: Vec<Point3D>,
    coincident_seeds: Vec<[f64; 4]>,
    traces: Vec<(Aabb, Vec<Aabb>, Vec<Point3D>)>,
}

impl Traced {
    /// Traces the curve through `seed` unless it was traced already. A
    /// seed where the surfaces touch is kept as a contact point when no
    /// curve runs from it.
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
        if outside_model(p) || self.crosses(&[p.x, p.y, p.z, p.x, p.y, p.z], tol) {
            return;
        }
        let touching = grazes(a, b, seed, tol);
        let points = trace(a, b, seed, touching, tol);
        // Where the surfaces touch at a point, the trace runs only across
        // the patch around it where they lie within confusion of each
        // other, shorter than a first step: a contact point, not a curve.
        let shortest = first_step(tol);
        let length =
            |points: &[Point3D]| points.windows(2).map(|w| w[0].distance(w[1])).sum::<f64>();
        let Some(points) = points.filter(|points| !touching || length(points) >= shortest) else {
            let gap = 8.0 * CHORD_DEVIATION * tol.confusion;
            if touching && osculates(a, b, seed, tol) {
                // The surfaces lie on each other here: no point stands
                // for the contact.
                self.coincident_seeds.push(seed);
            } else if touching && self.contact_points.iter().all(|q| q.distance(p) > gap) {
                self.contact_points.push(p);
            }
            return;
        };
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
            self.curves.push((fit, touching));
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

/// The length of a trace's first step: the chord of a unit circle
/// deviating from it by the allowed chord deviation.
fn first_step(tol: Tolerance) -> f64 {
    (8.0 * CHORD_DEVIATION * tol.confusion).sqrt()
}

/// Trace the intersection curve through `seed` both ways. Returns its
/// points in order (a closed loop ends on its first point), or `None`
/// when fewer than four were traced. Where the surfaces touch
/// (`touching`), the direction at each point is the one along which they
/// keep touching where their normals are parallel (see [`direction`]),
/// turned to go on the way the trace goes.
fn trace<A, B>(a: &A, b: &B, seed: [f64; 4], touching: bool, tol: Tolerance) -> Option<Vec<Point3D>>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    // Chord deviation allowed per step. The step follows the curvature of
    // the last one (a chord deviation of `chord_deviation`), and a step
    // whose corrected midpoint strays farther from its chord is halved, so
    // a feature as wide as the step is not crossed. On a B-spline side a
    // step crosses at most one knot (see `crosses_one_knot_at_most`), so it
    // never steps over a whole knot span: within a span its control points
    // bound the features.
    let chord_deviation = CHORD_DEVIATION * tol.confusion;
    // A trace ends on closing its loop, on a parameter bound, or on the
    // model-size cube; this count only guards against a runaway trace (a
    // loop whose closure is missed). A curve of length `L` and curvature
    // `k` takes about `L * sqrt(k / (8 * chord_deviation))` steps: a loop
    // of radius 50 about 16 per unit of length.
    const SAFETY_STEP_LIMIT: usize = 100_000;
    // Turn always allowed in one step. Beyond it, a step may turn by at
    // most `CURVATURE_GROWTH` times the previous step's curvature times
    // its chord: on a smooth curve turn/chord tends to the curvature as
    // the step is halved, so the step is eventually accepted, while at a
    // crossing of two branches (where the surfaces touch) the turn onto
    // the other branch stays finite, and the trace ends there.
    const FREE_TURN: f64 = 0.05;
    const CURVATURE_GROWTH: f64 = 4.0;
    let start = a.eval_point(seed[0], seed[1]);
    let t0 = direction(a, b, seed, touching, tol)?;
    let mut halves: [Vec<Point3D>; 2] = [Vec::new(), Vec::new()];
    let mut closed = false;
    for (half, dir) in halves.iter_mut().zip([1.0, -1.0]) {
        if closed {
            break;
        }
        let (mut x, mut p, mut t) = (seed, start, t0 * dir);
        let mut h = first_step(tol);
        // Curvature of the last accepted step (none before the first).
        let mut last_curvature = 0.0;
        for _ in 0..SAFETY_STEP_LIMIT {
            // Predict along the tangent, correct onto both surfaces in the
            // plane normal to it; halve the step until the result stays on
            // this branch with a small chord deviation.
            let mut step = None;
            let mut boundary_point = None;
            while h >= tol.confusion {
                let target = p + t * h;
                let guess = predict(a, b, x, t * h);
                if let Some((pin, bound)) = leaving(a, b, x, guess) {
                    // The step leaves a parameter domain: end on its boundary.
                    if let Some(y) = land(a, b, guess, pin, bound, tol) {
                        let q = a.eval_point(y[0], y[1]);
                        if (q - p).dot(t) >= 0.0
                            && q.distance(p) <= 2.0 * h
                            && crosses_one_knot_at_most(a, b, x, y, tol)
                            && !bulges(a, b, (x, p), (y, q), chord_deviation, touching, tol)
                        {
                            boundary_point = Some(q);
                            break;
                        }
                    }
                } else if let Some(y) = correct(a, b, guess, (target, t), touching, tol)
                    && let Some(ty) = direction(a, b, y, touching, tol).map(|ty| {
                        if touching && ty.dot(t) < 0.0 {
                            -ty
                        } else if touching {
                            ty
                        } else {
                            ty * dir
                        }
                    })
                {
                    let q = a.eval_point(y[0], y[1]);
                    let chord = q.distance(p);
                    let turn = t.cross(ty).magnitude().atan2(t.dot(ty));
                    let smooth = turn <= FREE_TURN.max(CURVATURE_GROWTH * last_curvature * chord);
                    if ty.dot(t) > 0.0
                        && chord <= 2.0 * h
                        && chord * turn <= 32.0 * chord_deviation
                        && smooth
                        && crosses_one_knot_at_most(a, b, x, y, tol)
                        && !bulges(a, b, (x, p), (y, q), chord_deviation, touching, tol)
                    {
                        step = Some((y, q, ty, turn / chord.max(f64::MIN_POSITIVE)));
                        break;
                    }
                }
                h *= 0.5;
            }
            if let Some(q) = boundary_point {
                if q.distance(p) > tol.confusion.max(crate::tol::CONFUSION) {
                    // A last step much shorter than the one before makes
                    // the fitted curve overshoot the boundary: the boundary point
                    // then replaces the last point.
                    let before = half.len().checked_sub(2).map_or(start, |i| half[i]);
                    if !half.is_empty() && q.distance(p) < 0.25 * p.distance(before) {
                        half.pop();
                    }
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
            if outside_model(q) {
                half.push(model_exit(a, b, (x, p), (y, q), tol));
                break;
            }
            half.push(q);
            (x, p, t) = (y, q, ty);
            last_curvature = curvature;
            h = (2.0 * h).min((8.0 * chord_deviation / curvature.max(f64::MIN_POSITIVE)).sqrt());
        }
    }
    let [forward, backward] = halves;
    let mut points: Vec<Point3D> = backward.into_iter().rev().collect();
    points.push(start);
    points.extend(forward);
    (points.len() >= 4).then_some(points)
}

/// Multiple of the chord deviation a step's midpoint may stray from its
/// chord (see [`bulges`]).
const MIDPOINT_SLACK: f64 = 2.0;

/// Whether the intersection between the traced points `p` (parameters
/// `x`) and `q` (parameters `y`) strays from their chord by more than
/// `deviation` at its midpoint, or cannot be found there: the midpoint of
/// the parameters converged onto both surfaces in the plane through the
/// chord's midpoint, normal to it.
fn bulges<A, B>(
    a: &A,
    b: &B,
    (x, p): ([f64; 4], Point3D),
    (y, q): ([f64; 4], Point3D),
    deviation: f64,
    touching: bool,
    tol: Tolerance,
) -> bool
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let chord = q - p;
    let length = chord.magnitude();
    if length <= tol.confusion {
        return false;
    }
    let middle = p + chord * 0.5;
    let guess = std::array::from_fn(|k| 0.5 * (x[k] + y[k]));
    correct(a, b, guess, (middle, chord * (1.0 / length)), touching, tol)
        .is_none_or(|z| a.eval_point(z[0], z[1]).distance(middle) > MIDPOINT_SLACK * deviation)
}

/// Whether the move from parameters `x` to parameters `y` crosses at most
/// one knot in each parameter of a B-spline side (knots at either end
/// count), so that it never both enters and leaves a knot span and steps
/// over the span. Spans crossed whole that together are no longer than
/// the confusion distance (see [`crossed_knots_fit`]) are let through: no
/// step could land inside them.
fn crosses_one_knot_at_most<A, B>(a: &A, b: &B, x: [f64; 4], y: [f64; 4], tol: Tolerance) -> bool
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let on = |surface: Option<&BSplineSurface>, k: usize| {
        surface.is_none_or(|s| {
            let (from, to) = ([x[k], x[k + 1]], [y[k], y[k + 1]]);
            crossed_knots_fit(s, 0, from, to, tol) && crossed_knots_fit(s, 1, from, to, tol)
        })
    };
    on(a.as_bspline(), 0) && on(b.as_bspline(), 2)
}

/// Whether the move from parameters `from` to parameters `to` on `surface`
/// crosses at most one knot in its parameter `direction` (0 for u, 1 for
/// v), or only a run of knots whose spans are tiny in space: the surface
/// points over the crossed knots, along the move (the other parameter
/// interpolated), lie on a path no longer than the confusion distance.
/// Their size is measured in space, not from the move's average speed,
/// because that speed may jump at a knot where the surface is only
/// continuous, and a cluster narrow in parameter can be long in space.
fn crossed_knots_fit(
    surface: &BSplineSurface,
    direction: usize,
    from: [f64; 2],
    to: [f64; 2],
    tol: Tolerance,
) -> bool {
    let (knots, periodic) = if direction == 0 {
        (surface.u_knots(), surface.is_u_periodic())
    } else {
        (surface.v_knots(), surface.is_v_periodic())
    };
    let (start, end) = (from[direction], to[direction]);
    let (low, high) = (start.min(end), start.max(end));
    let first = knots_below(knots, periodic, low, false);
    let last = knots_below(knots, periodic, high, true) - 1;
    if last <= first {
        return true;
    }
    // More than one knot crossed (so `start != end`): rare, and the
    // evaluations stop as soon as the crossed spans are known to be long.
    let other = 1 - direction;
    let point_over_knot = |index: i64| {
        let crossed = knot(knots, periodic, index);
        let fraction = (crossed - start) / (end - start);
        let interpolated = from[other] + fraction * (to[other] - from[other]);
        let (u, v) = if direction == 0 {
            (crossed, interpolated)
        } else {
            (interpolated, crossed)
        };
        surface.eval_point(u, v)
    };
    let mut length = 0.0;
    let mut previous = point_over_knot(first);
    for index in first + 1..=last {
        let point = point_over_knot(index);
        length += previous.distance(point);
        if length > tol.confusion {
            return false;
        }
        previous = point;
    }
    true
}

/// The number of knots below `at` (or at it when `inclusive`). A periodic
/// direction repeats its knots every period, its last knot being the next
/// period's first, so the count runs on past the range and below zero.
fn knots_below(knots: &[f64], periodic: bool, at: f64, inclusive: bool) -> i64 {
    let below = |knots: &[f64], at: f64| {
        let count = if inclusive {
            knots.partition_point(|&k| k <= at)
        } else {
            knots.partition_point(|&k| k < at)
        };
        count as i64
    };
    if !periodic {
        return below(knots, at);
    }
    let first = knots[0];
    let period = knots[knots.len() - 1] - first;
    let once = &knots[..knots.len() - 1];
    let turns = ((at - first) / period).floor();
    turns as i64 * once.len() as i64 + below(once, at - turns * period)
}

/// The knot of index `index` in the count of [`knots_below`]; minus or
/// plus infinity before or after a non-periodic range.
fn knot(knots: &[f64], periodic: bool, index: i64) -> f64 {
    if !periodic {
        return match usize::try_from(index) {
            Err(_) => f64::NEG_INFINITY,
            Ok(i) => knots.get(i).copied().unwrap_or(f64::INFINITY),
        };
    }
    let first = knots[0];
    let period = knots[knots.len() - 1] - first;
    let count = knots.len() as i64 - 1;
    knots[index.rem_euclid(count) as usize] + index.div_euclid(count) as f64 * period
}

/// Half the side of the model-size cube centred on the origin, where
/// traces end (see [`Tolerance::MODEL_SIZE`]).
const MODEL_HALF_SIZE: f64 = 0.5 * Tolerance::MODEL_SIZE;

/// Whether `p` lies outside the model-size cube.
fn outside_model(p: Point3D) -> bool {
    [p.x, p.y, p.z].iter().any(|c| c.abs() > MODEL_HALF_SIZE)
}

/// The point where the traced step from `p` (parameters `x`, inside the
/// model-size cube) to `q` (parameters `y`, outside it) leaves the cube:
/// converged onto both surfaces and the first face the chord crosses, or
/// `q` when that does not converge.
fn model_exit<A, B>(
    a: &A,
    b: &B,
    (x, p): ([f64; 4], Point3D),
    (y, q): ([f64; 4], Point3D),
    tol: Tolerance,
) -> Point3D
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let (from, to) = ([p.x, p.y, p.z], [q.x, q.y, q.z]);
    let mut first: Option<(usize, f64, f64)> = None;
    for k in 0..3 {
        if to[k].abs() > MODEL_HALF_SIZE {
            let face = MODEL_HALF_SIZE.copysign(to[k]);
            let fraction = (face - from[k]) / (to[k] - from[k]);
            if first.is_none_or(|(_, _, f)| fraction < f) {
                first = Some((k, face, fraction));
            }
        }
    }
    let Some((axis, face, fraction)) = first else {
        return q;
    };
    let guess = std::array::from_fn(|k| x[k] + (y[k] - x[k]) * fraction);
    let mut normal = [0.0; 3];
    normal[axis] = 1.0;
    let mut origin = [0.0; 3];
    origin[axis] = face;
    let plane = (
        Point3D::new(origin[0], origin[1], origin[2]),
        Vector3D::new(normal[0], normal[1], normal[2]),
    );
    meet(a, b, guess, Some(plane), tol).map_or(q, |z| a.eval_point(z[0], z[1]))
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

/// The direction of the intersection at parameters `x`: the unit `N1 x
/// N2` (see [`tangent`]), or where the surfaces touch (`touching`), the
/// direction along which they keep touching (see [`contact_direction`]):
/// there the normals are parallel, or so nearly that their cross product
/// points anywhere in the tangent plane. `None` where it does not exist.
fn direction<A, B>(a: &A, b: &B, x: [f64; 4], touching: bool, tol: Tolerance) -> Option<Vector3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    if touching {
        contact_direction(a, b, x, tol)
    } else {
        tangent(a, b, x, tol)
    }
}

/// The second fundamental forms of both surfaces at parameters `x`, in
/// one orthonormal basis `(e1, e2)` of the first surface's tangent plane,
/// the second surface's normal turned to agree with the first's: `[[l,
/// m], [m, n]]` for each, with the unit normals and the basis. `None`
/// where a normal degenerates.
#[allow(clippy::type_complexity)]
fn second_forms<A, B>(
    a: &A,
    b: &B,
    x: [f64; 4],
) -> Option<([[f64; 3]; 2], (Vector3D, Vector3D), (Vector3D, Vector3D))>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let (_, [au, av], a2) = a.eval_d2(x[0], x[1]);
    let (_, [bu, bv], b2) = b.eval_d2(x[2], x[3]);
    let na = unit_normal(au, av)?;
    let mut nb = unit_normal(bu, bv)?;
    if na.dot(nb) < 0.0 {
        nb = -nb;
    }
    let e1 = au.normalized()?;
    let e2 = na.cross(e1);
    let form = |su: Vector3D, sv: Vector3D, [suu, suv, svv]: [Vector3D; 3], n: Vector3D| {
        // Coordinates of `d`, a vector of the tangent plane, in `(su, sv)`.
        let m = [[su.dot(su), su.dot(sv)], [su.dot(sv), sv.dot(sv)]];
        let coordinates = |d: Vector3D| solve_2x2(m, [su.dot(d), sv.dot(d)], 1e-12);
        let [a1, b1] = coordinates(e1)?;
        let [a2, b2] = coordinates(e2)?;
        let [l, mixed, n2] = [suu.dot(n), suv.dot(n), svv.dot(n)];
        let pair =
            |p: f64, q: f64, r: f64, s: f64| p * r * l + (p * s + q * r) * mixed + q * s * n2;
        Some([
            pair(a1, b1, a1, b1),
            pair(a1, b1, a2, b2),
            pair(a2, b2, a2, b2),
        ])
    };
    let first = form(au, av, a2, na)?;
    let second = form(bu, bv, b2, nb)?;
    Some(([first, second], (na, nb), (e1, e2)))
}

/// Where the surfaces touch at parameters `x`, the direction along which
/// they keep touching: the direction of the common
/// tangent plane in which their normal curvatures agree best (the
/// eigenvector of the smaller eigenvalue of the difference of their
/// second fundamental forms). `None` where a normal degenerates, or where
/// the surfaces osculate (see [`osculates`]).
fn contact_direction<A, B>(a: &A, b: &B, x: [f64; 4], tol: Tolerance) -> Option<Vector3D>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let ([first, second], _, (e1, e2)) = second_forms(a, b, x)?;
    let [p, q, r] = std::array::from_fn(|k| first[k] - second[k]);
    let spread = (0.25 * (p - r) * (p - r) + q * q).sqrt();
    let mean = 0.5 * (p + r);
    if spread + mean.abs() <= osculation(tol) {
        return None;
    }
    // The eigenvalue of smaller magnitude, and its eigenvector.
    let lambda = if mean >= 0.0 {
        mean - spread
    } else {
        mean + spread
    };
    let (x1, y1) = (q, lambda - p);
    let (x2, y2) = (lambda - r, q);
    let (s, t) = if x1 * x1 + y1 * y1 >= x2 * x2 + y2 * y2 {
        (x1, y1)
    } else {
        (x2, y2)
    };
    if s == 0.0 && t == 0.0 {
        // Already diagonal with a zero off-diagonal: an axis is the
        // eigenvector.
        return Some(if p.abs() <= r.abs() { e1 } else { e2 });
    }
    (e1 * s + e2 * t).normalized()
}

/// Whether the surfaces, touching at parameters `x`, osculate: their
/// normal curvatures agree in every direction to within [`osculation`],
/// so that over a first step they stay within confusion of each other.
/// They then lie on each other there, at the resolution of marching.
fn osculates<A, B>(a: &A, b: &B, x: [f64; 4], tol: Tolerance) -> bool
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    second_forms(a, b, x).is_some_and(|([first, second], _, _)| {
        let [p, q, r] = std::array::from_fn(|k| first[k] - second[k]);
        let spread = (0.25 * (p - r) * (p - r) + q * q).sqrt();
        spread + (0.5 * (p + r)).abs() <= osculation(tol)
    })
}

/// The difference of normal curvatures below which touching surfaces
/// osculate: surfaces whose curvatures differ by `k` part by `k h^2 / 2`
/// over a step `h`, within confusion over a first step when `k <= 2
/// confusion / h^2`.
fn osculation(tol: Tolerance) -> f64 {
    2.0 * tol.confusion / (first_step(tol) * first_step(tol))
}

/// Whether the surfaces touch at parameters `x` (a point on both) rather
/// than cross: their normals are parallel within `tol.angular`, or they
/// cross at so small an angle that moving one by `tol.confusion` along
/// its normal makes them touch.
///
/// Across the intersection, at the distance `w` along the first surface,
/// the second lies `w sin(theta) + k w^2 / 2` away, `theta` being the
/// angle between the normals and `k` the difference of the surfaces'
/// normal curvatures across it. Moved by `g` along its normal, the second
/// surface touches the first (the separation has a double root) when
/// `sin^2(theta) = 2 k g`: the crossing is a touch within confusion when
/// `sin^2(theta) <= 2 |k| confusion`. This is the rule of the exact
/// solvers, a plane being tangent to a cylinder when its distance to the
/// axis is the radius within confusion. `false` where a normal
/// degenerates (a singular point).
fn grazes<A, B>(a: &A, b: &B, x: [f64; 4], tol: Tolerance) -> bool
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let Some(([first, second], (na, nb), (e1, e2))) = second_forms(a, b, x) else {
        return false;
    };
    let sine = na.cross(nb).magnitude();
    if sine <= tol.angular {
        return true;
    }
    // The unit direction across the intersection in the first surface's
    // tangent plane, in the basis `(e1, e2)`.
    let across = na.cross(na.cross(nb) * (1.0 / sine));
    let (c, s) = (across.dot(e1), across.dot(e2));
    let curvature = |[l, m, n]: [f64; 3]| c * c * l + 2.0 * c * s * m + s * s * n;
    sine * sine <= 2.0 * (curvature(first) - curvature(second)).abs() * tol.confusion
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
            let (trial_residual, trial_normal_residual, trial_merit) = residual(trial);
            if trial_merit < merit {
                (x, f, r, merit) = (trial, trial_residual, trial_normal_residual, trial_merit);
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

/// A point on both surfaces in the plane through `origin` normal to
/// `normal`, from parameters `x`: by [`meet`], or where the surfaces
/// touch (`touching`) by [`meet_touching`].
fn correct<A, B>(
    a: &A,
    b: &B,
    x: [f64; 4],
    plane: (Point3D, Vector3D),
    touching: bool,
    tol: Tolerance,
) -> Option<[f64; 4]>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    if touching {
        meet_touching(a, b, x, plane, tol)
    } else {
        meet(a, b, x, Some(plane), tol)
    }
}

/// [`meet`] where the surfaces touch: there the system of `meet` is
/// singular, its rows from `S1 - S2` spanning only the common tangent
/// plane, and the gap along the normal shrinks only to second order. The
/// least-squares step is damped instead (Levenberg-Marquardt), which
/// stays defined and closes that gap a fraction at a time. `None` when
/// the point is not reached within `tol.confusion`.
fn meet_touching<A, B>(
    a: &A,
    b: &B,
    x: [f64; 4],
    (origin, normal): (Point3D, Vector3D),
    tol: Tolerance,
) -> Option<[f64; 4]>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    let residual = |x: [f64; 4]| {
        let pa = a.eval_point(x[0], x[1]);
        let f = pa - b.eval_point(x[2], x[3]);
        let r = (pa - origin).dot(normal);
        ([f.x, f.y, f.z, r], f.dot(f) + r * r)
    };
    let mut x = within_bounds(a, b, x);
    let (mut f, mut merit) = residual(x);
    let mut damping = 1e-3;
    for _ in 0..64 {
        if merit <= (0.1 * tol.confusion).powi(2) {
            return Some(x);
        }
        let (_, [au, av]) = a.eval_d1(x[0], x[1]);
        let (_, [bu, bv]) = b.eval_d1(x[2], x[3]);
        let cols = [au, av, -bu, -bv];
        let rows = [
            cols.map(|c| c.x),
            cols.map(|c| c.y),
            cols.map(|c| c.z),
            [au.dot(normal), av.dot(normal), 0.0, 0.0],
        ];
        let normal_matrix: [[f64; 4]; 4] = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..4).map(|k| rows[k][i] * rows[k][j]).sum())
        });
        let gradient: [f64; 4] =
            std::array::from_fn(|i| -(0..4).map(|k| rows[k][i] * f[k]).sum::<f64>());
        let mut improved = false;
        for _ in 0..16 {
            let mut damped = normal_matrix;
            for (k, row) in damped.iter_mut().enumerate() {
                row[k] += damping * normal_matrix[k][k].max(f64::MIN_POSITIVE);
            }
            if let Some(delta) = solve_4x4(damped, gradient, 0.0) {
                let trial = within_bounds(a, b, std::array::from_fn(|k| x[k] + delta[k]));
                let (trial_residual, trial_merit) = residual(trial);
                if trial_merit < merit {
                    (x, f, merit) = (trial, trial_residual, trial_merit);
                    damping = (damping * 0.1).max(1e-12);
                    improved = true;
                    break;
                }
            }
            damping *= 10.0;
        }
        if !improved {
            break;
        }
    }
    (merit <= tol.confusion * tol.confusion).then_some(x)
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
            let trial_residual = residual(trial);
            if trial_residual.square_magnitude() < f.square_magnitude() {
                (x, f) = (trial, trial_residual);
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
        let curves = marched_curves(&xy, &zy, tol);
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
        let curves = marched_curves(&plane, &sphere, tol);
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
        let curves = marched_curves(&plane, &sphere, tol);
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
            let curves = marched_curves(&plane, &sphere, tol);
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
        let curves = marched_curves(&cylinder, &sphere, tol);
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
            let curves = marched_curves(&a, &b, tol);
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
        let curves = marched_curves(&plane, &sphere, tol);
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
        let curves = marched_curves(&xy, &zy, tol);
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
        let curves = marched_curves(&torus, &plane, tol);
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
            let curves = marched_curves(&torus, &plane, tol);
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
            marched_curves(&along_x, &along_y, tol),
            marched_curves(&along_y, &along_x, tol),
            marched_curves(&along_x, &along_y_far, tol),
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
        let curves = marched_curves(&along_x, &along_y, tol);
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
        let curves = marched_curves(&torus, &plane, tol);
        assert_eq!(curves.len(), 2);
        assert_closed_on_both(&curves, &torus, &plane);
    }

    /// Asserts every traced curve lies on both surfaces.
    fn assert_on_both<A, B>(curves: &[BSplineCurve3D], a: &A, b: &B)
    where
        A: ParametricSurface,
        B: ParametricSurface,
    {
        let tol = Tolerance::DEFAULT;
        for curve in curves {
            let (t0, t1) = curve.bounds();
            for k in 0..=40 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 40.0);
                assert!(a.project_point(p, tol).distance < 1e-3, "{p:?}");
                assert!(b.project_point(p, tol).distance < 1e-3, "{p:?}");
            }
        }
    }

    /// A cubic wall waving between x = 0.1 and x = 0.4 over y in
    /// `[-2.5, 2.5]`, extruded along z over `[-3, 3]`.
    fn extruded_wall() -> crate::surfaces::BSplineSurface {
        let poles = (0..6)
            .map(|i| {
                let y = -2.5 + i as f64;
                let x = if i % 2 == 0 { 0.4 } else { 0.1 };
                vec![Point3D::new(x, y, -3.0), Point3D::new(x, y, 3.0)]
            })
            .collect();
        crate::surfaces::BSplineSurface::new(
            3,
            1,
            poles,
            vec![0.0, 1.0, 2.0, 3.0],
            vec![4, 1, 1, 4],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_cylinder_along_an_extruded_wall_keeps_the_wall_seeds() {
        // Every ruling runs along (or nearly along) the extrusion, so the
        // wall's own grid seeds find the two curves.
        use crate::{Cylinder, Intersect};
        let tol = Tolerance::DEFAULT;
        let wall = extruded_wall();
        let tilt = 3f64.to_radians();
        for axis in [Vector3D::Z, Vector3D::new(0.0, tilt.sin(), tilt.cos())] {
            let cylinder = Cylinder::new(Point3D::ORIGIN, axis, 1.0).unwrap();
            let result = cylinder.intersect(&wall, tol);
            let curves: Vec<BSplineCurve3D> = result
                .curves()
                .iter()
                .map(|c| c.curve.as_bspline().unwrap().clone())
                .collect();
            assert_eq!(curves.len(), 2, "axis {axis:?}");
            assert_on_both(&curves, &cylinder, &wall);
        }
    }

    #[test]
    fn test_marching_parallel_cylinders_seed_from_circles() {
        // Every ruling is parallel to the other axis; a circle meets both
        // lines of the intersection.
        use crate::Cylinder;
        let tol = Tolerance::DEFAULT;
        let first = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let second = Cylinder::new(Point3D::new(0.5, 0.0, 0.0), Vector3D::Z, 1.0).unwrap();
        let curves = marched_curves(&first, &second, tol);
        assert_eq!(curves.len(), 2);
        assert_on_both(&curves, &first, &second);
    }

    #[test]
    fn test_marching_sphere_between_torus_meridians() {
        // Spheres centred halfway between two seeding meridians of the
        // torus: the meridians on either side miss them, or lie wholly
        // inside the large one, so the grids find the curves.
        use crate::{Sphere, Torus};
        let tol = Tolerance::DEFAULT;
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
        let azimuth = 7.5f64.to_radians();
        for (distance, radius, count) in
            [(3.0, 1.2, 2), (4.0, 0.3, 1), (4.0, 0.15, 1), (2.0, 0.25, 1)]
        {
            let centre = Point3D::new(distance * azimuth.cos(), distance * azimuth.sin(), 0.0);
            let sphere = Sphere::new(centre, radius).unwrap();
            let curves = marched_curves(&sphere, &torus, tol);
            assert_eq!(curves.len(), count, "radius {radius} at {distance}");
            assert_closed_on_both(&curves, &sphere, &torus);
        }
    }

    #[test]
    fn test_marching_against_a_torus_finds_what_the_grid_finds() {
        // Random spheres, cylinders and cones about a torus: wherever the
        // grid seeds find curves, so does the seeding.
        use crate::{Cone, Cylinder, Sphere, Torus};
        let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
        let mut state = 42u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        fn check<S: ParametricSurface>(surface: &S, torus: &crate::Torus, case: usize) {
            let tol = Tolerance::DEFAULT;
            if !grid_marching(surface, torus, tol).is_empty() {
                assert!(
                    !marched_curves(surface, torus, tol).is_empty(),
                    "case {case}"
                );
            }
        }
        for case in 0..24 {
            let azimuth = next() * std::f64::consts::TAU;
            let distance = 1.5 + 3.0 * next();
            let height = -1.5 + 3.0 * next();
            let centre = Point3D::new(distance * azimuth.cos(), distance * azimuth.sin(), height);
            let radius = 0.1 + 1.5 * next();
            let axis = Vector3D::new(next() - 0.5, next() - 0.5, next() - 0.5);
            check(&Sphere::new(centre, radius).unwrap(), &torus, case);
            check(
                &Cylinder::new(centre, axis, 0.1 + 0.6 * next()).unwrap(),
                &torus,
                case,
            );
            let (semi_angle, reference) = (0.1 + 0.5 * next(), 0.1 + 0.6 * next());
            check(
                &Cone::new(centre, axis, semi_angle, reference).unwrap(),
                &torus,
                case,
            );
        }
    }

    /// The straight wall along the diagonal of the xy plane over 280,
    /// extruded along z over `[-1, 1]`, with one pole pushed off by `bump`
    /// at `at` along it, its neighbours `width` apart, with the given knots
    /// (uniform when `None`).
    fn bumped_wall(
        bump: f64,
        width: f64,
        at: f64,
        knots: Option<&[f64]>,
    ) -> crate::surfaces::BSplineSurface {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let along = [
            0.0,
            40.0,
            80.0,
            120.0,
            at - 2.0 * width,
            at - width,
            at,
            at + width,
            at + 2.0 * width,
            200.0,
            240.0,
            280.0,
        ];
        let poles = along
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let off = if i == 6 { bump } else { 0.0 };
                let (x, y) = ((s + off) * r, (s - off) * r);
                vec![Point3D::new(x, y, -1.0), Point3D::new(x, y, 1.0)]
            })
            .collect::<Vec<_>>();
        let knots: Vec<f64> = knots.map_or_else(
            || (0..=poles.len() - 3).map(|k| k as f64).collect(),
            <[f64]>::to_vec,
        );
        let mut multiplicities = vec![1; knots.len()];
        multiplicities[0] = 4;
        *multiplicities.last_mut().unwrap() = 4;
        crate::surfaces::BSplineSurface::new(
            3,
            1,
            poles,
            knots,
            multiplicities,
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_marching_does_not_step_over_a_small_bump() {
        // A straight run with one small bump whose ends have the same
        // tangent: a long step across it would see no turn.
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        // Chord-length knots, and a wide span before narrow ones at the
        // bump: a step must not cross the narrow spans in one go.
        let chord_length = [
            0.0, 40.0, 80.0, 120.0, 149.0, 149.5, 150.0, 150.5, 151.0, 280.0,
        ];
        let wide_then_narrow = [0.0, 1.0, 2.0, 3.0, 3.001, 3.002, 3.003, 3.004, 3.005, 4.0];
        for (bump, width, at, knots) in [
            (0.05, 0.25, 125.0, None),
            (0.02, 0.1, 150.0, None),
            (0.2, 0.5, 175.0, None),
            (0.02, 0.1, 150.0, Some(&chord_length[..])),
            (0.05, 0.25, 150.0, Some(&wide_then_narrow[..])),
            (0.2, 0.5, 150.0, Some(&wide_then_narrow[..])),
        ] {
            let wall = bumped_wall(bump, width, at, knots);
            let curves = marched_curves(&plane, &wall, tol);
            assert_eq!(curves.len(), 1, "bump {bump} at {at}");
            let curve = &curves[0];
            let knots = curve.knots();
            for span in knots.windows(2) {
                for k in 0..20 {
                    let p = curve.eval_point(span[0] + (span[1] - span[0]) * k as f64 / 20.0);
                    assert!(
                        wall.project_point(p, tol).distance < 1e-3,
                        "bump {bump}: {p:?}"
                    );
                    assert!(p.z.abs() < 1e-3, "{p:?}");
                }
            }
        }
    }

    /// A wall of degree 1 along x over `[0, 100]`, extruded along z over
    /// `[-1, 1]`, at unit speed except over a tent of height `bump` on
    /// `[at, at + 2 * half]`, whose three knots are `cluster` apart: the
    /// speed jumps there by a factor `half / cluster`.
    fn tent_wall(bump: f64, half: f64, at: f64, cluster: f64) -> crate::surfaces::BSplineSurface {
        let along = [0.0, at, at + half, at + 2.0 * half, 100.0];
        let poles = along
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let y = if i == 2 { bump } else { 0.0 };
                vec![Point3D::new(x, y, -1.0), Point3D::new(x, y, 1.0)]
            })
            .collect::<Vec<_>>();
        let knots = vec![
            0.0,
            at,
            at + 0.5 * cluster,
            at + cluster,
            100.0 - 2.0 * half + cluster,
        ];
        crate::surfaces::BSplineSurface::new(
            1,
            1,
            poles,
            knots,
            vec![2, 1, 1, 1, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_marching_does_not_skip_a_tent_in_a_knot_cluster() {
        // The tent's knots are so close in parameter that their spans look
        // tiny at the step's average speed, while they are 0.1 to 1 long
        // in space: a step must not cross them whole. The curve breaks at
        // the tent's three corners into four pieces, each on both surfaces
        // and between consecutive corners, so none is skipped or traced
        // twice.
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let at = 50.0;
        for (cluster, bump, half, z0) in [
            (5e-8, 0.05, 0.05, 0.0),
            (5e-8, 0.2, 0.1, 0.0),
            (5e-8, 0.5, 0.5, 0.3),
            (1e-8, 0.05, 0.05, 0.3),
        ] {
            let wall = tent_wall(bump, half, at, cluster);
            let plane = Plane::new(Point3D::new(0.0, 0.0, z0), Vector3D::Z).unwrap();
            let curves = marched_curves(&plane, &wall, tol);
            assert_eq!(curves.len(), 4, "cluster {cluster:e} bump {bump}");
            // The distance to the wall, exactly: to the polyline through
            // its corners in the xy plane (its projection misreports inside
            // the tiny knot spans).
            let corners = [
                (0.0, 0.0),
                (at, 0.0),
                (at + half, bump),
                (at + 2.0 * half, 0.0),
                (100.0, 0.0),
            ];
            let wall_distance = |p: Point3D| {
                corners
                    .windows(2)
                    .map(|side| {
                        let ((ax, ay), (bx, by)) = (side[0], side[1]);
                        let (dx, dy) = (bx - ax, by - ay);
                        let along = (((p.x - ax) * dx + (p.y - ay) * dy) / (dx * dx + dy * dy))
                            .clamp(0.0, 1.0);
                        (p.x - ax - along * dx).hypot(p.y - ay - along * dy)
                    })
                    .fold(f64::INFINITY, f64::min)
            };
            let mut extents = Vec::new();
            for curve in &curves {
                let (mut lowest, mut highest) = (f64::INFINITY, f64::NEG_INFINITY);
                let knots = curve.knots();
                for span in knots.windows(2) {
                    for k in 0..=20 {
                        let p = curve.eval_point(span[0] + (span[1] - span[0]) * k as f64 / 20.0);
                        lowest = lowest.min(p.x);
                        highest = highest.max(p.x);
                        assert!(
                            wall_distance(p) < 1e-3,
                            "cluster {cluster:e} bump {bump}: {p:?}"
                        );
                        assert!((p.z - z0).abs() < 1e-3, "{p:?}");
                    }
                }
                extents.push((lowest, highest));
            }
            extents.sort_by(|a, b| a.0.total_cmp(&b.0));
            for ((lowest, highest), ends) in extents.iter().zip(corners.windows(2)) {
                assert!(
                    (lowest - ends[0].0).abs() < 1e-3 && (highest - ends[1].0).abs() < 1e-3,
                    "cluster {cluster:e} bump {bump}: piece [{lowest}, {highest}] for {ends:?}"
                );
            }
        }
    }

    #[test]
    fn test_marching_stops_on_the_model_size_cube() {
        // A cylinder whose axis is parallel to a generator of the cone, off
        // the apex: the intersection runs to infinity.
        use crate::{Cone, Cylinder};
        let tol = Tolerance::DEFAULT;
        let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.3, 1.0).unwrap();
        let generator = Vector3D::new(0.3f64.sin(), 0.0, 0.3f64.cos());
        let cylinder = Cylinder::new(Point3D::new(1.0, 0.0, 0.0), generator, 0.5).unwrap();
        let curves = marched_curves(&cone, &cylinder, tol);
        assert!(!curves.is_empty());
        let half = 0.5 * Tolerance::MODEL_SIZE;
        let on_cube = |p: Point3D| {
            [p.x, p.y, p.z]
                .iter()
                .any(|c| (c.abs() - half).abs() < 1e-3)
        };
        for curve in &curves {
            let (t0, t1) = curve.bounds();
            let ends = [curve.eval_point(t0), curve.eval_point(t1)];
            // Both branches of each curve run to infinity: every trace ends
            // on the cube, so none ran out of steps.
            assert!(ends.iter().all(|&p| on_cube(p)), "{ends:?}");
            assert!(curve.poles().len() < 2000, "{} poles", curve.poles().len());
        }
        assert_on_both(&curves, &cone, &cylinder);
    }

    #[test]
    fn test_marching_miss_is_empty() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(marched_curves(&plane, &sphere, tol).is_empty());
    }

    fn cylinder_patch(u_range: (f64, f64)) -> crate::surfaces::BSplineSurface {
        crate::Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0)
            .unwrap()
            .to_bspline_patch(u_range, (0.0, 2.0))
            .unwrap()
    }

    #[test]
    fn test_surface_and_its_bspline_form_are_coincident() {
        use crate::Sphere;
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::new(0.3, 0.2, 0.5), 0.8).unwrap();
        let copy = sphere.to_bspline();
        for result in [
            marching_intersection(&sphere, &copy, tol),
            marching_intersection(&copy, &sphere, tol),
            marching_intersection(&copy, &copy, tol),
        ] {
            assert!(result.is_coincident() && !result.is_empty());
            assert!(result.curves().is_empty());
            assert_eq!(result.completeness(), &Completeness::Sampled);
        }
    }

    /// The region of an incomplete result with one region.
    fn only_region(result: &SurfaceSurfaceIntersection) -> SurfaceSurfaceRegion {
        match result.completeness() {
            Completeness::Incomplete {
                reason: IncompleteReason::Unseparated,
                regions,
            } if regions.len() == 1 => regions[0],
            other => panic!("expected one unseparated region, got {other:?}"),
        }
    }

    #[test]
    fn test_patch_lying_on_its_surface_is_incomplete_over_the_patch() {
        use crate::{Cylinder, Plane};
        use std::f64::consts::{PI, TAU};
        let tol = Tolerance::DEFAULT;
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let whole = cylinder_patch((0.0, TAU));
        let result = marching_intersection(&cylinder, &whole, tol);
        assert!(!result.is_coincident() && result.curves().is_empty());
        let region = only_region(&result);
        // The patch spans the whole turn: the cylinder's range is the period.
        assert_eq!(region.first_u_range, (0.0, TAU));
        assert_eq!(region.first_v_range, (0.0, 2.0));
        assert_eq!(region.second_u_range, whole.u_bounds());
        // Swapped, the patch is the first surface.
        let region = only_region(&marching_intersection(&whole, &cylinder, tol));
        assert_eq!(region.first_u_range, whole.u_bounds());
        assert_eq!(region.second_u_range, (0.0, TAU));
        // A half turn keeps to its range on the cylinder.
        let half = cylinder_patch((0.0, PI));
        let region = only_region(&marching_intersection(&cylinder, &half, tol));
        assert!(region.first_u_range.0.abs() < 1e-9 && (region.first_u_range.1 - PI).abs() < 1e-9);
        // A plane against its own bilinear patch.
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let patch = plane.to_bspline_patch((-1.0, 2.0), (0.0, 3.0)).unwrap();
        let region = only_region(&marching_intersection(&plane, &patch, tol));
        assert_eq!(
            (region.first_u_range, region.first_v_range),
            ((-1.0, 2.0), (0.0, 3.0))
        );
        assert_eq!(region.min_corner, Point3D::new(-1.0, 0.0, 0.0));
        assert_eq!(region.max_corner, Point3D::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn test_crossing_surfaces_are_not_taken_for_coincident() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        // Crossing the patch through its first sample row: the check stops
        // at the first sample off the plane.
        let plane = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::new(1.0, 0.0, 1.0)).unwrap();
        let result =
            marching_intersection(&plane, &cylinder_patch((0.0, std::f64::consts::TAU)), tol);
        assert!(!result.is_coincident());
        assert_eq!(result.completeness(), &Completeness::Sampled);
        assert!(!result.curves().is_empty());
    }

    #[test]
    fn test_plane_touching_a_cylinder_patch_traces_a_tangent_ruling() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let patch = cylinder_patch((0.0, std::f64::consts::TAU));
        // Touching along x = 1, y = 0, and 1e-9 inside it, where the two
        // crossings lie 4.5e-5 apart at an angle of 4.5e-5 radians: a touch
        // within confusion.
        for offset in [1.0, 1.0 - 1e-9] {
            let plane = Plane::new(Point3D::new(offset, 0.0, 0.0), Vector3D::X).unwrap();
            let result = marching_intersection(&plane, &patch, tol);
            assert_eq!(result.curves().len(), 1);
            assert!(result.curves()[0].tangent);
            let Curve3D::BSpline(curve) = &result.curves()[0].curve else {
                panic!("a numeric curve is a B-spline");
            };
            let (t0, t1) = curve.bounds();
            let mut heights = [curve.eval_point(t0).z, curve.eval_point(t1).z];
            heights.sort_by(f64::total_cmp);
            assert!(heights[0].abs() < 1e-6 && (heights[1] - 2.0).abs() < 1e-6);
            for k in 0..=8 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 8.0);
                assert!((p.x - offset).abs() < 1e-7 && p.y.abs() < 1e-6, "{p:?}");
            }
        }
        // Crossing at 0.99, the two curves are not tangent.
        let plane = Plane::new(Point3D::new(0.99, 0.0, 0.0), Vector3D::X).unwrap();
        let result = marching_intersection(&plane, &patch, tol);
        assert_eq!(result.curves().len(), 2);
        assert!(result.curves().iter().all(|piece| !piece.tangent));
    }

    #[test]
    fn test_sphere_inside_a_cylinder_patch_traces_the_touching_circle() {
        use crate::{Sphere, Torus};
        let tol = Tolerance::DEFAULT;
        let patch = cylinder_patch((0.0, std::f64::consts::TAU));
        let sphere = Sphere::new(Point3D::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let torus = Torus::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z, 2.0, 1.0).unwrap();
        for result in [
            marching_intersection(&sphere, &patch, tol),
            marching_intersection(&torus, &patch, tol),
        ] {
            assert_eq!(result.curves().len(), 1);
            assert!(result.curves()[0].tangent);
            let Curve3D::BSpline(curve) = &result.curves()[0].curve else {
                panic!("a numeric curve is a B-spline");
            };
            let (t0, t1) = curve.bounds();
            assert!(curve.eval_point(t0).distance(curve.eval_point(t1)) < 1e-9);
            for k in 0..=16 {
                let p = curve.eval_point(t0 + (t1 - t0) * k as f64 / 16.0);
                let radius = (p.x * p.x + p.y * p.y).sqrt();
                assert!(
                    (radius - 1.0).abs() < 1e-6 && (p.z - 1.0).abs() < 1e-6,
                    "{p:?}"
                );
            }
        }
    }

    #[test]
    fn test_sphere_touching_a_cylinder_patch_at_a_point_is_a_contact_point() {
        use crate::Sphere;
        let tol = Tolerance::DEFAULT;
        let patch = cylinder_patch((0.0, std::f64::consts::TAU));
        let sphere = Sphere::new(Point3D::new(2.0, 0.0, 1.0), 1.0).unwrap();
        let result = marching_intersection(&sphere, &patch, tol);
        assert!(result.curves().is_empty());
        assert_eq!(result.contact_points().len(), 1);
        assert!(result.contact_points()[0].distance(Point3D::new(1.0, 0.0, 1.0)) < 1e-6);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_patches_overlapping_in_part_report_a_region_over_the_overlap() {
        let tol = Tolerance::DEFAULT;
        let first = cylinder_patch((0.0, 3.0));
        let second = cylinder_patch((1.5, 4.5));
        let result = marching_intersection(&first, &second, tol);
        assert!(!result.is_coincident());
        let Completeness::Incomplete { reason, regions } = result.completeness() else {
            panic!(
                "expected an incomplete result, got {:?}",
                result.completeness()
            );
        };
        assert_eq!(*reason, IncompleteReason::Unseparated);
        // The overlap is u in [1.5, 3] on both patches.
        assert!(
            regions
                .iter()
                .any(|region| region.first_u_range.0 <= 1.5 + 1e-6
                    && region.first_u_range.1 >= 3.0 - 1e-6)
        );
        assert!(result.contact_points().is_empty());
    }

    /// `surface` moved by `offset`.
    fn shifted(
        surface: &crate::surfaces::BSplineSurface,
        offset: Vector3D,
    ) -> crate::surfaces::BSplineSurface {
        surface.transformed(crate::Transform::translation(offset))
    }

    #[test]
    fn test_near_copies_within_confusion_are_coincident() {
        use crate::{Cylinder, Sphere};
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::new(0.3, 0.2, 0.5), 0.8).unwrap();
        let copy = sphere.to_bspline();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
        let patch = cylinder_patch((0.0, std::f64::consts::PI));
        for shift in [3e-8, 1e-7] {
            let offset = Vector3D::new(1.0, 0.5, 0.25).normalized().unwrap() * shift;
            let moved = shifted(&copy, offset);
            for result in [
                marching_intersection(&sphere, &moved, tol),
                marching_intersection(&moved, &sphere, tol),
                marching_intersection(&copy, &moved, tol),
            ] {
                assert!(result.is_coincident(), "shift {shift}");
                assert!(result.curves().is_empty());
            }
            let result = marching_intersection(&cylinder, &shifted(&patch, offset), tol);
            assert!(!result.is_coincident() && result.curves().is_empty());
            let region = only_region(&result);
            assert!(region.first_u_range.0.abs() < 1e-6);
            assert!((region.first_u_range.1 - std::f64::consts::PI).abs() < 1e-6);
        }
    }

    #[test]
    fn test_copies_apart_beyond_confusion_are_not_coincident() {
        use crate::Sphere;
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::new(0.3, 0.2, 0.5), 0.8).unwrap();
        let copy = sphere.to_bspline();
        // Equal spheres 1e-5 apart cross in one great circle.
        let offset = Vector3D::new(1.0, 0.5, 0.25).normalized().unwrap() * 1e-5;
        let result = marching_intersection(&sphere, &shifted(&copy, offset), tol);
        assert!(!result.is_coincident());
        assert_eq!(result.completeness(), &Completeness::Sampled);
        assert_eq!(result.curves().len(), 1);
    }

    /// A flat cubic patch over [0, 1] x [0, 1] in z = 0 with bumps of
    /// `height`, alternately up and down, between the lines u = k / 6:
    /// each bump is a single basis function of a uniform 24-span knot
    /// vector, with zero value and slope on those lines.
    fn bumpy_patch(height: f64) -> crate::surfaces::BSplineSurface {
        // A uniform cubic basis function peaks at 2/3 of its pole.
        let pole_height = 1.5 * height;
        let poles = (0..27)
            .map(|i| {
                let x = if i == 0 {
                    0.0
                } else if i == 26 {
                    1.0
                } else {
                    (i as f64 - 1.0) / 24.0
                };
                let z = if i % 4 == 3 && i <= 23 {
                    if (i / 4) % 2 == 0 {
                        pole_height
                    } else {
                        -pole_height
                    }
                } else {
                    0.0
                };
                vec![Point3D::new(x, 0.0, z), Point3D::new(x, 1.0, z)]
            })
            .collect();
        let knots = (0..=24).map(|k| k as f64 / 24.0).collect();
        let mut multiplicities = vec![1; 25];
        (multiplicities[0], multiplicities[24]) = (4, 4);
        crate::surfaces::BSplineSurface::new(
            3,
            1,
            poles,
            knots,
            multiplicities,
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_bumps_between_the_sample_lines_are_not_coincident() {
        use crate::Plane;
        let tol = Tolerance::DEFAULT;
        let flat = Plane::new(Point3D::ORIGIN, Vector3D::Z)
            .unwrap()
            .to_bspline_patch((0.0, 1.0), (0.0, 1.0))
            .unwrap();
        for height in [0.1, 1e-3] {
            let bumpy = bumpy_patch(height);
            for k in 0..6 {
                let peak = bumpy.eval_point((k as f64 + 0.5) / 6.0, 0.5);
                assert!((peak.z.abs() - height).abs() < 1e-9, "{peak:?}");
            }
            for result in [
                marching_intersection(&flat, &bumpy, tol),
                marching_intersection(&bumpy, &flat, tol),
            ] {
                assert!(!result.is_coincident());
                // The tall bumps touch the flat patch along the seven
                // lines; the low ones lie flat enough along them that
                // marching may leave part unseparated, but never takes
                // the two for one surface.
                if height == 0.1 {
                    assert_eq!(result.curves().len(), 7);
                    assert_eq!(result.completeness(), &Completeness::Sampled);
                }
            }
        }
    }
}
