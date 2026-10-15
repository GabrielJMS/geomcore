//! Numeric curve-surface intersection for any parametric curve and
//! surface: sampled seeds refined by Newton iteration, or Bézier piece
//! pairs on B-splines.

use super::curve_surface_hull::CurveSurfaceHits;
use super::results::sort_by_known_parameter;
use super::{
    CurveSurfaceHit, CurveSurfaceIntersection, curve_surface_multiplicity, mixed, overlap_runs,
};
use crate::curves::ParametricCurve3D;
use crate::math::{brent_minimum, newton_3d};
use crate::surfaces::ParametricSurface;
use crate::{Point3D, Tolerance};

/// Intersects any parametric curve with any parametric surface by the
/// numeric search: the entry point for curve and surface types defined
/// outside this crate.
///
/// For the crate's own types, call [`Intersect::intersect`](crate::Intersect::intersect)
/// instead: it solves the pairs with a closed form exactly, and runs this
/// same search for the others. This function always searches, so its
/// result's [`method`](CurveSurfaceIntersection::method) is
/// [`Numeric`](crate::IntersectionMethod::Numeric), its
/// [`completeness`](CurveSurfaceIntersection::completeness) says whether
/// every hit was found. Where the curve lies on the surface, the hits and
/// regions the search leaves along it are joined into an overlap, and the
/// result stays [`Incomplete`](crate::Completeness::Incomplete) with an
/// `Unseparated` region over it (see
/// [`CurveSurfaceIntersection::overlaps`]).
///
/// # Examples
///
/// A helix, defined by the caller, crossing a plane:
///
/// ```
/// use geomcore::curves::ParametricCurve3D;
/// use geomcore::{IntersectionMethod, Plane, Point3D, Tolerance, Vector3D, intersect_curve_surface};
///
/// /// A helix of radius 1 about the z axis, rising 1 per turn.
/// struct Helix;
///
/// impl ParametricCurve3D for Helix {
///     fn eval_point(&self, u: f64) -> Point3D {
///         let turn = std::f64::consts::TAU;
///         Point3D::new(u.cos(), u.sin(), u / turn)
///     }
///     fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
///         let turn = std::f64::consts::TAU;
///         match order {
///             1 => Vector3D::new(-u.sin(), u.cos(), 1.0 / turn),
///             _ => Vector3D::new(-u.cos(), -u.sin(), 0.0),
///         }
///     }
///     fn bounds(&self) -> (f64, f64) {
///         (0.0, 3.0 * std::f64::consts::TAU)
///     }
///     fn period(&self) -> Option<f64> {
///         None
///     }
/// }
///
/// let plane = Plane::new(Point3D::new(0.0, 0.0, 1.5), Vector3D::Z).unwrap();
/// let result = intersect_curve_surface(&Helix, &plane, Tolerance::DEFAULT);
/// assert_eq!(result.method(), IntersectionMethod::Numeric);
/// assert_eq!(result.hits().len(), 1);
/// assert!((result.hits()[0].curve_parameter.unwrap() - 3.0 * std::f64::consts::PI).abs() < 1e-9);
/// ```
pub fn intersect_curve_surface<C, S>(
    curve: &C,
    surface: &S,
    tol: Tolerance,
) -> CurveSurfaceIntersection
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    let hits = search_curve_surface(curve, surface, tol);
    let completeness = hits.completeness().clone();
    let (hits, overlaps, completeness) =
        overlap_runs::curve_surface_overlaps(curve, surface, hits.into_hits(), completeness, tol);
    CurveSurfaceIntersection::numeric(hits, overlaps, completeness)
}

/// Generic curve-surface intersection for any parametric curve and surface.
///
/// Curve samples projected onto the surface seed Newton iteration on
/// `C(t) - S(u, v) = 0` (first derivatives only); converged roots with a
/// residual within `tol.confusion` report, ordered by curve parameter.
/// Where Newton fails from a sample that is a local minimum of the
/// distance to the surface, the distance from `C(t)` to the surface is
/// minimized over `t` alone between the neighbouring samples. That finds
/// hits at singular points of the surface parametrization (a sphere
/// pole, a cone apex, a collapsed B-spline boundary), where the Newton system
/// is singular; their `surface_parameters` are the projection's.
/// Unbounded curves seed `[-10, 10]`. Overlapping pairs yield dense hit
/// sets along the overlap; the routed [`Intersect`](crate::Intersect)
/// calls this search only for pairs without a closed form, and solves the
/// others exactly, tangency and coincidence included.
///
/// The hits are [`Completeness::Sampled`](crate::Completeness::Sampled):
/// a hit between samples, or whose Newton iteration strays, can be
/// missed.
///
/// A B-spline curve, or a trimmed curve over one (see
/// [`ParametricCurve3D::as_bspline`]), against a B-spline surface is
/// searched without sampling: the curve's knot spans over its range and
/// the surface's knot-span patches are Bézier pieces, and the pairs of
/// pieces whose control points' boxes overlap are halved until each is
/// proven apart or provably holds at most one hit, which Newton iteration
/// from the pair's centre then reaches. So is a B-spline against an
/// elementary shape (see [`ParametricCurve3D::as_elementary`] and
/// [`ParametricSurface::as_elementary`]), either way round, converted to
/// its exact rational B-spline. A hit lies on the B-spline, so within
/// the box of its control points: a circle, ellipse, sphere or torus is
/// converted whole, a trimmed curve over its range, and a line,
/// parabola, hyperbola, plane, cylinder or cone over the part inside
/// that box grown by a margin. A cone whose apex lies in it becomes two
/// patches, one per nappe, each ending on the apex. Parameters are
/// reported on the shapes themselves, within their ranges (a periodic
/// one within its period). When every pair settles, the hits are
/// [`Completeness::Complete`](crate::Completeness::Complete). A pair that
/// never separates (a tangency, an overlap, a hit at a singular point),
/// or a search that runs out of its budget of halvings, leaves them
/// [`Completeness::Incomplete`](crate::Completeness::Incomplete), with
/// the reason and the regions where hits may be missing; the hits are
/// then those found, together with the sampled search's hits that do not
/// merge into them.
pub(crate) fn search_curve_surface<C, S>(curve: &C, surface: &S, tol: Tolerance) -> CurveSurfaceHits
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    if let Some(mut hits) = mixed::curve_surface(curve, surface, tol) {
        if !hits.is_complete() {
            hits.add_sampled(sampled_curve_surface(curve, surface, tol), tol);
        }
        return hits;
    }
    CurveSurfaceHits::sampled(sampled_curve_surface(curve, surface, tol))
}

/// The sampled search of [`search_curve_surface`].
pub(super) fn sampled_curve_surface<C, S>(
    curve: &C,
    surface: &S,
    tol: Tolerance,
) -> Vec<CurveSurfaceHit>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    let (t0, t1) = curve.bounds();
    // Unbounded curves seed a documented default window.
    const WINDOW: f64 = 10.0;
    let (lo, hi) = (
        if t0.is_finite() { t0 } else { -WINDOW },
        if t1.is_finite() { t1 } else { WINDOW },
    );
    const SEEDS: usize = 128;
    const KEEP: usize = 24;
    let step = (hi - lo) / (SEEDS - 1) as f64;
    let samples: Vec<(f64, f64)> = (0..SEEDS)
        .map(|i| {
            let t = lo + step * i as f64;
            (t, surface.project_point(curve.eval_point(t), tol).distance)
        })
        .collect();
    let local_min = |i: usize| {
        (i == 0 || samples[i].1 <= samples[i - 1].1)
            && (i + 1 == SEEDS || samples[i].1 <= samples[i + 1].1)
    };
    let mut order: Vec<usize> = (0..SEEDS).collect();
    order.sort_by(|&a, &b| samples[a].1.total_cmp(&samples[b].1));
    let mut hits: Vec<CurveSurfaceHit> = Vec::new();
    // Contact-level dedupe (sqrt scale): near-tangent seeds reach distinct
    // points of one grazing contact; transversal hits are far apart by
    // comparison. Extrema (stationary points) keep tight dedupe elsewhere
    // — distinct minima are real data.
    let merge_gap = tol.confusion.sqrt();
    let push = |hits: &mut Vec<CurveSurfaceHit>, hit: CurveSurfaceHit| {
        if !hits
            .iter()
            .any(|h| h.point.distance(hit.point) <= merge_gap)
        {
            hits.push(hit);
        }
    };
    let multiplicity_at = |t: f64, u: f64, v: f64, point: Point3D| {
        let (_, tangent) = curve.eval_d1(t);
        let (_, [su, sv]) = surface.eval_d1(u, v);
        curve_surface_multiplicity(tangent, su.cross(sv), point, tol)
    };
    for i in order.into_iter().take(KEEP) {
        let t = samples[i].0;
        let proj = surface.project_point(curve.eval_point(t), tol);
        let f = |x: [f64; 3]| {
            let d = curve.eval_point(x[0]) - surface.eval_point(x[1], x[2]);
            [d.x, d.y, d.z]
        };
        let jac = |x: [f64; 3]| {
            let (_, ct) = curve.eval_d1(x[0]);
            let (_, [su, sv]) = surface.eval_d1(x[1], x[2]);
            [
                [ct.x, -su.x, -sv.x],
                [ct.y, -su.y, -sv.y],
                [ct.z, -su.z, -sv.z],
            ]
        };
        if let Some([rt, ru, rv]) = newton_3d(f, jac, [t, proj.u, proj.v], tol.confusion, 50) {
            let p = curve.eval_point(rt);
            if p.distance(surface.eval_point(ru, rv)) <= tol.confusion {
                push(
                    &mut hits,
                    CurveSurfaceHit {
                        curve_parameter: Some(rt),
                        surface_parameters: Some((ru, rv)),
                        point: p,
                        multiplicity: multiplicity_at(rt, ru, rv, p),
                    },
                );
                continue;
            }
        }
        if local_min(i) {
            // The distance to the projected point, not the projection's
            // own, which snaps to zero within `tol.confusion` and would
            // leave the minimum anywhere in that band.
            let distance = |t: f64| {
                let p = curve.eval_point(t);
                let proj = surface.project_point(p, tol);
                p.distance(surface.eval_point(proj.u, proj.v))
            };
            let (a, b) = ((t - step).max(lo), (t + step).min(hi));
            let rt = brent_minimum(distance, a, b, 1e-13);
            let p = curve.eval_point(rt);
            let proj = surface.project_point(p, tol);
            if proj.distance <= tol.confusion {
                push(
                    &mut hits,
                    CurveSurfaceHit {
                        curve_parameter: Some(rt),
                        surface_parameters: Some((proj.u, proj.v)),
                        point: p,
                        multiplicity: multiplicity_at(rt, proj.u, proj.v, p),
                    },
                );
            }
        }
    }
    sort_by_known_parameter(&mut hits, |hit| hit.curve_parameter);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntersectExact, Line3D, LineQuadricIntersection, Point3D, Sphere, Vector3D};

    #[test]
    fn test_generic_line_sphere_matches_exact() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let hits = search_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].curve_parameter.unwrap() + 2.0).abs() < 1e-9);
        assert!((hits[1].curve_parameter.unwrap() - 2.0).abs() < 1e-9);
        // Cross-check against the exact intersection.
        match line.intersect_exact(&sphere, tol) {
            LineQuadricIntersection::TwoPoints((t1, _), (t2, _)) => {
                assert!((hits[0].curve_parameter.unwrap() - t1).abs() < 1e-9);
                assert!((hits[1].curve_parameter.unwrap() - t2).abs() < 1e-9);
            }
            _ => panic!("exact intersection expected two points"),
        }
    }

    #[test]
    fn test_generic_line_sphere_miss() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(search_curve_surface(&line, &sphere, tol).is_empty());
    }

    /// Hits where the surface parametrization is singular (`Su × Sv = 0`):
    /// the Newton system on `(t, u, v)` is singular there.
    #[test]
    fn test_generic_curve_surface_hits_at_singular_points() {
        use crate::{BSplineCurve3D, Cone};
        let tol = Tolerance::DEFAULT;
        let on_both = |hit: &CurveSurfaceHit, surface: &dyn ParametricSurface| {
            surface.contains(hit.point, tol)
                && surface
                    .eval_point(
                        hit.surface_parameters.unwrap().0,
                        hit.surface_parameters.unwrap().1,
                    )
                    .distance(hit.point)
                    <= 1e-6
        };

        // A line along the sphere's axis crosses at both poles.
        let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        let line = Line3D::new(Point3D::new(0.0, 0.0, -3.0), Vector3D::Z).unwrap();
        let hits = search_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!(
            (hits[0].curve_parameter.unwrap() - 2.0).abs() < 1e-9,
            "{hits:?}"
        );
        assert!(
            (hits[1].curve_parameter.unwrap() - 4.0).abs() < 1e-9,
            "{hits:?}"
        );
        assert!(hits.iter().all(|h| on_both(h, &sphere)));

        // A line through a cone's apex, square to the axis, meets it there
        // only.
        let cone = Cone::new(Point3D::new(0.0, 0.0, -1.0), Vector3D::Z, 0.5, 0.5).unwrap();
        let apex = cone.apex();
        let line = Line3D::new(apex - Vector3D::X * 3.0, Vector3D::X).unwrap();
        let hits = search_curve_surface(&line, &cone, tol);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(hits[0].point.distance(apex) < 1e-6, "{hits:?}");
        assert!(on_both(&hits[0], &cone));

        // A B-spline curve, which has no exact solver, crossing the sphere
        // at its north pole (t = 0.5) and once more on the way out.
        let poles = vec![
            Point3D::new(-1.0, 0.0, 3.0),
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 1.0),
        ];
        let curve = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
        assert_eq!(curve.eval_point(0.5), Point3D::new(0.0, 0.0, 1.0));
        let hits = search_curve_surface(&curve, &sphere, tol);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!(
            (hits[0].curve_parameter.unwrap() - 0.5).abs() < 1e-9,
            "{hits:?}"
        );
        assert!(hits.iter().all(|h| on_both(h, &sphere)));
    }

    #[test]
    fn test_generic_segment_plane_hit() {
        use crate::{BSplineCurve3D, Plane};
        let tol = Tolerance::DEFAULT;
        let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        let plane = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
        let hits = search_curve_surface(&curve, &plane, tol);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].curve_parameter.unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(hits[0].point, Point3D::new(1.0, 0.0, 0.0));
    }
}
