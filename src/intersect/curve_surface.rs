//! Numeric curve-surface intersection for any parametric curve and
//! surface: sampled seeds refined by Newton iteration, or Bézier piece
//! pairs on B-splines.

use super::{CurveSurfaceHit, curve_surface_hull};
use crate::Tolerance;
use crate::curves::ParametricCurve3D;
use crate::math::{brent_minimum, newton_3d};
use crate::surfaces::ParametricSurface;

/// Generic curve-surface intersection for any parametric curve and surface.
///
/// Curve samples projected onto the surface seed Newton iteration on
/// `C(t) - S(u, v) = 0` (first derivatives only); converged roots with a
/// residual within `tol.confusion` report, ordered by curve parameter.
/// Where Newton fails from a sample that is a local minimum of the
/// distance to the surface, the distance from `C(t)` to the surface is
/// minimized over `t` alone between the neighbouring samples. That finds
/// hits at singular points of the surface parametrization (a sphere
/// pole, a cone apex, a collapsed B-spline edge), where the Newton system
/// is singular; their `surface_params` are the projection's.
/// Unbounded curves seed `[-10, 10]`. Overlapping pairs yield dense hit
/// sets along the overlap — use the exact intersection
/// ([`Intersect::intersect`](crate::Intersect::intersect)) when
/// classification (tangent, coincident) matters.
///
/// A B-spline curve, or a trimmed curve over one (see
/// [`ParametricCurve3D::as_bspline`]), against a B-spline surface is
/// searched without sampling: the curve's knot spans over its range and
/// the surface's knot-span patches are Bézier pieces, and the pairs of
/// pieces whose control points' boxes overlap are halved until each
/// provably holds at most one hit, which Newton iteration from the pair's
/// centre then reaches. Parameters are reported within the ranges (a
/// periodic one within its period). A pair that never separates (a
/// tangency, an overlap, a hit at a singular point) exhausts the search's
/// budget, and the samples above are used instead.
///
/// # Examples
///
/// ```
/// use geomcore::{Line3D, Point3D, Sphere, Tolerance, Vector3D, intersect_curve_surface};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
/// let hits = intersect_curve_surface(&line, &sphere, Tolerance::DEFAULT);
/// assert_eq!(hits.len(), 2);
/// assert!((hits[0].curve_param + 2.0).abs() < 1e-9);
/// assert!((hits[1].curve_param - 2.0).abs() < 1e-9);
/// ```
pub fn intersect_curve_surface<C, S>(curve: &C, surface: &S, tol: Tolerance) -> Vec<CurveSurfaceHit>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    if let (Some(bspline), Some(patches)) = (curve.as_bspline(), surface.as_bspline()) {
        let periodic = curve.period().is_some();
        let hits = curve_surface_hull::bspline_curve_surface(
            bspline,
            curve.bounds(),
            periodic,
            patches,
            tol,
        );
        if let Some(hits) = hits {
            return hits;
        }
    }
    sampled_curve_surface(curve, surface, tol)
}

/// The sampled search of [`intersect_curve_surface`].
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
                        curve_param: rt,
                        surface_params: (ru, rv),
                        point: p,
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
                        curve_param: rt,
                        surface_params: (proj.u, proj.v),
                        point: p,
                    },
                );
            }
        }
    }
    hits.sort_by(|a, b| a.curve_param.partial_cmp(&b.curve_param).unwrap());
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Intersect, Line3D, LineQuadricIntersection, Point3D, Sphere, Vector3D};

    #[test]
    fn test_generic_line_sphere_matches_exact() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let hits = intersect_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].curve_param + 2.0).abs() < 1e-9);
        assert!((hits[1].curve_param - 2.0).abs() < 1e-9);
        // Cross-check against the exact intersection.
        match line.intersect(&sphere, tol) {
            LineQuadricIntersection::TwoPoints((t1, _), (t2, _)) => {
                assert!((hits[0].curve_param - t1).abs() < 1e-9);
                assert!((hits[1].curve_param - t2).abs() < 1e-9);
            }
            _ => panic!("exact intersection expected two points"),
        }
    }

    #[test]
    fn test_generic_line_sphere_miss() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(intersect_curve_surface(&line, &sphere, tol).is_empty());
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
                    .eval_point(hit.surface_params.0, hit.surface_params.1)
                    .distance(hit.point)
                    <= 1e-6
        };

        // A line along the sphere's axis crosses at both poles.
        let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
        let line = Line3D::new(Point3D::new(0.0, 0.0, -3.0), Vector3D::Z).unwrap();
        let hits = intersect_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].curve_param - 2.0).abs() < 1e-9, "{hits:?}");
        assert!((hits[1].curve_param - 4.0).abs() < 1e-9, "{hits:?}");
        assert!(hits.iter().all(|h| on_both(h, &sphere)));

        // A line through a cone's apex, square to the axis, meets it there
        // only.
        let cone = Cone::new(Point3D::new(0.0, 0.0, -1.0), Vector3D::Z, 0.5, 0.5).unwrap();
        let apex = cone.apex();
        let line = Line3D::new(apex - Vector3D::X * 3.0, Vector3D::X).unwrap();
        let hits = intersect_curve_surface(&line, &cone, tol);
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
        let hits = intersect_curve_surface(&curve, &sphere, tol);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].curve_param - 0.5).abs() < 1e-9, "{hits:?}");
        assert!(hits.iter().all(|h| on_both(h, &sphere)));
    }

    #[test]
    fn test_generic_segment_plane_hit() {
        use crate::{BSplineCurve3D, Plane};
        let tol = Tolerance::DEFAULT;
        let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)];
        let curve = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
        let plane = Plane::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::X).unwrap();
        let hits = intersect_curve_surface(&curve, &plane, tol);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].curve_param - 0.5).abs() < 1e-9);
        assert_eq!(hits[0].point, Point3D::new(1.0, 0.0, 0.0));
    }
}
