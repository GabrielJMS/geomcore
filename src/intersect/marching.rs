//! Numeric surface-surface intersection by marching: seeds from a
//! parameter grid, traced along the intersection and fitted.

use crate::curves::{BSplineCurve3D, InterpParametrization};
use crate::math::{solve_2x2, solve_3x3, solve_4x4};
use crate::surfaces::ParametricSurface;
use crate::{Point3D, Tolerance, Vector3D};

/// Marching surface-surface intersection for any two parametric surfaces.
///
/// Seeds come from both parameter grids: the samples where the distance
/// to the other surface is a local minimum are converged onto the
/// intersection by Newton's method on `S1(u1, v1) = S2(u2, v2)`, so curves
/// running between grid lines are found too. Each seed traces both ways
/// along the `N1 x N2` tangent; every step is corrected by Newton's method
/// on the same system plus the plane through the predicted point normal to
/// the tangent, and the step length follows the curvature (a chord
/// deviation of about `1000 * confusion`, at most 0.1). A trace ends when
/// it closes a loop, reaches a parameter bound, or stalls. Traces fit
/// [`BSplineCurve3D`] interpolants (degree 3, closed loops by repeated end
/// points).
///
/// The fits are APPROXIMATIONS (interpolation error, typically ~1e-4 at
/// default tolerance — assert residuals, not containment): exact
/// classification stays with the exact intersection
/// ([`Intersect::intersect`](crate::Intersect::intersect)). Likewise
/// no tangency classification (grazing pairs may trace partially or not
/// at all), no marching through singular points (poles, apexes), and a
/// fixed seed grid (a loop that leaves no distance minimum on either grid
/// is missed) — v1 heuristics, documented so callers can judge fit.
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
    let mut seeds = grid_seeds(a, b, tol);
    seeds.extend(
        grid_seeds(b, a, tol)
            .into_iter()
            .map(|x| [x[2], x[3], x[0], x[1]]),
    );
    let mut curves: Vec<BSplineCurve3D> = Vec::new();
    // Seeds on an already traced curve are suppressed by projection onto
    // its fit (the fit error exceeds confusion, so tight containment
    // cannot decide duplication). Near-tangent distinct curves (closer
    // than this) are out of scope.
    let suppress = 1000.0 * tol.confusion;
    for seed in seeds {
        let p = a.eval_point(seed[0], seed[1]);
        if curves
            .iter()
            .any(|c| c.project_point(p, tol).distance <= suppress)
        {
            continue;
        }
        if let Some(points) = trace(a, b, seed, tol) {
            let degree = 3.min(points.len() - 1);
            if let Ok(fit) =
                BSplineCurve3D::interpolate(&points, degree, InterpParametrization::Centripetal)
            {
                curves.push(fit);
            }
        }
    }
    curves
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

/// Trace the intersection curve through `seed` both ways. Returns its
/// points in order (a closed loop ends on its first point), or `None`
/// when fewer than four were traced.
fn trace<A, B>(a: &A, b: &B, seed: [f64; 4], tol: Tolerance) -> Option<Vec<Point3D>>
where
    A: ParametricSurface,
    B: ParametricSurface,
{
    // Chord deviation allowed per step, and the longest step.
    let chord_deviation = 1000.0 * tol.confusion;
    const H_MAX: f64 = 0.1;
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
        for _ in 0..2000 {
            // Predict along the tangent, correct onto both surfaces in the
            // plane normal to it; halve the step until the result stays on
            // this branch with a small chord deviation.
            let mut step = None;
            while h >= tol.confusion {
                let target = p + t * h;
                if let Some(y) = meet(a, b, predict(a, b, x, t * h), Some((target, t)), tol)
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
    let clamp = |s: &dyn ParametricSurface, u: f64, v: f64| {
        let (u0, u1) = s.u_bounds();
        let (v0, v1) = s.v_bounds();
        (
            if s.u_period().is_none() {
                u.clamp(u0, u1)
            } else {
                u
            },
            if s.v_period().is_none() {
                v.clamp(v0, v1)
            } else {
                v
            },
        )
    };
    let bounded = |x: [f64; 4]| {
        let (u1, v1) = clamp(a, x[0], x[1]);
        let (u2, v2) = clamp(b, x[2], x[3]);
        [u1, v1, u2, v2]
    };
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
    fn test_marching_miss_is_empty() {
        use crate::{Plane, Sphere};
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::Z).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(marching_intersection(&plane, &sphere, tol).is_empty());
    }
}
