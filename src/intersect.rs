//! Analytic intersection result types.
//!
//! The `intersect_*` methods on the analytic curves and surfaces return
//! these enums: the intersecting geometry on success, or an explicit
//! classification when the pair is tangent, disjoint, parallel, or
//! coincident. Tangency and coincidence are decided with
//! [`crate::Tolerance`], so near-degenerate configurations classify
//! robustly instead of collapsing to noise.

use crate::curves::ParametricCurve3D;
use crate::math::newton_3d;
use crate::surfaces::ParametricSurface;
use crate::{Circle3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D, Point2D, Point3D, Tolerance};

/// Result of intersecting two planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanePlaneIntersection {
    /// The planes meet in a line.
    Line(Line3D),
    /// The planes are parallel but distinct.
    Parallel,
    /// The planes coincide.
    Coincident,
}

/// Result of intersecting a plane with a sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneSphereIntersection {
    /// The plane cuts the sphere in a circle.
    Circle(Circle3D),
    /// The plane grazes the sphere in a single point.
    TangentPoint(Point3D),
    /// The plane misses the sphere.
    Empty,
}

/// Result of intersecting two spheres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereSphereIntersection {
    /// The spheres meet in a circle.
    Circle(Circle3D),
    /// The spheres touch in a single point.
    TangentPoint(Point3D),
    /// The spheres are separate, one inside the other without contact, or
    /// concentric with different radii.
    Empty,
    /// The spheres coincide (same center and radius within tolerance).
    Coincident,
}

/// Result of intersecting a plane with a cylinder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneCylinderIntersection {
    /// The plane is perpendicular to the axis: a circle.
    Circle(Circle3D),
    /// The general oblique section: an ellipse.
    Ellipse(Ellipse3D),
    /// The plane is parallel to the axis and cuts two generators.
    TwoLines(Line3D, Line3D),
    /// The plane is parallel to the axis and grazes one generator.
    TangentLine(Line3D),
    /// The plane is parallel to the axis and misses the cylinder.
    Empty,
}

/// Result of intersecting a plane with a cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneConeIntersection {
    /// The plane is perpendicular to the axis: a circle.
    Circle(Circle3D),
    /// The plane meets the cone in an ellipse.
    Ellipse(Ellipse3D),
    /// The plane is parallel to a generator: a parabola.
    Parabola(Parabola3D),
    /// The plane cuts both nappes' worth of angle: a hyperbola branch.
    Hyperbola(Hyperbola3D),
    /// The plane passes through the apex and cuts two generators.
    TwoLines(Line3D, Line3D),
    /// The plane passes through the apex along a single generator.
    TangentLine(Line3D),
    /// The plane passes through the apex without meeting a generator.
    ApexPoint(Point3D),
    /// The section lies entirely behind the apex, off the single nappe.
    Empty,
}

/// Result of intersecting a line with a plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinePlaneIntersection {
    /// Transversal hit: the line parameter and the point.
    Point(f64, Point3D),
    /// The line is parallel to the plane but offset from it.
    Parallel,
    /// The line lies in the plane.
    Coincident,
}

/// Result of intersecting a line with a sphere, cylinder, or cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineQuadricIntersection {
    /// Two transversal hits: `(parameter, point)` pairs, ordered.
    TwoPoints((f64, Point3D), (f64, Point3D)),
    /// A single transversal hit: the linear case (line parallel to a
    /// cylinder axis or cone generator) or a nappe-filtered root.
    OnePoint(f64, Point3D),
    /// Grazing contact: the line parameter and the point.
    Tangent(f64, Point3D),
    /// No intersection.
    Empty,
    /// The line lies entirely on the surface (a cylinder or cone
    /// generator; a line in a plane reports through
    /// [`LinePlaneIntersection::Coincident`] instead).
    Coincident,
}

/// Result of intersecting a sphere with a cylinder.
///
/// The analytic path requires the cylinder axis through the sphere
/// center (within `tol.confusion`); otherwise the section is a space
/// quartic and reports [`SphereCylinderIntersection::NotAnalytic`],
/// reserved for the numeric surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereCylinderIntersection {
    /// Latitude circle(s): one tangent ring or a symmetric pair.
    Circle(Circle3D),
    /// Two latitude rings symmetric about the equatorial plane.
    TwoCircles(Circle3D, Circle3D),
    /// The cylinder misses the sphere.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a sphere with a cone.
///
/// The analytic path requires the sphere center on the cone axis
/// (within `tol.confusion`); otherwise the section is a space quartic
/// and reports [`SphereConeIntersection::NotAnalytic`], reserved for the
/// numeric surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereConeIntersection {
    /// Latitude circle on the nappe.
    Circle(Circle3D),
    /// Two latitude circles on the nappe.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing contact ring on the nappe.
    TangentCircle(Circle3D),
    /// No latitude circle lies on the single nappe.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two cylinders.
///
/// The analytic path requires parallel axes (within `tol.angular`);
/// skew or crossing axes give a space quartic and report
/// [`CylinderCylinderIntersection::NotAnalytic`], reserved for the
/// numeric surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CylinderCylinderIntersection {
    /// Two shared generators.
    TwoLines(Line3D, Line3D),
    /// One shared grazing generator.
    TangentLine(Line3D),
    /// Separate, nested without contact, or concentric with different radii.
    Empty,
    /// Same axis and radius within tolerance.
    Coincident,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a cone with a cylinder.
///
/// The analytic path needs coaxial axes (within tolerance): the tube of
/// radius `r` meets the nappe at axial `r/tan(phi)`, always a single
/// latitude circle (or the apex for a degenerate tube). Anything else is
/// a space quartic and reports
/// [`ConeCylinderIntersection::NotAnalytic`], reserved for the numeric
/// surface-surface path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConeCylinderIntersection {
    /// Latitude circle on the nappe.
    Circle(Circle3D),
    /// Degenerate tube (its axis) meeting the nappe at the apex.
    ApexPoint(Point3D),
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a plane.
///
/// The analytic path needs the plane through the axis (two meridian
/// circles) or perpendicular to it (latitude circles). Anything else is
/// a quartic spiric section and reports
/// [`TorusPlaneIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusPlaneIntersection {
    /// Two meridian circles (axis-containing plane).
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring (perpendicular plane at tube top/bottom).
    TangentCircle(Circle3D),
    /// Single latitude ring (spindle case).
    Circle(Circle3D),
    /// The plane misses the torus.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a sphere.
///
/// The analytic path needs the sphere center on the torus axis, reducing
/// to a quadratic in the axial coordinate. Anything else is high-degree
/// and reports [`TorusSphereIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusSphereIntersection {
    /// Latitude ring.
    Circle(Circle3D),
    /// Two latitude rings.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring.
    TangentCircle(Circle3D),
    /// No ring.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a cylinder.
///
/// The analytic path needs coaxial axes. Anything else reports
/// [`TorusCylinderIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusCylinderIntersection {
    /// Two latitude rings symmetric about the equatorial plane.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring in the equatorial plane.
    TangentCircle(Circle3D),
    /// The tube misses the torus.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting a torus with a cone.
///
/// The analytic path needs coaxial axes, reducing to a quadratic in the
/// axial coordinate with nappe filtering. Anything else reports
/// [`TorusConeIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusConeIntersection {
    /// Latitude ring on the nappe.
    Circle(Circle3D),
    /// Two latitude rings on the nappe.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring on the nappe.
    TangentCircle(Circle3D),
    /// No ring on the single nappe.
    Empty,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two tori.
///
/// The analytic path needs collinear axes, reducing to quadratics in the
/// axial coordinate. Anything else (up to degree 16) reports
/// [`TorusTorusIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TorusTorusIntersection {
    /// Latitude ring.
    Circle(Circle3D),
    /// Two latitude rings.
    TwoCircles(Circle3D, Circle3D),
    /// Grazing ring.
    TangentCircle(Circle3D),
    /// No rings.
    Empty,
    /// Same center and radii within tolerance.
    Coincident,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two cones.
///
/// The analytic path needs coaxial axes: equal latitude-ring radii give
/// one axial position (or apex/coincident degenerates). Anything else is
/// a space quartic and reports [`ConeConeIntersection::NotAnalytic`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConeConeIntersection {
    /// Latitude ring on both nappes.
    Circle(Circle3D),
    /// The rings collapse to the shared apex.
    ApexPoint(Point3D),
    /// No ring on the nappes (parallel distinct nappes, or valid roots
    /// behind an apex).
    Empty,
    /// Same apex, axis, and semi-angle within tolerance.
    Coincident,
    /// No closed form: the pair is not in analytic configuration.
    NotAnalytic,
}

/// Result of intersecting two 2D lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineLine2DIntersection {
    /// Transversal hit: line parameter on self, point, line parameter other.
    Point(f64, Point2D, f64),
    /// Parallel distinct lines.
    Parallel,
    /// Coincident lines.
    Coincident,
}

/// Result of intersecting a 2D line with a 2D circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineCircle2DIntersection {
    /// Two hits: `(line parameter, point)` pairs, ordered.
    Points((f64, Point2D), (f64, Point2D)),
    /// Grazing contact: line parameter and point.
    Tangent(f64, Point2D),
    /// No intersection.
    Empty,
}

/// Result of intersecting two 2D circles.
///
/// Parameters are intentionally omitted (no natural primary curve);
/// recover them with [`Circle2D::parameter_of`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircleCircle2DIntersection {
    /// Two meeting points.
    Points(Point2D, Point2D),
    /// Grazing contact point.
    Tangent(Point2D),
    /// Separate, nested, or concentric with different radii.
    Empty,
    /// Same center and radius within tolerance.
    Coincident,
}

/// Result of intersecting two 3D lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineLine3DIntersection {
    /// Transversal hit: parameter on self, point, parameter on other.
    Point(f64, Point3D, f64),
    /// Parallel distinct lines, with their distance.
    Parallel(f64),
    /// Coincident lines.
    Coincident,
    /// Skew lines: parameters and distance of the closest pair.
    Skew {
        /// Parameter on the first line.
        s: f64,
        /// Parameter on the second line.
        t: f64,
        /// Distance between the closest points.
        distance: f64,
    },
}

/// Result of intersecting a 3D line with a 3D circle.
///
/// The line meets the circle's plane in at most one point (transversal
/// graze or miss); a line lying in the plane reduces to the 2D problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineCircle3DIntersection {
    /// Two hits: `(line parameter, point)` pairs, ordered.
    Points((f64, Point3D), (f64, Point3D)),
    /// Grazing contact: line parameter and point.
    Tangent(f64, Point3D),
    /// No intersection.
    Empty,
}

/// Result of intersecting two 3D circles.
///
/// Only coplanar pairs admit a closed form (via 2D images in the shared
/// plane); anything else reports [`CircleCircle3DIntersection::NotAnalytic`].
/// Parameters are omitted (no natural primary curve); recover them with
/// [`Circle3D::parameter_of`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircleCircle3DIntersection {
    /// Two meeting points.
    Points(Point3D, Point3D),
    /// Grazing contact point.
    Tangent(Point3D),
    /// No intersection.
    Empty,
    /// Same center, radius, and plane within tolerance.
    Coincident,
    /// No closed form: the circles are not coplanar.
    NotAnalytic,
}

/// Solution classification for `a*t^2 + b*t + c = 0` with tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QuadraticSolution {
    /// Two distinct roots, ordered.
    Two(f64, f64),
    /// A double root (discriminant within tolerance of zero).
    One(f64),
    /// Degenerate linear equation with a single root.
    Linear(f64),
    /// Degenerate `0 = 0` (caller decides coincidence separately).
    Degenerate,
    /// No (real) roots.
    Empty,
}

/// Solve `a*t^2 + b*t + c = 0`, classifying with `tol`.
///
/// Near-zero `a` (against `tol.angular`, the coefficients being unit
/// scale for unit-direction lines) falls back to the linear equation.
/// Otherwise the root half-separation `sqrt(|disc|) / (2|a|)` decides:
/// separations within `tol.confusion * (1 + |t0|)` of the double root
/// report grazing contact, wider positive discriminants two ordered
/// roots, wider negative ones empty.
pub(crate) fn solve_quadratic(a: f64, b: f64, c: f64, tol: Tolerance) -> QuadraticSolution {
    if a.abs() <= tol.angular {
        if b.abs() <= tol.confusion {
            return QuadraticSolution::Degenerate;
        }
        return QuadraticSolution::Linear(-c / b);
    }
    let disc = b * b - 4.0 * a * c;
    let mid = -b / (2.0 * a);
    let sep = disc.abs().sqrt() / (2.0 * a.abs());
    if sep <= tol.confusion * (1.0 + mid.abs()) {
        QuadraticSolution::One(mid)
    } else if disc < 0.0 {
        QuadraticSolution::Empty
    } else {
        let (t1, t2) = (mid - sep, mid + sep);
        QuadraticSolution::Two(t1, t2)
    }
}

/// One transversal meeting of a curve and a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSurfaceHit {
    /// Parameter on the curve.
    pub curve_param: f64,
    /// Parameters on the surface.
    pub surface_params: (f64, f64),
    /// The meeting point (on the curve; within tolerance of the surface).
    pub point: Point3D,
}

/// Generic curve-surface intersection for any parametric curve and surface.
///
/// Curve samples projected onto the surface seed Newton iteration on
/// `C(t) - S(u, v) = 0` (first derivatives only); converged roots with a
/// residual within `tol.confusion` report, ordered by curve parameter.
/// Unbounded curves seed `[-10, 10]`. Overlapping pairs yield dense hit
/// sets along the overlap — use the analytic `intersect_*` methods when
/// classification (tangent, coincident) matters.
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
    let (t0, t1) = curve.bounds();
    // Unbounded curves seed a documented default window.
    const WINDOW: f64 = 10.0;
    let (lo, hi) = (
        if t0.is_finite() { t0 } else { -WINDOW },
        if t1.is_finite() { t1 } else { WINDOW },
    );
    const SEEDS: usize = 128;
    const KEEP: usize = 24;
    let mut samples: Vec<(f64, f64)> = (0..SEEDS)
        .map(|i| {
            let t = lo + (hi - lo) * i as f64 / (SEEDS - 1) as f64;
            (t, surface.project_point(curve.eval_point(t), tol).distance)
        })
        .collect();
    samples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut hits: Vec<CurveSurfaceHit> = Vec::new();
    for (t, _) in samples.into_iter().take(KEEP) {
        let proj = surface.project_point(curve.eval_point(t), tol);
        let f = |x: [f64; 3]| {
            let d = curve.eval_point(x[0]) - surface.eval_point(x[1], x[2]);
            [d.x, d.y, d.z]
        };
        let jac = |x: [f64; 3]| {
            let ct = curve.eval_derivative(x[0], 1);
            let su = surface.eval_derivative(x[1], x[2], 1, 0);
            let sv = surface.eval_derivative(x[1], x[2], 0, 1);
            [
                [ct.x, -su.x, -sv.x],
                [ct.y, -su.y, -sv.y],
                [ct.z, -su.z, -sv.z],
            ]
        };
        if let Some([rt, ru, rv]) = newton_3d(f, jac, [t, proj.u, proj.v], tol.confusion, 50) {
            let p = curve.eval_point(rt);
            let q = surface.eval_point(ru, rv);
            if p.distance(q) <= tol.confusion
                && !hits
                    .iter()
                    .any(|h: &CurveSurfaceHit| h.point.distance(p) <= tol.confusion)
            {
                hits.push(CurveSurfaceHit {
                    curve_param: rt,
                    surface_params: (ru, rv),
                    point: p,
                });
            }
        }
    }
    hits.sort_by(|a, b| a.curve_param.partial_cmp(&b.curve_param).unwrap());
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tolerance;
    use crate::{Line3D, Point3D, Sphere, Vector3D};

    #[test]
    fn test_solve_quadratic_two_roots_ordered() {
        // t^2 - 5t + 6 = 0 -> 2, 3.
        assert_eq!(
            solve_quadratic(1.0, -5.0, 6.0, Tolerance::DEFAULT),
            QuadraticSolution::Two(2.0, 3.0)
        );
    }

    #[test]
    fn test_solve_quadratic_tangent_band() {
        // Exact double root.
        assert_eq!(
            solve_quadratic(1.0, -4.0, 4.0, Tolerance::DEFAULT),
            QuadraticSolution::One(2.0)
        );
        // Miss by sqrt(c) = 1e-7, i.e. exactly at tolerance: still grazing.
        assert_eq!(
            solve_quadratic(1.0, 0.0, 1e-14, Tolerance::DEFAULT),
            QuadraticSolution::One(0.0)
        );
        // Miss by sqrt(c) = 3e-5, beyond tolerance: empty.
        assert_eq!(
            solve_quadratic(1.0, 0.0, 1e-9, Tolerance::DEFAULT),
            QuadraticSolution::Empty
        );
        // Clearly negative discriminant misses.
        assert_eq!(
            solve_quadratic(1.0, 0.0, 1.0, Tolerance::DEFAULT),
            QuadraticSolution::Empty
        );
    }

    #[test]
    fn test_solve_quadratic_degenerate_arms() {
        // Linear: 2t + 4 = 0.
        assert_eq!(
            solve_quadratic(0.0, 2.0, 4.0, Tolerance::DEFAULT),
            QuadraticSolution::Linear(-2.0)
        );
        // Fully degenerate.
        assert_eq!(
            solve_quadratic(0.0, 0.0, 1.0, Tolerance::DEFAULT),
            QuadraticSolution::Degenerate
        );
    }

    #[test]
    fn test_generic_line_sphere_matches_analytic() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let hits = intersect_curve_surface(&line, &sphere, tol);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].curve_param + 2.0).abs() < 1e-9);
        assert!((hits[1].curve_param - 2.0).abs() < 1e-9);
        // Cross-check against the analytic solver.
        match line.intersect_sphere(&sphere, tol) {
            LineQuadricIntersection::TwoPoints((t1, _), (t2, _)) => {
                assert!((hits[0].curve_param - t1).abs() < 1e-9);
                assert!((hits[1].curve_param - t2).abs() < 1e-9);
            }
            _ => panic!("analytic expected two points"),
        }
    }

    #[test]
    fn test_generic_line_sphere_miss() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::X).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        assert!(intersect_curve_surface(&line, &sphere, tol).is_empty());
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
