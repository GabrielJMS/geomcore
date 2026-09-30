//! Infinite lines in 2D and 3D: parametric evaluation and parameter
//! inversion, thin wrappers over [`crate::curve_math::analytic`].

use crate::curve_math::analytic;
use crate::curves::Curve2D;
use crate::curves::parametrize::{self, ParametrizeError};
use crate::intersect::{
    LineCircle2DIntersection, LineCircle3DIntersection, LineLine2DIntersection,
    LineLine3DIntersection, LinePlaneIntersection, LineQuadricIntersection, QuadraticSolution,
    solve_quadratic,
};
use crate::math::solve_2x2;
use crate::projection::{self, CurveProjection};
use crate::surfaces::Surface;
use crate::{
    Axis2D, Axis3D, Circle2D, Circle3D, Cone, Cylinder, Plane, Point2D, Point3D, Sphere, Tolerance,
    Vector2D, Vector3D,
};
use std::fmt;

/// Error returned when a [`Line3D`] or [`Line2D`] cannot be constructed from
/// the given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineConstructionError {
    /// A direction vector could not be normalized (zero length).
    NullDirection,
    /// The two points given to build the line are coincident (or too close
    /// to distinguish), so no direction can be derived from them.
    ConfusedPoints,
}

impl fmt::Display for LineConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            LineConstructionError::NullDirection => "direction has zero length",
            LineConstructionError::ConfusedPoints => "the two points are confused",
        };
        f.write_str(message)
    }
}

impl std::error::Error for LineConstructionError {}

/// An infinite line in 3D: an origin point and a unit direction, evaluated
/// as `origin + u * direction`.
///
/// # Examples
///
/// ```
/// use geomcore::{Line3D, Point3D, Vector3D};
/// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
/// assert_eq!(line.eval_point(3.0), Point3D::new(3.0, 0.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3D {
    axis: Axis3D,
}

impl Line3D {
    /// Creates a new line from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`LineConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(2.0, 0.0, 0.0)).unwrap();
    /// assert_eq!(line.direction(), Vector3D::X);
    /// ```
    pub fn new(origin: Point3D, direction: Vector3D) -> Result<Line3D, LineConstructionError> {
        let axis =
            Axis3D::new(origin, direction).map_err(|_| LineConstructionError::NullDirection)?;
        Ok(Line3D { axis })
    }

    /// Creates a line from an axis directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Line3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Y).unwrap();
    /// let line = Line3D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis3D) -> Line3D {
        Line3D { axis }
    }

    /// Creates a line through two points; the direction points from `p1` to
    /// `p2`.
    ///
    /// The two points must be distinct. Returns
    /// [`LineConstructionError::ConfusedPoints`] if `p2 - p1` cannot be
    /// normalized (the points are coincident or too close to distinguish).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D};
    /// let line = Line3D::from_two_points(Point3D::ORIGIN, Point3D::new(2.0, 0.0, 0.0)).unwrap();
    /// assert_eq!(line.origin(), Point3D::ORIGIN);
    /// ```
    pub fn from_two_points(p1: Point3D, p2: Point3D) -> Result<Line3D, LineConstructionError> {
        let direction = (p2 - p1)
            .normalized()
            .ok_or(LineConstructionError::ConfusedPoints)?;
        Ok(Line3D {
            axis: Axis3D::new(p1, direction).map_err(|_| LineConstructionError::ConfusedPoints)?,
        })
    }

    /// Returns the origin point of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::X).unwrap();
    /// assert_eq!(line.origin(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn origin(&self) -> Point3D {
        self.axis.origin()
    }

    /// Returns the unit direction of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 5.0, 0.0)).unwrap();
    /// assert_eq!(line.direction(), Vector3D::Y);
    /// ```
    pub fn direction(&self) -> Vector3D {
        self.axis.direction()
    }

    /// Returns the line's underlying axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Line3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let line = Line3D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn axis(&self) -> Axis3D {
        self.axis
    }

    /// Evaluates the point on the line at parameter `u`: `origin + u * direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(line.eval_point(3.0), Point3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point3D {
        analytic::line_d0(&self.axis, u)
    }

    /// Evaluates the points on the line at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let points = line.eval_points(&[0.0, 1.0, 2.0]);
    /// assert_eq!(points, vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)]);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// The line is affine in `u`, so the first derivative is the constant
    /// direction vector and every derivative of order 2 or higher is zero.
    /// `u` does not affect the result (included for API consistency with
    /// curved parametric types).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Line3D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 3.0, 0.0)).unwrap();
    /// assert_eq!(line.eval_derivative(1.5, 1), Vector3D::Y);
    /// assert_eq!(line.eval_derivative(1.5, 2), Vector3D::ZERO);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        let _ = u;
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => analytic::line_d1(&self.axis),
            _ => Vector3D::ZERO,
        }
    }

    /// Recovers the parameter `u` of a point on (or near) the line: the
    /// signed projection `(point - origin) . direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(line.parameter_of(Point3D::new(2.5, 0.0, 0.0)), 2.5);
    /// ```
    pub fn parameter_of(&self, point: Point3D) -> f64 {
        analytic::line_parameter(&self.axis, point)
    }

    /// Returns whether `point` lies on the line: the inverse parameter is
    /// recovered with [`Line3D::parameter_of`] and re-evaluated, and the
    /// point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(line.contains(Point3D::new(2.5, 0.0, 0.0), tol));
    /// assert!(!line.contains(Point3D::new(2.5, 1.0, 0.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the line, returning the parameter of the
    /// closest point and its distance. Distances within `tol.confusion`
    /// snap to `0.0`, matching [`Line3D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let proj = line.project_point(Point3D::new(2.5, 1.0, 0.0), Tolerance::DEFAULT);
    /// assert_eq!(proj.parameter, 2.5);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> CurveProjection {
        let u = self.parameter_of(point);
        let distance = self.eval_point(u).distance(point);
        CurveProjection {
            parameter: u,
            distance: projection::snap_distance(distance, tol.confusion),
        }
    }

    /// Projects each point in `points` onto the line.
    ///
    /// Default-style batch wrapper over [`Line3D::project_point`]: one
    /// native call per batch, mirroring [`Line3D::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects this line with a plane.
    ///
    /// A transversal line meets the plane once; a line parallel to the
    /// plane (direction against the normal within `tol.angular`) is either
    /// offset ([`LinePlaneIntersection::Parallel`]) or contained
    /// ([`LinePlaneIntersection::Coincident`], decided by `tol.confusion`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, LinePlaneIntersection, Plane, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match line.intersect_plane(&plane, tol) {
    ///     LinePlaneIntersection::Point(t, p) => {
    ///         assert_eq!(t, 0.0);
    ///         assert_eq!(p, Point3D::ORIGIN);
    ///     }
    ///     _ => panic!("expected a point"),
    /// }
    /// ```
    pub fn intersect_plane(&self, plane: &Plane, tol: Tolerance) -> LinePlaneIntersection {
        let n = plane.normal();
        let b = n.dot(self.direction());
        let c = n.dot(self.origin() - plane.frame().origin());
        if b.abs() <= tol.angular {
            if c.abs() <= tol.confusion {
                LinePlaneIntersection::Coincident
            } else {
                LinePlaneIntersection::Parallel
            }
        } else {
            let t = -c / b;
            LinePlaneIntersection::Point(t, self.eval_point(t))
        }
    }

    /// Intersects this line with a sphere.
    ///
    /// Substituting the unit-speed parametrization into `|X - C|^2 = R^2`
    /// gives a quadratic (leading coefficient exactly 1), classified with
    /// `solve_quadratic`: two ordered hits, a grazing tangent, or empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, LineQuadricIntersection, Point3D, Sphere, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match line.intersect_sphere(&sphere, tol) {
    ///     LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => {
    ///         assert_eq!((t1, t2), (-2.0, 2.0));
    ///         assert_eq!(p1, Point3D::new(-2.0, 0.0, 0.0));
    ///         assert_eq!(p2, Point3D::new(2.0, 0.0, 0.0));
    ///     }
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    pub fn intersect_sphere(&self, sphere: &Sphere, tol: Tolerance) -> LineQuadricIntersection {
        let d = self.direction();
        let w = self.origin() - sphere.center();
        // A is exactly 1: the direction is unit by construction.
        match solve_quadratic(1.0, 2.0 * d.dot(w), w.dot(w) - sphere.radius().powi(2), tol) {
            QuadraticSolution::Two(t1, t2) => LineQuadricIntersection::TwoPoints(
                (t1, self.eval_point(t1)),
                (t2, self.eval_point(t2)),
            ),
            QuadraticSolution::One(t) => LineQuadricIntersection::Tangent(t, self.eval_point(t)),
            QuadraticSolution::Empty => LineQuadricIntersection::Empty,
            QuadraticSolution::Linear(_) | QuadraticSolution::Degenerate => {
                unreachable!("unit direction gives a leading coefficient of 1")
            }
        }
    }

    /// Intersects this line with a cylinder.
    ///
    /// The implicit equation `|X - C|^2 - ((X - C).a)^2 = r^2` along the
    /// line is quadratic; a direction parallel to the axis degrades to the
    /// linear case (one transversal hit) or, when fully degenerate, to a
    /// generator coincidence check.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cylinder, Line3D, LineQuadricIntersection, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match line.intersect_cylinder(&cylinder, tol) {
    ///     LineQuadricIntersection::TwoPoints((t1, _), (t2, _)) => {
    ///         assert_eq!((t1, t2), (-2.0, 2.0));
    ///     }
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    pub fn intersect_cylinder(
        &self,
        cylinder: &Cylinder,
        tol: Tolerance,
    ) -> LineQuadricIntersection {
        let d = self.direction();
        let a = cylinder.axis().direction();
        let w = self.origin() - cylinder.axis().origin();
        let da = d.dot(a);
        let wa = w.dot(a);
        let qa = 1.0 - da * da;
        let qb = 2.0 * (d.dot(w) - da * wa);
        let qc = w.dot(w) - wa * wa - cylinder.radius().powi(2);
        match solve_quadratic(qa, qb, qc, tol) {
            QuadraticSolution::Two(t1, t2) => LineQuadricIntersection::TwoPoints(
                (t1, self.eval_point(t1)),
                (t2, self.eval_point(t2)),
            ),
            QuadraticSolution::One(t) => LineQuadricIntersection::Tangent(t, self.eval_point(t)),
            QuadraticSolution::Linear(t) => {
                LineQuadricIntersection::OnePoint(t, self.eval_point(t))
            }
            QuadraticSolution::Empty => LineQuadricIntersection::Empty,
            QuadraticSolution::Degenerate => {
                if cylinder.contains(self.origin(), tol) {
                    LineQuadricIntersection::Coincident
                } else {
                    LineQuadricIntersection::Empty
                }
            }
        }
    }

    /// Intersects this line with a cone.
    ///
    /// The half-angle equation `((X - A).a)^2 = |X - A|^2*cos^2(phi)`
    /// along the line is quadratic; roots on the wrong nappe (behind the
    /// apex) are filtered, leaving two hits, one transversal hit, a
    /// tangent, or empty. A direction parallel to a generator degrades to
    /// the linear case; full degeneracy resolves to a generator
    /// coincidence check.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Frame3D, Line3D, LineQuadricIntersection, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::X).unwrap();
    /// let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match line.intersect_cone(&cone, tol) {
    ///     LineQuadricIntersection::TwoPoints((t1, _), (t2, _)) => {
    ///         assert!(t1 < t2);
    ///     }
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    pub fn intersect_cone(&self, cone: &Cone, tol: Tolerance) -> LineQuadricIntersection {
        let d = self.direction();
        let a = cone.frame().z_direction();
        let cos_phi = cone.semi_angle().cos();
        let m = self.origin() - cone.apex();
        let da = d.dot(a);
        let ma = m.dot(a);
        let qa = da * da - cos_phi * cos_phi;
        let qb = 2.0 * (da * ma - cos_phi * cos_phi * d.dot(m));
        let qc = ma * ma - cos_phi * cos_phi * m.dot(m);
        // Roots on the wrong nappe (behind the apex) are filtered; a
        // single surviving transversal root reports as one point.
        let keep = |t: f64| {
            let p = self.eval_point(t);
            if (p - cone.apex()).dot(a) >= -tol.confusion {
                Some((t, p))
            } else {
                None
            }
        };
        match solve_quadratic(qa, qb, qc, tol) {
            QuadraticSolution::Two(t1, t2) => match (keep(t1), keep(t2)) {
                (Some(q1), Some(q2)) => LineQuadricIntersection::TwoPoints(q1, q2),
                (Some(q), None) | (None, Some(q)) => LineQuadricIntersection::OnePoint(q.0, q.1),
                (None, None) => LineQuadricIntersection::Empty,
            },
            QuadraticSolution::One(t) => match keep(t) {
                Some((t, p)) => LineQuadricIntersection::Tangent(t, p),
                None => LineQuadricIntersection::Empty,
            },
            QuadraticSolution::Linear(t) => match keep(t) {
                Some((t, p)) => LineQuadricIntersection::OnePoint(t, p),
                None => LineQuadricIntersection::Empty,
            },
            QuadraticSolution::Empty => LineQuadricIntersection::Empty,
            QuadraticSolution::Degenerate => {
                if cone.contains(self.origin(), tol) {
                    LineQuadricIntersection::Coincident
                } else {
                    LineQuadricIntersection::Empty
                }
            }
        }
    }

    /// Computes the exact 2D representation of this line in a surface's
    /// parameter space: a [`Curve2D`] `q(t)` such that
    /// `surface.eval_point(q(t)) == self.eval_point(t)` for the same `t`.
    ///
    /// Only a few line/surface pairs admit a closed-form 2D image: a line on
    /// a plane (projects to a 2D line), a line parallel to a cylinder or cone
    /// axis / a cone generator (a vertical iso-`u` line in `(u, v)`). Every
    /// other pair — including a line on a sphere or torus — returns
    /// [`ParametrizeError::NotAnalytic`].
    ///
    /// The projection math assumes the line lies on the surface. As a
    /// geomcore safeguard, the candidate 2D image is verified against the
    /// surface at a few parameters; if `surface.eval_point(q(t))` disagrees
    /// with `self.eval_point(t)`, [`ParametrizeError::CurveNotOnSurface`] is
    /// returned. For periodic surfaces the result is normalized so `q(0)`
    /// lies in the canonical parameter window.
    ///
    /// # Examples
    ///
    /// A vertical line on a cylinder maps to a vertical line in `(u, v)`:
    ///
    /// ```
    /// use geomcore::curves::{Curve2D, ParametricCurve2D};
    /// use geomcore::{Cylinder, Line3D, Point3D, Vector3D};
    ///
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let line = Line3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z).unwrap();
    /// let pcurve = line.parametrize_on(&cylinder).unwrap();
    /// // q(0) sits at u = 0 (angle of x-axis), v = 0 (height of the origin).
    /// let q0 = pcurve.eval_point(0.0);
    /// assert!(q0.x.abs() < 1e-9 && q0.y.abs() < 1e-9);
    /// assert!(matches!(pcurve, Curve2D::Line(_)));
    /// ```
    pub fn parametrize_on(&self, surface: impl Into<Surface>) -> Result<Curve2D, ParametrizeError> {
        parametrize::line_on_surface(self, &surface.into())
    }

    /// Intersects this line with another 3D line.
    ///
    /// Skew lines report both parameters and their distance (the classic
    /// `sc = (b*e - d)/(1 - b^2)` solution); parallel lines report their
    /// distance or coincidence. Transversal hits carry both parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line3D, LineLine3DIntersection, Point3D, Tolerance, Vector3D};
    /// let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let l2 = Line3D::new(Point3D::ORIGIN, Vector3D::Y).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match l1.intersect_line(&l2, tol) {
    ///     LineLine3DIntersection::Point(s, p, t) => {
    ///         assert_eq!((s, t), (0.0, 0.0));
    ///         assert_eq!(p, Point3D::ORIGIN);
    ///     }
    ///     _ => panic!("expected a point"),
    /// }
    /// ```
    pub fn intersect_line(&self, other: &Line3D, tol: Tolerance) -> LineLine3DIntersection {
        let (d1, d2) = (self.direction(), other.direction());
        let w0 = self.origin() - other.origin();
        let b = d1.dot(d2);
        let denom = 1.0 - b * b;
        if denom.abs() <= tol.angular {
            // Parallel: distance from other's origin to self, or coincident.
            if other.contains(self.origin(), tol) {
                LineLine3DIntersection::Coincident
            } else {
                LineLine3DIntersection::Parallel(self.project_point(other.origin(), tol).distance)
            }
        } else {
            let d = d1.dot(w0);
            let e = d2.dot(w0);
            let s = (b * e - d) / denom;
            let t = (e - b * d) / denom;
            let (p1, p2) = (self.eval_point(s), other.eval_point(t));
            let distance = p1.distance(p2);
            if distance <= tol.confusion {
                LineLine3DIntersection::Point(s, p1, t)
            } else {
                LineLine3DIntersection::Skew { s, t, distance }
            }
        }
    }

    /// Intersects this line with a 3D circle.
    ///
    /// A transversal line meets the circle's plane once: on-circle hits
    /// graze, the rest miss. A line lying in the plane reduces to the 2D
    /// problem (both 2D images exist in closed form), mapping hits back
    /// with the plane evaluation and this line's inverse parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Line3D, LineCircle3DIntersection, Point3D, Tolerance, Vector3D};
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match line.intersect_circle(&circle, tol) {
    ///     LineCircle3DIntersection::Points((t1, _), (t2, _)) => {
    ///         assert_eq!((t1, t2), (-2.0, 2.0));
    ///     }
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    pub fn intersect_circle(&self, circle: &Circle3D, tol: Tolerance) -> LineCircle3DIntersection {
        let plane = Plane::from_frame(circle.frame());
        match self.intersect_plane(&plane, tol) {
            LinePlaneIntersection::Point(t, p) => {
                if circle.contains(p, tol) {
                    LineCircle3DIntersection::Tangent(t, p)
                } else {
                    LineCircle3DIntersection::Empty
                }
            }
            LinePlaneIntersection::Parallel => LineCircle3DIntersection::Empty,
            LinePlaneIntersection::Coincident => {
                let l2 = match self.parametrize_on(plane) {
                    Ok(Curve2D::Line(l)) => l,
                    _ => return LineCircle3DIntersection::Empty,
                };
                let c2 = match circle.parametrize_on(plane) {
                    Ok(Curve2D::Circle(c)) => c,
                    _ => return LineCircle3DIntersection::Empty,
                };
                match l2.intersect_circle(&c2, tol) {
                    LineCircle2DIntersection::Points((_, p1), (_, p2)) => {
                        let (q1, q2) = (plane.eval_point(p1.x, p1.y), plane.eval_point(p2.x, p2.y));
                        LineCircle3DIntersection::Points(
                            (self.parameter_of(q1), q1),
                            (self.parameter_of(q2), q2),
                        )
                    }
                    LineCircle2DIntersection::Tangent(_, p) => {
                        let q = plane.eval_point(p.x, p.y);
                        LineCircle3DIntersection::Tangent(self.parameter_of(q), q)
                    }
                    LineCircle2DIntersection::Empty => LineCircle3DIntersection::Empty,
                }
            }
        }
    }
}

/// An infinite line in 2D: an origin point and a unit direction, evaluated
/// as `origin + u * direction`.
///
/// # Examples
///
/// ```
/// use geomcore::{Line2D, Point2D, Vector2D};
/// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
/// assert_eq!(line.eval_point(3.0), Point2D::new(3.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line2D {
    axis: Axis2D,
}

impl Line2D {
    /// Creates a new line from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`LineConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 5.0)).unwrap();
    /// assert_eq!(line.direction(), Vector2D::Y);
    /// ```
    pub fn new(origin: Point2D, direction: Vector2D) -> Result<Line2D, LineConstructionError> {
        let axis =
            Axis2D::new(origin, direction).map_err(|_| LineConstructionError::NullDirection)?;
        Ok(Line2D { axis })
    }

    /// Creates a line from an axis directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Line2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let line = Line2D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn from_axis(axis: Axis2D) -> Line2D {
        Line2D { axis }
    }

    /// Creates a line through two points; the direction points from `p1` to
    /// `p2`.
    ///
    /// The two points must be distinct. Returns
    /// [`LineConstructionError::ConfusedPoints`] if `p2 - p1` cannot be
    /// normalized (the points are coincident or too close to distinguish).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D};
    /// let line = Line2D::from_two_points(Point2D::ORIGIN, Point2D::new(0.0, 4.0)).unwrap();
    /// assert_eq!(line.origin(), Point2D::ORIGIN);
    /// ```
    pub fn from_two_points(p1: Point2D, p2: Point2D) -> Result<Line2D, LineConstructionError> {
        let direction = (p2 - p1)
            .normalized()
            .ok_or(LineConstructionError::ConfusedPoints)?;
        Ok(Line2D {
            axis: Axis2D::new(p1, direction).map_err(|_| LineConstructionError::ConfusedPoints)?,
        })
    }

    /// Returns the origin point of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::new(1.0, 2.0), Vector2D::X).unwrap();
    /// assert_eq!(line.origin(), Point2D::new(1.0, 2.0));
    /// ```
    pub fn origin(&self) -> Point2D {
        self.axis.origin()
    }

    /// Returns the unit direction of the line.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 5.0)).unwrap();
    /// assert_eq!(line.direction(), Vector2D::Y);
    /// ```
    pub fn direction(&self) -> Vector2D {
        self.axis.direction()
    }

    /// Returns the line's underlying axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Line2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let line = Line2D::from_axis(axis);
    /// assert_eq!(line.axis(), axis);
    /// ```
    pub fn axis(&self) -> Axis2D {
        self.axis
    }

    /// Evaluates the point on the line at parameter `u`: `origin + u * direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// assert_eq!(line.eval_point(3.0), Point2D::new(3.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64) -> Point2D {
        analytic::line2d_d0(&self.axis, u)
    }

    /// Evaluates the points on the line at each parameter in `us`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let points = line.eval_points(&[0.0, 1.0, 2.0]);
    /// assert_eq!(points, vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0), Point2D::new(2.0, 0.0)]);
    /// ```
    pub fn eval_points(&self, us: &[f64]) -> Vec<Point2D> {
        us.iter().map(|&u| self.eval_point(u)).collect()
    }

    /// Evaluates the derivative of the given `order` at parameter `u`.
    ///
    /// The line is affine in `u`, so the first derivative is the constant
    /// direction vector and every derivative of order 2 or higher is zero.
    /// `u` does not affect the result (included for API consistency with
    /// curved parametric types).
    ///
    /// # Panics
    ///
    /// Panics if `order == 0`; use [`Line2D::eval_point`] to evaluate the
    /// position itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 3.0)).unwrap();
    /// assert_eq!(line.eval_derivative(1.5, 1), Vector2D::Y);
    /// assert_eq!(line.eval_derivative(1.5, 2), Vector2D::ZERO);
    /// ```
    pub fn eval_derivative(&self, u: f64, order: u32) -> Vector2D {
        let _ = u;
        match order {
            0 => panic!("eval_derivative: order must be >= 1 (use eval_point for order 0)"),
            1 => self.axis.direction(),
            _ => Vector2D::ZERO,
        }
    }

    /// Recovers the parameter `u` of a point on (or near) the line: the
    /// signed projection `(point - origin) . direction`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// assert_eq!(line.parameter_of(Point2D::new(2.5, 0.0)), 2.5);
    /// ```
    pub fn parameter_of(&self, point: Point2D) -> f64 {
        analytic::line2d_parameter(&self.axis, point)
    }

    /// Returns whether `point` lies on the line: the inverse parameter is
    /// recovered with [`Line2D::parameter_of`] and re-evaluated, and the
    /// point counts as contained when the re-evaluated point is within
    /// `tol.confusion` of it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Tolerance, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(line.contains(Point2D::new(2.5, 0.0), tol));
    /// assert!(!line.contains(Point2D::new(2.5, 1.0), tol));
    /// ```
    pub fn contains(&self, point: Point2D, tol: Tolerance) -> bool {
        let u = self.parameter_of(point);
        self.eval_point(u).distance(point) <= tol.confusion
    }

    /// Projects `point` onto the line, returning the parameter of the
    /// closest point and its distance. Distances within `tol.confusion`
    /// snap to `0.0`, matching [`Line2D::contains`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, Point2D, Tolerance, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let proj = line.project_point(Point2D::new(2.5, 1.0), Tolerance::DEFAULT);
    /// assert_eq!(proj.parameter, 2.5);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point2D, tol: Tolerance) -> CurveProjection {
        let u = self.parameter_of(point);
        let distance = self.eval_point(u).distance(point);
        CurveProjection {
            parameter: u,
            distance: projection::snap_distance(distance, tol.confusion),
        }
    }

    /// Projects each point in `points` onto the line.
    ///
    /// Default-style batch wrapper over [`Line2D::project_point`]: one
    /// native call per batch, mirroring [`Line2D::eval_points`].
    pub fn project_points(&self, points: &[Point2D], tol: Tolerance) -> Vec<CurveProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// Intersects this line with another 2D line.
    ///
    /// The 2x2 system solves both parameters at once; singularity (against
    /// `tol.angular`) falls back to a coincidence check.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Line2D, LineLine2DIntersection, Point2D, Tolerance, Vector2D};
    /// let l1 = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let l2 = Line2D::new(Point2D::ORIGIN, Vector2D::Y).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match l1.intersect_line(&l2, tol) {
    ///     LineLine2DIntersection::Point(s, p, t) => {
    ///         assert_eq!((s, t), (0.0, 0.0));
    ///         assert_eq!(p, Point2D::ORIGIN);
    ///     }
    ///     _ => panic!("expected a point"),
    /// }
    /// ```
    pub fn intersect_line(&self, other: &Line2D, tol: Tolerance) -> LineLine2DIntersection {
        let (d1, d2) = (self.direction(), other.direction());
        let w = other.origin() - self.origin();
        let lhs = [[d1.x, -d2.x], [d1.y, -d2.y]];
        match solve_2x2(lhs, [w.x, w.y], tol.angular) {
            Some([s, t]) => LineLine2DIntersection::Point(s, self.eval_point(s), t),
            None => {
                if other.contains(self.origin(), tol) {
                    LineLine2DIntersection::Coincident
                } else {
                    LineLine2DIntersection::Parallel
                }
            }
        }
    }

    /// Intersects this line with a 2D circle.
    ///
    /// Substituting the unit-speed parametrization into
    /// `|X - C|^2 = r^2` gives a quadratic with leading coefficient
    /// exactly 1, classified with `solve_quadratic`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle2D, Line2D, LineCircle2DIntersection, Point2D, Tolerance, Vector2D};
    /// let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// match line.intersect_circle(&circle, tol) {
    ///     LineCircle2DIntersection::Points((t1, _), (t2, _)) => {
    ///         assert_eq!((t1, t2), (-2.0, 2.0));
    ///     }
    ///     _ => panic!("expected two points"),
    /// }
    /// ```
    pub fn intersect_circle(&self, circle: &Circle2D, tol: Tolerance) -> LineCircle2DIntersection {
        let d = self.direction();
        let w = self.origin() - circle.center();
        // A is exactly 1: the direction is unit by construction.
        match solve_quadratic(1.0, 2.0 * d.dot(w), w.dot(w) - circle.radius().powi(2), tol) {
            QuadraticSolution::Two(t1, t2) => LineCircle2DIntersection::Points(
                (t1, self.eval_point(t1)),
                (t2, self.eval_point(t2)),
            ),
            QuadraticSolution::One(t) => LineCircle2DIntersection::Tangent(t, self.eval_point(t)),
            QuadraticSolution::Empty => LineCircle2DIntersection::Empty,
            QuadraticSolution::Linear(_) | QuadraticSolution::Degenerate => {
                unreachable!("unit direction gives a leading coefficient of 1")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Axis2D, Axis3D, Circle2D, Circle3D, Cone, Cylinder, Frame3D, Line2D, Line3D,
        LineCircle2DIntersection, LineCircle3DIntersection, LineConstructionError,
        LineLine2DIntersection, LineLine3DIntersection, LinePlaneIntersection,
        LineQuadricIntersection, Plane, Point2D, Point3D, Sphere, Tolerance, Vector2D, Vector3D,
    };

    // ---- Line3D construction ----

    #[test]
    fn test_line3d_new_normalizes_direction() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(line.origin(), Point3D::ORIGIN);
        assert_eq!(line.direction(), Vector3D::X);
    }

    #[test]
    fn test_line3d_new_null_direction_errors() {
        assert_eq!(
            Line3D::new(Point3D::ORIGIN, Vector3D::ZERO),
            Err(LineConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_line3d_from_axis() {
        let axis = Axis3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Y).unwrap();
        let line = Line3D::from_axis(axis);
        assert_eq!(line.origin(), Point3D::new(1.0, 2.0, 3.0));
        assert_eq!(line.direction(), Vector3D::Y);
        assert_eq!(line.axis(), axis);
    }

    #[test]
    fn test_line3d_from_two_points() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(2.0, 0.0, 0.0);
        let line = Line3D::from_two_points(p1, p2).unwrap();
        assert_eq!(line.origin(), p1);
        assert_eq!(line.direction(), Vector3D::X);
    }

    #[test]
    fn test_line3d_from_two_points_confused_errors() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(
            Line3D::from_two_points(p, p),
            Err(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- Line3D evaluation ----

    #[test]
    fn test_line3d_eval_point() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(line.eval_point(3.0), Point3D::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_line3d_eval_points_matches_loop() {
        let line = Line3D::new(Point3D::new(1.0, -2.0, 0.5), Vector3D::Y).unwrap();
        let us = [0.0, 1.5, -3.0];
        let expected: Vec<Point3D> = us.iter().map(|&u| line.eval_point(u)).collect();
        assert_eq!(line.eval_points(&us), expected);
    }

    #[test]
    fn test_line3d_eval_derivative_order1_is_direction() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 3.0, 0.0)).unwrap();
        assert_eq!(line.eval_derivative(1.5, 1), Vector3D::Y);
    }

    #[test]
    fn test_line3d_eval_derivative_order2_is_zero() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 2), Vector3D::ZERO);
    }

    #[test]
    fn test_line3d_eval_derivative_order3_is_zero() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 3), Vector3D::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_line3d_eval_derivative_order0_panics() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        line.eval_derivative(1.5, 0);
    }

    #[test]
    fn test_line3d_parameter_of_round_trip() {
        let line = Line3D::new(Point3D::new(1.0, -2.0, 0.5), Vector3D::new(1.0, 2.0, 2.0)).unwrap();
        for u in [0.3, 2.0, -5.5] {
            let p = line.eval_point(u);
            assert!((line.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    // ---- Line2D construction ----

    #[test]
    fn test_line2d_new_normalizes_direction() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 5.0)).unwrap();
        assert_eq!(line.origin(), Point2D::ORIGIN);
        assert_eq!(line.direction(), Vector2D::Y);
    }

    #[test]
    fn test_line2d_new_null_direction_errors() {
        assert_eq!(
            Line2D::new(Point2D::ORIGIN, Vector2D::ZERO),
            Err(LineConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_line2d_from_axis() {
        let axis = Axis2D::new(Point2D::new(1.0, 2.0), Vector2D::X).unwrap();
        let line = Line2D::from_axis(axis);
        assert_eq!(line.origin(), Point2D::new(1.0, 2.0));
        assert_eq!(line.direction(), Vector2D::X);
        assert_eq!(line.axis(), axis);
    }

    #[test]
    fn test_line2d_from_two_points() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(0.0, 4.0);
        let line = Line2D::from_two_points(p1, p2).unwrap();
        assert_eq!(line.origin(), p1);
        assert_eq!(line.direction(), Vector2D::Y);
    }

    #[test]
    fn test_line2d_from_two_points_confused_errors() {
        let p = Point2D::new(1.0, 2.0);
        assert_eq!(
            Line2D::from_two_points(p, p),
            Err(LineConstructionError::ConfusedPoints)
        );
    }

    // ---- Line2D evaluation ----

    #[test]
    fn test_line2d_eval_point() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        assert_eq!(line.eval_point(3.0), Point2D::new(3.0, 0.0));
    }

    #[test]
    fn test_line2d_eval_points_matches_loop() {
        let line = Line2D::new(Point2D::new(1.0, -2.0), Vector2D::Y).unwrap();
        let us = [0.0, 1.5, -3.0];
        let expected: Vec<Point2D> = us.iter().map(|&u| line.eval_point(u)).collect();
        assert_eq!(line.eval_points(&us), expected);
    }

    #[test]
    fn test_line2d_eval_derivative_order1_is_direction() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 3.0)).unwrap();
        assert_eq!(line.eval_derivative(1.5, 1), Vector2D::Y);
    }

    #[test]
    fn test_line2d_eval_derivative_order2_is_zero() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        assert_eq!(line.eval_derivative(1.5, 2), Vector2D::ZERO);
    }

    #[test]
    #[should_panic]
    fn test_line2d_eval_derivative_order0_panics() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        line.eval_derivative(1.5, 0);
    }

    #[test]
    fn test_line2d_parameter_of_round_trip() {
        let line = Line2D::new(Point2D::new(1.0, -2.0), Vector2D::new(3.0, 4.0)).unwrap();
        for u in [0.3, 2.0, 5.5] {
            let p = line.eval_point(u);
            assert!((line.parameter_of(p) - u).abs() < 1e-9);
        }
    }

    // ---- LineConstructionError ----

    #[test]
    fn test_line_construction_error_display() {
        assert_eq!(
            LineConstructionError::NullDirection.to_string(),
            "direction has zero length"
        );
        assert_eq!(
            LineConstructionError::ConfusedPoints.to_string(),
            "the two points are confused"
        );
    }

    #[test]
    fn test_line_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&LineConstructionError::NullDirection);
    }

    #[test]
    fn test_line3d_contains() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(line.contains(Point3D::new(2.5, 0.0, 0.0), tol));
        assert!(line.contains(line.eval_point(-3.0), tol));
        assert!(!line.contains(Point3D::new(2.5, 1.0, 0.0), tol));
        // Tolerance boundary: 1e-6 off the line fails at default, passes loose.
        let near = Point3D::new(0.0, 1e-6, 0.0);
        assert!(!line.contains(near, tol));
        assert!(line.contains(
            near,
            Tolerance {
                confusion: 1e-5,
                ..tol
            }
        ));
    }

    #[test]
    fn test_line2d_contains() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        assert!(line.contains(Point2D::new(2.5, 0.0), tol));
        assert!(line.contains(line.eval_point(-3.0), tol));
        assert!(!line.contains(Point2D::new(2.5, 1.0), tol));
    }

    #[test]
    fn test_line3d_project_point() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        let on = line.project_point(Point3D::new(2.5, 0.0, 0.0), tol);
        assert_eq!(on.parameter, 2.5);
        assert_eq!(on.distance, 0.0);
        let off = line.project_point(Point3D::new(2.5, 1.0, 0.0), tol);
        assert_eq!(off.parameter, 2.5);
        assert_eq!(off.distance, 1.0);
        // contains agrees with zero distance.
        assert_eq!(
            line.contains(Point3D::new(2.5, 0.0, 0.0), tol),
            on.distance == 0.0
        );
    }

    #[test]
    fn test_line3d_project_points_batch() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        let points = [
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 2.0, 0.0),
            Point3D::new(-1.0, 0.0, 3.0),
        ];
        let projs = line.project_points(&points, tol);
        assert_eq!(projs.len(), 3);
        assert_eq!(projs[0].distance, 0.0);
        assert_eq!(projs[1].parameter, 1.0);
        assert_eq!(projs[1].distance, 2.0);
        assert_eq!(projs[2].parameter, -1.0);
        assert_eq!(projs[2].distance, 3.0);
    }

    #[test]
    fn test_line2d_project_point() {
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        let tol = Tolerance::DEFAULT;
        let proj = line.project_point(Point2D::new(2.5, 1.0), tol);
        assert_eq!(proj.parameter, 2.5);
        assert_eq!(proj.distance, 1.0);
    }

    #[test]
    fn test_line3d_intersect_plane() {
        let tol = Tolerance::DEFAULT;
        let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        match line.intersect_plane(&plane, tol) {
            LinePlaneIntersection::Point(t, p) => {
                assert_eq!(t, 0.0);
                assert_eq!(p, Point3D::ORIGIN);
            }
            _ => panic!("expected a point"),
        }
        // Parallel offset.
        let off = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::X).unwrap();
        assert_eq!(
            off.intersect_plane(&plane, tol),
            LinePlaneIntersection::Parallel
        );
        // Contained.
        let flat = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_eq!(
            flat.intersect_plane(&plane, tol),
            LinePlaneIntersection::Coincident
        );
    }

    #[test]
    fn test_line3d_intersect_sphere() {
        let tol = Tolerance::DEFAULT;
        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        match line.intersect_sphere(&sphere, tol) {
            LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => {
                assert_eq!((t1, t2), (-2.0, 2.0));
                assert_eq!(p1, Point3D::new(-2.0, 0.0, 0.0));
                assert_eq!(p2, Point3D::new(2.0, 0.0, 0.0));
            }
            _ => panic!("expected two points"),
        }
        // Tangent at the north pole.
        let tangent = Line3D::new(Point3D::new(0.0, 0.0, 2.0), Vector3D::X).unwrap();
        match tangent.intersect_sphere(&sphere, tol) {
            LineQuadricIntersection::Tangent(t, p) => {
                assert_eq!(t, 0.0);
                assert_eq!(p, Point3D::new(0.0, 0.0, 2.0));
            }
            _ => panic!("expected a tangent"),
        }
        // Clear miss.
        let miss = Line3D::new(Point3D::new(0.0, 0.0, 3.0), Vector3D::X).unwrap();
        assert_eq!(
            miss.intersect_sphere(&sphere, tol),
            LineQuadricIntersection::Empty
        );
    }

    #[test]
    fn test_line3d_intersect_cylinder() {
        let tol = Tolerance::DEFAULT;
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        match line.intersect_cylinder(&cylinder, tol) {
            LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => {
                assert_eq!((t1, t2), (-2.0, 2.0));
                assert_eq!(p1, Point3D::new(-2.0, 0.0, 0.0));
                assert_eq!(p2, Point3D::new(2.0, 0.0, 0.0));
            }
            _ => panic!("expected two points"),
        }
        // Parallel offset: miss.
        let side = Line3D::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z).unwrap();
        assert_eq!(
            side.intersect_cylinder(&cylinder, tol),
            LineQuadricIntersection::Empty
        );
        // Generator: coincident.
        let generator = Line3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z).unwrap();
        assert_eq!(
            generator.intersect_cylinder(&cylinder, tol),
            LineQuadricIntersection::Coincident
        );
    }

    #[test]
    fn test_line3d_intersect_cone() {
        let tol = Tolerance::DEFAULT;
        let cone = Cone::from_frame(Frame3D::WORLD, 0.4, 2.0).unwrap();
        // Horizontal line at z = 5: radius there is (5 + 4.729...) * tan(0.4).
        let line = Line3D::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::X).unwrap();
        match line.intersect_cone(&cone, tol) {
            LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => {
                assert!(t1 < t2);
                assert!((t1.abs() - t2.abs()).abs() < 1e-9);
                assert!(cone.contains(p1, tol));
                assert!(cone.contains(p2, tol));
            }
            _ => panic!("expected two points"),
        }
        // Generator through the apex: coincident.
        let apex = cone.apex();
        let generator = Line3D::new(apex, Vector3D::new(0.4f64.sin(), 0.0, 0.4f64.cos())).unwrap();
        assert_eq!(
            generator.intersect_cone(&cone, tol),
            LineQuadricIntersection::Coincident
        );
        // Vertical line at radial distance 20: one nappe-filtered hit.
        let side = Line3D::new(Point3D::new(20.0, 0.0, 0.0), Vector3D::Z).unwrap();
        match side.intersect_cone(&cone, tol) {
            LineQuadricIntersection::OnePoint(t, p) => {
                assert!(cone.contains(p, tol));
                assert_eq!(p, side.eval_point(t));
            }
            _ => panic!("expected one point"),
        }
        // Horizontal line far from the axis: clear miss.
        let miss = Line3D::new(Point3D::new(0.0, 20.0, 0.0), Vector3D::X).unwrap();
        assert_eq!(
            miss.intersect_cone(&cone, tol),
            LineQuadricIntersection::Empty
        );
        // Generator direction but offset: linear single hit.
        let slanted = Line3D::new(
            Point3D::new(0.0, 5.0, 0.0),
            Vector3D::new(0.4f64.sin(), 0.0, 0.4f64.cos()),
        )
        .unwrap();
        match slanted.intersect_cone(&cone, tol) {
            LineQuadricIntersection::OnePoint(t, p) => {
                assert!(cone.contains(p, tol));
                assert_eq!(p, slanted.eval_point(t));
            }
            _ => panic!("expected one point"),
        }
    }

    #[test]
    fn test_line3d_intersect_line() {
        let tol = Tolerance::DEFAULT;
        let l1 = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let l2 = Line3D::new(Point3D::ORIGIN, Vector3D::Y).unwrap();
        match l1.intersect_line(&l2, tol) {
            LineLine3DIntersection::Point(s, p, t) => {
                assert_eq!((s, t), (0.0, 0.0));
                assert_eq!(p, Point3D::ORIGIN);
            }
            _ => panic!("expected a point"),
        }
        // Skew: x-axis against a z-shifted y-line.
        let skew = Line3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Y).unwrap();
        match l1.intersect_line(&skew, tol) {
            LineLine3DIntersection::Skew { s, t, distance } => {
                assert_eq!((s, t), (0.0, 0.0));
                assert_eq!(distance, 1.0);
            }
            _ => panic!("expected skew"),
        }
        // Parallel distinct.
        let parallel = Line3D::new(Point3D::new(0.0, 1.0, 0.0), Vector3D::X).unwrap();
        match l1.intersect_line(&parallel, tol) {
            LineLine3DIntersection::Parallel(d) => assert_eq!(d, 1.0),
            _ => panic!("expected parallel"),
        }
        // Coincident.
        let same = Line3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::X).unwrap();
        assert_eq!(
            l1.intersect_line(&same, tol),
            LineLine3DIntersection::Coincident
        );
    }

    #[test]
    fn test_line2d_intersect_line() {
        let tol = Tolerance::DEFAULT;
        let l1 = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        let l2 = Line2D::new(Point2D::ORIGIN, Vector2D::Y).unwrap();
        match l1.intersect_line(&l2, tol) {
            LineLine2DIntersection::Point(s, p, t) => {
                assert_eq!((s, t), (0.0, 0.0));
                assert_eq!(p, Point2D::ORIGIN);
            }
            _ => panic!("expected a point"),
        }
        let parallel = Line2D::new(Point2D::new(0.0, 1.0), Vector2D::X).unwrap();
        assert_eq!(
            l1.intersect_line(&parallel, tol),
            LineLine2DIntersection::Parallel
        );
        assert_eq!(
            l1.intersect_line(&l1, tol),
            LineLine2DIntersection::Coincident
        );
    }

    #[test]
    fn test_line2d_intersect_circle() {
        let tol = Tolerance::DEFAULT;
        let line = Line2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
        let circle = Circle2D::new(Point2D::ORIGIN, 2.0).unwrap();
        match line.intersect_circle(&circle, tol) {
            LineCircle2DIntersection::Points((t1, p1), (t2, p2)) => {
                assert_eq!((t1, t2), (-2.0, 2.0));
                assert_eq!(p1, Point2D::new(-2.0, 0.0));
                assert_eq!(p2, Point2D::new(2.0, 0.0));
            }
            _ => panic!("expected two points"),
        }
        // Tangent at the top.
        let tangent = Line2D::new(Point2D::new(0.0, 2.0), Vector2D::X).unwrap();
        match tangent.intersect_circle(&circle, tol) {
            LineCircle2DIntersection::Tangent(t, p) => {
                assert_eq!(t, 0.0);
                assert_eq!(p, Point2D::new(0.0, 2.0));
            }
            _ => panic!("expected a tangent"),
        }
        // Clear miss.
        let miss = Line2D::new(Point2D::new(0.0, 3.0), Vector2D::X).unwrap();
        assert_eq!(
            miss.intersect_circle(&circle, tol),
            LineCircle2DIntersection::Empty
        );
    }

    #[test]
    fn test_line3d_intersect_circle() {
        let tol = Tolerance::DEFAULT;
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        // In-plane line: two hits.
        match line.intersect_circle(&circle, tol) {
            LineCircle3DIntersection::Points((t1, p1), (t2, p2)) => {
                assert_eq!((t1, t2), (-2.0, 2.0));
                assert_eq!(p1, Point3D::new(-2.0, 0.0, 0.0));
                assert_eq!(p2, Point3D::new(2.0, 0.0, 0.0));
            }
            _ => panic!("expected two points"),
        }
        // Transversal graze through a circle point.
        let graze = Line3D::new(Point3D::new(2.0, 0.0, 0.0), Vector3D::Z).unwrap();
        match graze.intersect_circle(&circle, tol) {
            LineCircle3DIntersection::Tangent(t, p) => {
                assert_eq!(t, 0.0);
                assert_eq!(p, Point3D::new(2.0, 0.0, 0.0));
            }
            _ => panic!("expected a tangent"),
        }
        // Transversal miss.
        let miss = Line3D::new(Point3D::new(3.0, 0.0, 0.0), Vector3D::Z).unwrap();
        assert_eq!(
            miss.intersect_circle(&circle, tol),
            LineCircle3DIntersection::Empty
        );
    }
}
