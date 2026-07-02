//! Planes in 3D: parametric evaluation, parameter inversion, and
//! construction from points or an implicit equation, thin wrappers over
//! [`crate::surface_math::analytic`].

use crate::surface_math::analytic;
use crate::tol;
use crate::{Frame3, Point3, Vector3};
use std::fmt;

/// Error returned when a [`Plane`] cannot be constructed from the given
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneConstructionError {
    /// The normal has zero length.
    NullNormal,
    /// Two (or more) of the points given to build the plane are coincident
    /// (or too close to distinguish).
    ConfusedPoints,
    /// The three points given to build the plane are collinear, so no
    /// normal can be derived from them.
    CollinearPoints,
    /// The equation `ax + by + cz + d = 0` is degenerate: `a`, `b`, and `c`
    /// are all (numerically) zero, so no normal can be derived from it.
    BadEquation,
}

impl fmt::Display for PlaneConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            PlaneConstructionError::NullNormal => "normal has zero length",
            PlaneConstructionError::ConfusedPoints => "the points are confused",
            PlaneConstructionError::CollinearPoints => "the points are collinear",
            PlaneConstructionError::BadEquation => "the equation is degenerate",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PlaneConstructionError {}

/// A plane in 3D: a [`Frame3`] (origin plus local x/y directions spanning
/// the plane), evaluated as `origin + u*x_dir + v*y_dir`.
///
/// # Examples
///
/// ```
/// use geomrust::{Plane, Point3, Vector3};
/// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
/// assert_eq!(plane.eval_point(2.0, 3.0), Point3::new(2.0, 3.0, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    frame: Frame3,
}

impl Plane {
    /// Creates a plane from a point and a normal.
    ///
    /// The plane frame is derived from `normal` via [`Frame3::from_z`].
    ///
    /// # Errors
    ///
    /// Returns [`PlaneConstructionError::NullNormal`] if `normal` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// assert_eq!(plane.normal(), Vector3::Z);
    /// ```
    pub fn new(point: Point3, normal: Vector3) -> Result<Plane, PlaneConstructionError> {
        let frame =
            Frame3::from_z(point, normal).map_err(|_| PlaneConstructionError::NullNormal)?;
        Ok(Plane::from_frame(frame))
    }

    /// Creates a plane from a frame directly. Infallible: any frame spans a
    /// valid plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Plane};
    /// let plane = Plane::from_frame(Frame3::WORLD);
    /// assert_eq!(plane.frame(), Frame3::WORLD);
    /// ```
    pub fn from_frame(frame: Frame3) -> Plane {
        Plane { frame }
    }

    /// Creates a plane through three points.
    ///
    /// Fails with [`PlaneConstructionError::ConfusedPoints`] if `p1` and
    /// `p2` are coincident (or too close to distinguish); otherwise, if the
    /// three points are collinear (`|(p2-p1) x (p3-p1)| <= tol::CONFUSION *
    /// max(|p2-p1|, |p3-p1|)`), returns
    /// [`PlaneConstructionError::CollinearPoints`].
    ///
    /// The resulting frame's origin is `p1`, its normal is
    /// `normalize((p2-p1) x (p3-p1))`, and its x direction is
    /// `normalize(p2-p1)`.
    ///
    /// # Errors
    ///
    /// Returns [`PlaneConstructionError::ConfusedPoints`] or
    /// [`PlaneConstructionError::CollinearPoints`] as described above.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::from_three_points(
    ///     Point3::ORIGIN,
    ///     Point3::new(1.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
    /// )
    /// .unwrap();
    /// assert_eq!(plane.normal(), Vector3::Z);
    /// ```
    pub fn from_three_points(
        p1: Point3,
        p2: Point3,
        p3: Point3,
    ) -> Result<Plane, PlaneConstructionError> {
        if p1.distance(p2) < tol::CONFUSION {
            return Err(PlaneConstructionError::ConfusedPoints);
        }

        let v12 = p2 - p1;
        let v13 = p3 - p1;
        let normal = v12.cross(v13);
        if normal.magnitude() <= tol::CONFUSION * v12.magnitude().max(v13.magnitude()) {
            return Err(PlaneConstructionError::CollinearPoints);
        }

        let frame =
            Frame3::new(p1, normal, v12).map_err(|_| PlaneConstructionError::CollinearPoints)?;
        Ok(Plane::from_frame(frame))
    }

    /// Creates a plane from the implicit equation `ax + by + cz + d = 0`.
    ///
    /// Fails with [`PlaneConstructionError::BadEquation`] if
    /// `a*a + b*b + c*c <= f64::MIN_POSITIVE` (the equation has no
    /// well-defined normal).
    ///
    /// The plane's normal is `normalize((a, b, c))`. Its location is *not*
    /// the point closest to the world origin; it matches the reference
    /// implementation's axis-intercept convention: `(0, 0, -d/c)` if `c` is
    /// the largest-magnitude coefficient, `(0, -d/b, 0)` if `b` is, or
    /// `(-d/a, 0, 0)` if `a` is (picking whichever axis the equation is
    /// solved against avoids dividing by a near-zero coefficient). The x
    /// direction is an arbitrary vector perpendicular to the normal, chosen
    /// so that zeroing the smallest-magnitude component of the normal and
    /// swapping the other two (with a sign matching the golden fixture)
    /// gives a perpendicular vector; y is `normal x x`.
    ///
    /// # Errors
    ///
    /// Returns [`PlaneConstructionError::BadEquation`] as described above.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Plane;
    /// let plane = Plane::from_coefficients(0.0, 0.0, 1.0, -2.0).unwrap();
    /// assert_eq!(plane.frame().origin(), geomrust::Point3::new(0.0, 0.0, 2.0));
    /// ```
    pub fn from_coefficients(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
    ) -> Result<Plane, PlaneConstructionError> {
        if a * a + b * b + c * c <= f64::MIN_POSITIVE {
            return Err(PlaneConstructionError::BadEquation);
        }

        let (aa, ab, ac) = (a.abs(), b.abs(), c.abs());
        let origin = if ac >= aa && ac >= ab {
            Point3::new(0.0, 0.0, -d / c)
        } else if ab >= aa && ab >= ac {
            Point3::new(0.0, -d / b, 0.0)
        } else {
            Point3::new(-d / a, 0.0, 0.0)
        };

        let normal = Vector3::new(a, b, c);
        let frame = arbitrary_perpendicular_frame(origin, normal);
        Ok(Plane::from_frame(frame))
    }

    /// Returns the plane's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Plane};
    /// let plane = Plane::from_frame(Frame3::WORLD);
    /// assert_eq!(plane.frame(), Frame3::WORLD);
    /// ```
    pub fn frame(&self) -> Frame3 {
        self.frame
    }

    /// Returns the unit normal of the plane.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// assert_eq!(plane.normal(), Vector3::Z);
    /// ```
    pub fn normal(&self) -> Vector3 {
        self.frame.z_direction()
    }

    /// Evaluates the point on the plane at `(u, v)`:
    /// `origin + u*x_dir + v*y_dir`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// assert_eq!(plane.eval_point(2.0, 3.0), Point3::new(2.0, 3.0, 0.0));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3 {
        analytic::plane_d0(&self.frame, u, v)
    }

    /// Evaluates the points on the plane at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// let points = plane.eval_points(&[(1.0, 0.0), (0.0, 1.0)]);
    /// assert_eq!(points[0], Point3::new(1.0, 0.0, 0.0));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the derivative of order `(du, dv)` at `(u, v)`.
    ///
    /// `Su = x_dir`, `Sv = y_dir`; every second derivative is identically
    /// zero (the plane is linear in `u` and `v`).
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`Plane::eval_point`] for the position
    /// itself) or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// assert_eq!(plane.eval_derivative(0.0, 0.0, 1, 0), Vector3::X);
    /// ```
    pub fn eval_derivative(&self, _u: f64, _v: f64, du: u32, dv: u32) -> Vector3 {
        match du + dv {
            0 => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            1..=2 => analytic::plane_derivative(&self.frame, du, dv),
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
            ),
        }
    }

    /// Recovers `(u, v)` of a point on (or near) the plane: its local x/y
    /// coordinates in the plane's frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Plane, Point3, Vector3};
    /// let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// assert_eq!(plane.parameters_of(Point3::new(2.0, 3.0, 0.0)), (2.0, 3.0));
    /// ```
    pub fn parameters_of(&self, point: Point3) -> (f64, f64) {
        analytic::plane_parameters(&self.frame, point)
    }
}

/// Builds a frame at `origin` with `z_dir = normalize(normal)` and an
/// arbitrary x direction perpendicular to it, matching the reference
/// implementation's convention for planes derived from an implicit
/// equation: zero the smallest-magnitude component of the normal and swap
/// the other two (with a sign that reproduces the golden fixture).
///
/// `normal` must be non-zero (checked by the caller).
fn arbitrary_perpendicular_frame(origin: Point3, normal: Vector3) -> Frame3 {
    let z = normal.normalized().expect("normal is non-zero (checked)");
    let (ax, ay, az) = (z.x.abs(), z.y.abs(), z.z.abs());
    let x_dir = if ax <= ay && ax <= az {
        Vector3::new(0.0, z.z, -z.y)
    } else if ay <= ax && ay <= az {
        Vector3::new(z.z, 0.0, -z.x)
    } else {
        Vector3::new(z.y, -z.x, 0.0)
    }
    .normalized()
    .expect("z has unit length and is nonzero on at least two axes, so the swap is nonzero");
    Frame3::new(origin, z, x_dir).expect("x_dir constructed perpendicular to z by design")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;

    // ---- construction ----

    #[test]
    fn test_new_ok() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(p.normal(), Vector3::Z);
        assert_eq!(p.frame().origin(), Point3::ORIGIN);
    }

    #[test]
    fn test_new_null_normal_errors() {
        assert_eq!(
            Plane::new(Point3::ORIGIN, Vector3::ZERO),
            Err(PlaneConstructionError::NullNormal)
        );
    }

    #[test]
    fn test_from_frame_is_infallible_and_roundtrips() {
        let p = Plane::from_frame(Frame3::WORLD);
        assert_eq!(p.frame(), Frame3::WORLD);
    }

    #[test]
    fn test_from_three_points_ok() {
        let p = Plane::from_three_points(
            Point3::ORIGIN,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert_eq!(p.normal(), Vector3::Z);
        assert_eq!(p.frame().origin(), Point3::ORIGIN);
    }

    #[test]
    fn test_from_three_points_confused_errors() {
        let p1 = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(
            Plane::from_three_points(p1, p1, Point3::new(4.0, 5.0, 6.0)),
            Err(PlaneConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_from_three_points_collinear_errors() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(1.0, 0.0, 0.0);
        let p3 = Point3::new(2.0, 0.0, 0.0);
        assert_eq!(
            Plane::from_three_points(p1, p2, p3),
            Err(PlaneConstructionError::CollinearPoints)
        );
    }

    #[test]
    fn test_from_coefficients_ok_axis_aligned() {
        let p = Plane::from_coefficients(0.0, 0.0, 1.0, -2.0).unwrap();
        assert_eq!(p.frame().origin(), Point3::new(0.0, 0.0, 2.0));
        assert_eq!(p.normal(), Vector3::Z);
    }

    #[test]
    fn test_from_coefficients_bad_equation_errors() {
        assert_eq!(
            Plane::from_coefficients(0.0, 0.0, 0.0, 5.0),
            Err(PlaneConstructionError::BadEquation)
        );
    }

    #[test]
    fn test_from_coefficients_matches_golden_case() {
        // From tests/fixtures/construction.json: planes_from_coefficients[0].
        let p = Plane::from_coefficients(1.0, 2.0, 3.0, 4.0).unwrap();
        let frame = p.frame();
        assert!((frame.origin().x - 0.0).abs() < 1e-9);
        assert!((frame.origin().y - 0.0).abs() < 1e-9);
        assert!((frame.origin().z - (-1.3333333333333333)).abs() < 1e-9);
        let x = frame.x_direction();
        assert!((x.x - 1.3877787807814454e-17).abs() < 1e-9);
        assert!((x.y - 0.8320502943378437).abs() < 1e-9);
        assert!((x.z - (-0.5547001962252293)).abs() < 1e-9);
        let y = frame.y_direction();
        assert!((y.x - (-0.9636241116594315)).abs() < 1e-9);
        assert!((y.y - 0.14824986333222026).abs() < 1e-9);
        assert!((y.z - 0.22237479499833032).abs() < 1e-9);
    }

    // ---- evaluation ----

    #[test]
    fn test_eval_point() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(p.eval_point(2.0, 3.0), Point3::new(2.0, 3.0, 0.0));
    }

    #[test]
    fn test_eval_points_matches_loop() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        let uvs = [(0.0, 0.0), (1.0, 2.0), (-1.0, 3.0)];
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| p.eval_point(u, v)).collect();
        assert_eq!(p.eval_points(&uvs), expected);
    }

    #[test]
    fn test_eval_derivative_first_orders() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(p.eval_derivative(0.0, 0.0, 1, 0), Vector3::X);
        assert_eq!(p.eval_derivative(0.0, 0.0, 0, 1), Vector3::Y);
    }

    #[test]
    fn test_eval_derivative_second_orders_are_zero() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(p.eval_derivative(0.0, 0.0, 2, 0), Vector3::ZERO);
        assert_eq!(p.eval_derivative(0.0, 0.0, 0, 2), Vector3::ZERO);
        assert_eq!(p.eval_derivative(0.0, 0.0, 1, 1), Vector3::ZERO);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        p.eval_derivative(0.0, 0.0, 0, 0);
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_order_too_high_panics() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        p.eval_derivative(0.0, 0.0, 2, 1);
    }

    #[test]
    fn test_parameters_of_round_trip() {
        let p = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(p.parameters_of(Point3::new(2.0, 3.0, 0.0)), (2.0, 3.0));
    }

    // ---- PlaneConstructionError ----

    #[test]
    fn test_error_display() {
        assert_eq!(
            PlaneConstructionError::NullNormal.to_string(),
            "normal has zero length"
        );
        assert_eq!(
            PlaneConstructionError::ConfusedPoints.to_string(),
            "the points are confused"
        );
        assert_eq!(
            PlaneConstructionError::CollinearPoints.to_string(),
            "the points are collinear"
        );
        assert_eq!(
            PlaneConstructionError::BadEquation.to_string(),
            "the equation is degenerate"
        );
    }

    #[test]
    fn test_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&PlaneConstructionError::NullNormal);
    }
}
