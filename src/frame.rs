//! Axis and frame placement types: the coordinate systems that curves and
//! surfaces are positioned in.

use crate::tol;
use crate::{Point2, Point3, Vector2, Vector3};
use std::fmt;

/// Error returned when an axis or frame cannot be constructed from the
/// given inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameConstructionError {
    /// A direction vector could not be normalized (zero length).
    NullDirection,
    /// Two directions that should span a plane are parallel.
    ParallelDirections,
    /// Two directions that should be orthogonal are not.
    NotOrthogonal,
}

impl fmt::Display for FrameConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            FrameConstructionError::NullDirection => "direction has zero length",
            FrameConstructionError::ParallelDirections => "directions are parallel",
            FrameConstructionError::NotOrthogonal => "directions are not orthogonal",
        };
        f.write_str(message)
    }
}

impl std::error::Error for FrameConstructionError {}

/// A 3D axis: an origin point and a unit direction.
///
/// # Examples
///
/// ```
/// use geomrust::{Axis3, Point3, Vector3};
/// let axis = Axis3::new(Point3::ORIGIN, Vector3::new(0.0, 0.0, 2.0)).unwrap();
/// assert_eq!(axis.direction(), Vector3::Z);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis3 {
    origin: Point3,
    direction: Vector3,
}

impl Axis3 {
    /// Creates a new axis from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`FrameConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Point3, Vector3};
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::X).unwrap();
    /// assert_eq!(axis.origin(), Point3::ORIGIN);
    /// assert_eq!(axis.direction(), Vector3::X);
    /// ```
    pub fn new(origin: Point3, direction: Vector3) -> Result<Axis3, FrameConstructionError> {
        let direction = direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        Ok(Axis3 { origin, direction })
    }

    /// Returns the origin point of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Point3, Vector3};
    /// let axis = Axis3::new(Point3::new(1.0, 2.0, 3.0), Vector3::X).unwrap();
    /// assert_eq!(axis.origin(), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn origin(self) -> Point3 {
        self.origin
    }

    /// Returns the unit direction of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Point3, Vector3};
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::new(0.0, 5.0, 0.0)).unwrap();
    /// assert_eq!(axis.direction(), Vector3::Y);
    /// ```
    pub fn direction(self) -> Vector3 {
        self.direction
    }
}

/// A right-handed 3D coordinate frame: an origin and three mutually
/// orthogonal unit directions (x, y, z) with `x × y = z`.
///
/// # Examples
///
/// ```
/// use geomrust::Frame3;
/// let f = Frame3::WORLD;
/// assert_eq!(f.z_direction(), geomrust::Vector3::Z);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame3 {
    origin: Point3,
    x_dir: Vector3,
    y_dir: Vector3,
    z_dir: Vector3,
}

impl Frame3 {
    /// The world frame: origin at [`Point3::ORIGIN`], axes aligned with
    /// [`Vector3::X`], [`Vector3::Y`], [`Vector3::Z`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3, Vector3};
    /// assert_eq!(Frame3::WORLD.origin(), Point3::ORIGIN);
    /// assert_eq!(Frame3::WORLD.x_direction(), Vector3::X);
    /// ```
    pub const WORLD: Frame3 = Frame3 {
        origin: Point3::ORIGIN,
        x_dir: Vector3::X,
        y_dir: Vector3::Y,
        z_dir: Vector3::Z,
    };

    /// Creates a frame from an origin, a main (z) direction, and a hint
    /// for the x direction.
    ///
    /// `z_direction` is normalized to give `z`. The hint is projected onto
    /// the plane perpendicular to `z` via a double cross product,
    /// `x = (z × x_hint) × z`, then normalized to give `x`. This rejects the
    /// component of `x_hint` along `z`, leaving only the part of `x_hint`
    /// that lies in the plane perpendicular to `z`. Finally `y = z × x`,
    /// which makes the frame right-handed by construction.
    ///
    /// # Errors
    ///
    /// Returns [`FrameConstructionError::NullDirection`] if `z_direction` or
    /// `x_hint` cannot be normalized (zero length), or
    /// [`FrameConstructionError::ParallelDirections`] if `x_hint` is
    /// parallel to `z_direction` (the projection is zero).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3, Vector3};
    /// let f = Frame3::new(Point3::ORIGIN, Vector3::Z, Vector3::X).unwrap();
    /// assert_eq!(f.x_direction(), Vector3::X);
    /// assert_eq!(f.y_direction(), Vector3::Y);
    /// assert_eq!(f.z_direction(), Vector3::Z);
    /// ```
    pub fn new(
        origin: Point3,
        z_direction: Vector3,
        x_hint: Vector3,
    ) -> Result<Frame3, FrameConstructionError> {
        let z_dir = z_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        let x_hint = x_hint
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        let x_dir = z_dir
            .cross(x_hint)
            .cross(z_dir)
            .normalized()
            .ok_or(FrameConstructionError::ParallelDirections)?;
        let y_dir = z_dir.cross(x_dir);
        Ok(Frame3 {
            origin,
            x_dir,
            y_dir,
            z_dir,
        })
    }

    /// Creates a frame from an origin and a main (z) direction alone,
    /// deriving an arbitrary but well-defined x direction.
    ///
    /// The hint used is the world axis (X, Y, or Z) whose component along
    /// `z_direction` has the smallest absolute value (ties broken in favor
    /// of X, then Y), which keeps the projection in [`Frame3::new`]
    /// numerically well-conditioned. Construction then delegates to
    /// [`Frame3::new`].
    ///
    /// # Errors
    ///
    /// Returns [`FrameConstructionError::NullDirection`] if `z_direction`
    /// cannot be normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3, Vector3};
    /// let f = Frame3::from_z(Point3::ORIGIN, Vector3::Z).unwrap();
    /// assert!((f.x_direction().dot(f.y_direction())).abs() < 1e-10);
    /// assert!((f.x_direction().cross(f.y_direction()) - f.z_direction()).magnitude() < 1e-10);
    /// ```
    pub fn from_z(origin: Point3, z_direction: Vector3) -> Result<Frame3, FrameConstructionError> {
        let z_dir = z_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        let candidates = [Vector3::X, Vector3::Y, Vector3::Z];
        let hint = candidates
            .into_iter()
            .min_by(|a, b| {
                z_dir
                    .dot(*a)
                    .abs()
                    .partial_cmp(&z_dir.dot(*b).abs())
                    .unwrap()
            })
            .unwrap();
        Frame3::new(origin, z_dir, hint)
    }

    /// Returns the origin point of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3};
    /// assert_eq!(Frame3::WORLD.origin(), Point3::ORIGIN);
    /// ```
    pub fn origin(self) -> Point3 {
        self.origin
    }

    /// Returns the unit x direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Vector3};
    /// assert_eq!(Frame3::WORLD.x_direction(), Vector3::X);
    /// ```
    pub fn x_direction(self) -> Vector3 {
        self.x_dir
    }

    /// Returns the unit y direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Vector3};
    /// assert_eq!(Frame3::WORLD.y_direction(), Vector3::Y);
    /// ```
    pub fn y_direction(self) -> Vector3 {
        self.y_dir
    }

    /// Returns the unit z direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Vector3};
    /// assert_eq!(Frame3::WORLD.z_direction(), Vector3::Z);
    /// ```
    pub fn z_direction(self) -> Vector3 {
        self.z_dir
    }

    /// Returns the frame's main axis (origin and z direction).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3, Vector3};
    /// let axis = Frame3::WORLD.axis();
    /// assert_eq!(axis.origin(), Point3::ORIGIN);
    /// assert_eq!(axis.direction(), Vector3::Z);
    /// ```
    pub fn axis(self) -> Axis3 {
        Axis3 {
            origin: self.origin,
            direction: self.z_dir,
        }
    }

    /// Returns the coordinates of `p` expressed in this frame, as
    /// `((p - origin)·x, (p - origin)·y, (p - origin)·z)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3};
    /// let (u, v, w) = Frame3::WORLD.local_coordinates(Point3::new(1.0, 2.0, 3.0));
    /// assert_eq!((u, v, w), (1.0, 2.0, 3.0));
    /// ```
    pub fn local_coordinates(self, p: Point3) -> (f64, f64, f64) {
        let d = p - self.origin;
        (d.dot(self.x_dir), d.dot(self.y_dir), d.dot(self.z_dir))
    }

    /// Returns the point at `origin + u*x + v*y + w*z`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3};
    /// let p = Frame3::WORLD.point_at(1.0, 2.0, 3.0);
    /// assert_eq!(p, Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn point_at(self, u: f64, v: f64, w: f64) -> Point3 {
        self.origin + u * self.x_dir + v * self.y_dir + w * self.z_dir
    }
}

/// A 2D axis: an origin point and a unit direction.
///
/// # Examples
///
/// ```
/// use geomrust::{Axis2, Point2, Vector2};
/// let axis = Axis2::new(Point2::ORIGIN, Vector2::new(0.0, 3.0)).unwrap();
/// assert_eq!(axis.direction(), Vector2::Y);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis2 {
    origin: Point2,
    direction: Vector2,
}

impl Axis2 {
    /// Creates a new axis from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`FrameConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis2, Point2, Vector2};
    /// let axis = Axis2::new(Point2::ORIGIN, Vector2::X).unwrap();
    /// assert_eq!(axis.origin(), Point2::ORIGIN);
    /// assert_eq!(axis.direction(), Vector2::X);
    /// ```
    pub fn new(origin: Point2, direction: Vector2) -> Result<Axis2, FrameConstructionError> {
        let direction = direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        Ok(Axis2 { origin, direction })
    }

    /// Returns the origin point of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis2, Point2, Vector2};
    /// let axis = Axis2::new(Point2::new(1.0, 2.0), Vector2::X).unwrap();
    /// assert_eq!(axis.origin(), Point2::new(1.0, 2.0));
    /// ```
    pub fn origin(self) -> Point2 {
        self.origin
    }

    /// Returns the unit direction of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis2, Point2, Vector2};
    /// let axis = Axis2::new(Point2::ORIGIN, Vector2::new(5.0, 0.0)).unwrap();
    /// assert_eq!(axis.direction(), Vector2::X);
    /// ```
    pub fn direction(self) -> Vector2 {
        self.direction
    }
}

/// A 2D coordinate frame: an origin and two orthogonal unit directions
/// (x, y). The pair may be either direct (counterclockwise, `x × y > 0`) or
/// indirect (clockwise); see [`Frame2::is_direct`].
///
/// # Examples
///
/// ```
/// use geomrust::Frame2;
/// let f = Frame2::WORLD;
/// assert!(f.is_direct());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame2 {
    origin: Point2,
    x_dir: Vector2,
    y_dir: Vector2,
}

impl Frame2 {
    /// The world frame: origin at [`Point2::ORIGIN`], axes aligned with
    /// [`Vector2::X`], [`Vector2::Y`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame2, Point2, Vector2};
    /// assert_eq!(Frame2::WORLD.origin(), Point2::ORIGIN);
    /// assert_eq!(Frame2::WORLD.x_direction(), Vector2::X);
    /// ```
    pub const WORLD: Frame2 = Frame2 {
        origin: Point2::ORIGIN,
        x_dir: Vector2::X,
        y_dir: Vector2::Y,
    };

    /// Creates a frame from an origin and two directions.
    ///
    /// Both directions are normalized independently; the resulting pair is
    /// stored as given (either handedness is accepted).
    ///
    /// # Errors
    ///
    /// Returns [`FrameConstructionError::NullDirection`] if either
    /// direction cannot be normalized (zero length), or
    /// [`FrameConstructionError::NotOrthogonal`] if the normalized
    /// directions are not orthogonal (`|x·y| > 1e-9`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame2, Point2, Vector2};
    /// let f = Frame2::new(Point2::ORIGIN, Vector2::X, Vector2::Y).unwrap();
    /// assert_eq!(f.x_direction(), Vector2::X);
    /// assert_eq!(f.y_direction(), Vector2::Y);
    /// ```
    pub fn new(
        origin: Point2,
        x_direction: Vector2,
        y_direction: Vector2,
    ) -> Result<Frame2, FrameConstructionError> {
        let x_dir = x_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        let y_dir = y_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        if x_dir.dot(y_dir).abs() > tol::P_CONFUSION {
            return Err(FrameConstructionError::NotOrthogonal);
        }
        Ok(Frame2 {
            origin,
            x_dir,
            y_dir,
        })
    }

    /// Creates a direct frame from an origin and an x direction; y is the
    /// counterclockwise perpendicular of x.
    ///
    /// # Errors
    ///
    /// Returns [`FrameConstructionError::NullDirection`] if `x_direction`
    /// cannot be normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame2, Point2, Vector2};
    /// let f = Frame2::from_x(Point2::ORIGIN, Vector2::new(0.6, 0.8)).unwrap();
    /// assert_eq!(f.y_direction(), Vector2::new(-0.8, 0.6));
    /// assert!(f.is_direct());
    /// ```
    pub fn from_x(origin: Point2, x_direction: Vector2) -> Result<Frame2, FrameConstructionError> {
        let x_dir = x_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        Ok(Frame2 {
            origin,
            x_dir,
            y_dir: x_dir.perp(),
        })
    }

    /// Returns the origin point of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame2, Point2};
    /// assert_eq!(Frame2::WORLD.origin(), Point2::ORIGIN);
    /// ```
    pub fn origin(self) -> Point2 {
        self.origin
    }

    /// Returns the unit x direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame2, Vector2};
    /// assert_eq!(Frame2::WORLD.x_direction(), Vector2::X);
    /// ```
    pub fn x_direction(self) -> Vector2 {
        self.x_dir
    }

    /// Returns the unit y direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame2, Vector2};
    /// assert_eq!(Frame2::WORLD.y_direction(), Vector2::Y);
    /// ```
    pub fn y_direction(self) -> Vector2 {
        self.y_dir
    }

    /// Returns whether the frame is direct (counterclockwise, `x × y > 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Frame2;
    /// assert!(Frame2::WORLD.is_direct());
    /// ```
    pub fn is_direct(self) -> bool {
        self.x_dir.cross(self.y_dir) > 0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Axis2, Axis3, Frame2, Frame3, FrameConstructionError, Point2, Point3, Vector2, Vector3,
    };

    #[test]
    fn test_axis3_new_unit_direction() {
        let axis = Axis3::new(Point3::ORIGIN, Vector3::new(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(axis.origin(), Point3::ORIGIN);
        assert_eq!(axis.direction(), Vector3::X);
    }

    #[test]
    fn test_axis3_new_zero_direction_errors() {
        assert_eq!(
            Axis3::new(Point3::ORIGIN, Vector3::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame3_world_const() {
        let f = Frame3::WORLD;
        assert_eq!(f.origin(), Point3::ORIGIN);
        assert_eq!(f.x_direction(), Vector3::X);
        assert_eq!(f.y_direction(), Vector3::Y);
        assert_eq!(f.z_direction(), Vector3::Z);
    }

    #[test]
    fn test_frame3_new_axis_aligned_hint() {
        let f = Frame3::new(Point3::ORIGIN, Vector3::Z, Vector3::X).unwrap();
        assert_eq!(f.x_direction(), Vector3::X);
        assert_eq!(f.y_direction(), Vector3::Y);
        assert_eq!(f.z_direction(), Vector3::Z);
    }

    #[test]
    fn test_frame3_new_diagonal_hint_projects_perpendicular() {
        let f = Frame3::new(Point3::ORIGIN, Vector3::Z, Vector3::new(1.0, 1.0, 0.0)).unwrap();
        let expected_x = Vector3::new(1.0, 1.0, 0.0).normalized().unwrap();
        assert!((f.x_direction().x - expected_x.x).abs() < 1e-10);
        assert!((f.x_direction().y - expected_x.y).abs() < 1e-10);
        assert!((f.x_direction().z - expected_x.z).abs() < 1e-10);
        let expected_y = Vector3::Z.cross(f.x_direction());
        assert!((f.y_direction().x - expected_y.x).abs() < 1e-10);
        assert!((f.y_direction().y - expected_y.y).abs() < 1e-10);
        assert!((f.y_direction().z - expected_y.z).abs() < 1e-10);
    }

    #[test]
    fn test_frame3_new_parallel_hint_errors() {
        assert_eq!(
            Frame3::new(Point3::ORIGIN, Vector3::Z, Vector3::Z),
            Err(FrameConstructionError::ParallelDirections)
        );
        assert_eq!(
            Frame3::new(Point3::ORIGIN, Vector3::Z, -Vector3::Z),
            Err(FrameConstructionError::ParallelDirections)
        );
    }

    #[test]
    fn test_frame3_new_null_z_errors() {
        assert_eq!(
            Frame3::new(Point3::ORIGIN, Vector3::ZERO, Vector3::X),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame3_new_null_hint_errors() {
        assert_eq!(
            Frame3::new(Point3::ORIGIN, Vector3::Z, Vector3::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    fn assert_orthonormal_right_handed(f: Frame3) {
        let x = f.x_direction();
        let y = f.y_direction();
        let z = f.z_direction();
        assert!(x.dot(y).abs() < 1e-10, "x.y = {}", x.dot(y));
        assert!(y.dot(z).abs() < 1e-10, "y.z = {}", y.dot(z));
        assert!(x.dot(z).abs() < 1e-10, "x.z = {}", x.dot(z));
        assert!((x.magnitude() - 1.0).abs() < 1e-10);
        assert!((y.magnitude() - 1.0).abs() < 1e-10);
        assert!((z.magnitude() - 1.0).abs() < 1e-10);
        let cross = x.cross(y);
        assert!((cross.x - z.x).abs() < 1e-10);
        assert!((cross.y - z.y).abs() < 1e-10);
        assert!((cross.z - z.z).abs() < 1e-10);
    }

    #[test]
    fn test_frame3_from_z_axis_aligned() {
        let f = Frame3::from_z(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_orthonormal_right_handed(f);
    }

    #[test]
    fn test_frame3_from_z_x_axis_aligned() {
        let f = Frame3::from_z(Point3::ORIGIN, Vector3::X).unwrap();
        assert_orthonormal_right_handed(f);
    }

    #[test]
    fn test_frame3_from_z_diagonal() {
        let z = Vector3::new(1.0, 1.0, 1.0).normalized().unwrap();
        let f = Frame3::from_z(Point3::ORIGIN, z).unwrap();
        assert_orthonormal_right_handed(f);
    }

    #[test]
    fn test_frame3_from_z_null_errors() {
        assert_eq!(
            Frame3::from_z(Point3::ORIGIN, Vector3::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame3_axis() {
        let f = Frame3::WORLD;
        let axis = f.axis();
        assert_eq!(axis.origin(), Point3::ORIGIN);
        assert_eq!(axis.direction(), Vector3::Z);
    }

    #[test]
    fn test_frame3_local_coordinates_and_point_at_round_trip() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let f = Frame3::new(
            origin,
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 0.0),
        )
        .unwrap();
        let p = Point3::new(5.0, -4.0, 7.0);
        let (u, v, w) = f.local_coordinates(p);
        let round_tripped = f.point_at(u, v, w);
        assert!((round_tripped.x - p.x).abs() < 1e-10);
        assert!((round_tripped.y - p.y).abs() < 1e-10);
        assert!((round_tripped.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_frame3_local_coordinates_of_origin_is_zero() {
        let f = Frame3::new(Point3::new(1.0, 2.0, 3.0), Vector3::Z, Vector3::X).unwrap();
        let (u, v, w) = f.local_coordinates(f.origin());
        assert_eq!((u, v, w), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_axis2_new_unit_direction() {
        let axis = Axis2::new(Point2::ORIGIN, Vector2::new(0.0, 3.0)).unwrap();
        assert_eq!(axis.origin(), Point2::ORIGIN);
        assert_eq!(axis.direction(), Vector2::Y);
    }

    #[test]
    fn test_axis2_new_zero_direction_errors() {
        assert_eq!(
            Axis2::new(Point2::ORIGIN, Vector2::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame2_world_const() {
        let f = Frame2::WORLD;
        assert_eq!(f.origin(), Point2::ORIGIN);
        assert_eq!(f.x_direction(), Vector2::X);
        assert_eq!(f.y_direction(), Vector2::Y);
    }

    #[test]
    fn test_frame2_new_orthogonal_ok() {
        let f = Frame2::new(Point2::ORIGIN, Vector2::X, Vector2::Y).unwrap();
        assert_eq!(f.x_direction(), Vector2::X);
        assert_eq!(f.y_direction(), Vector2::Y);
    }

    #[test]
    fn test_frame2_new_non_orthogonal_errors() {
        assert_eq!(
            Frame2::new(Point2::ORIGIN, Vector2::X, Vector2::new(1.0, 1.0)),
            Err(FrameConstructionError::NotOrthogonal)
        );
    }

    #[test]
    fn test_frame2_new_null_direction_errors() {
        assert_eq!(
            Frame2::new(Point2::ORIGIN, Vector2::ZERO, Vector2::Y),
            Err(FrameConstructionError::NullDirection)
        );
        assert_eq!(
            Frame2::new(Point2::ORIGIN, Vector2::X, Vector2::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame2_from_x() {
        let f = Frame2::from_x(Point2::ORIGIN, Vector2::new(0.6, 0.8)).unwrap();
        assert!((f.x_direction().x - 0.6).abs() < 1e-10);
        assert!((f.x_direction().y - 0.8).abs() < 1e-10);
        assert!((f.y_direction().x - (-0.8)).abs() < 1e-10);
        assert!((f.y_direction().y - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_frame2_from_x_null_errors() {
        assert_eq!(
            Frame2::from_x(Point2::ORIGIN, Vector2::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame2_is_direct_true_for_world() {
        assert!(Frame2::WORLD.is_direct());
    }

    #[test]
    fn test_frame2_is_direct_false_for_left_handed() {
        let f = Frame2::new(Point2::ORIGIN, Vector2::X, -Vector2::Y).unwrap();
        assert!(!f.is_direct());
    }

    #[test]
    fn test_frame_construction_error_display() {
        assert_eq!(
            FrameConstructionError::NullDirection.to_string(),
            "direction has zero length"
        );
        assert_eq!(
            FrameConstructionError::ParallelDirections.to_string(),
            "directions are parallel"
        );
        assert_eq!(
            FrameConstructionError::NotOrthogonal.to_string(),
            "directions are not orthogonal"
        );
    }

    #[test]
    fn test_frame_construction_error_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&FrameConstructionError::NullDirection);
    }
}
