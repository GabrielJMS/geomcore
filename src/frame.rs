//! Axis and frame placement types: the coordinate systems that curves and
//! surfaces are positioned in.

use crate::tol;
use crate::{Point2D, Point3D, Vector2D, Vector3D};
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
/// use geomcore::{Axis3D, Point3D, Vector3D};
/// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 0.0, 2.0)).unwrap();
/// assert_eq!(axis.direction(), Vector3D::Z);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis3D {
    origin: Point3D,
    direction: Vector3D,
}

impl Axis3D {
    /// Creates a new axis from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`FrameConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(axis.origin(), Point3D::ORIGIN);
    /// assert_eq!(axis.direction(), Vector3D::X);
    /// ```
    pub fn new(origin: Point3D, direction: Vector3D) -> Result<Axis3D, FrameConstructionError> {
        let direction = direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        Ok(Axis3D { origin, direction })
    }

    /// Returns the origin point of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::X).unwrap();
    /// assert_eq!(axis.origin(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn origin(self) -> Point3D {
        self.origin
    }

    /// Returns the unit direction of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis3D, Point3D, Vector3D};
    /// let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::new(0.0, 5.0, 0.0)).unwrap();
    /// assert_eq!(axis.direction(), Vector3D::Y);
    /// ```
    pub fn direction(self) -> Vector3D {
        self.direction
    }
}

/// A right-handed 3D coordinate frame: an origin and three mutually
/// orthogonal unit directions (x, y, z) with `x × y = z`.
///
/// # Examples
///
/// ```
/// use geomcore::Frame3D;
/// let f = Frame3D::WORLD;
/// assert_eq!(f.z_direction(), geomcore::Vector3D::Z);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame3D {
    origin: Point3D,
    x_dir: Vector3D,
    y_dir: Vector3D,
    z_dir: Vector3D,
}

impl Frame3D {
    /// The world frame: origin at [`Point3D::ORIGIN`], axes aligned with
    /// [`Vector3D::X`], [`Vector3D::Y`], [`Vector3D::Z`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Point3D, Vector3D};
    /// assert_eq!(Frame3D::WORLD.origin(), Point3D::ORIGIN);
    /// assert_eq!(Frame3D::WORLD.x_direction(), Vector3D::X);
    /// ```
    pub const WORLD: Frame3D = Frame3D {
        origin: Point3D::ORIGIN,
        x_dir: Vector3D::X,
        y_dir: Vector3D::Y,
        z_dir: Vector3D::Z,
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
    /// use geomcore::{Frame3D, Point3D, Vector3D};
    /// let f = Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X).unwrap();
    /// assert_eq!(f.x_direction(), Vector3D::X);
    /// assert_eq!(f.y_direction(), Vector3D::Y);
    /// assert_eq!(f.z_direction(), Vector3D::Z);
    /// ```
    pub fn new(
        origin: Point3D,
        z_direction: Vector3D,
        x_hint: Vector3D,
    ) -> Result<Frame3D, FrameConstructionError> {
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
        Ok(Frame3D {
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
    /// of X, then Y), which keeps the projection in [`Frame3D::new`]
    /// numerically well-conditioned. Construction then delegates to
    /// [`Frame3D::new`].
    ///
    /// # Errors
    ///
    /// Returns [`FrameConstructionError::NullDirection`] if `z_direction`
    /// cannot be normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Point3D, Vector3D};
    /// let f = Frame3D::from_z(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert!((f.x_direction().dot(f.y_direction())).abs() < 1e-10);
    /// assert!((f.x_direction().cross(f.y_direction()) - f.z_direction()).magnitude() < 1e-10);
    /// ```
    pub fn from_z(
        origin: Point3D,
        z_direction: Vector3D,
    ) -> Result<Frame3D, FrameConstructionError> {
        let z_dir = z_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        let candidates = [Vector3D::X, Vector3D::Y, Vector3D::Z];
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
        Frame3D::new(origin, z_dir, hint)
    }

    /// Returns the origin point of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Point3D};
    /// assert_eq!(Frame3D::WORLD.origin(), Point3D::ORIGIN);
    /// ```
    pub fn origin(self) -> Point3D {
        self.origin
    }

    /// Returns the unit x direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Vector3D};
    /// assert_eq!(Frame3D::WORLD.x_direction(), Vector3D::X);
    /// ```
    pub fn x_direction(self) -> Vector3D {
        self.x_dir
    }

    /// Returns the unit y direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Vector3D};
    /// assert_eq!(Frame3D::WORLD.y_direction(), Vector3D::Y);
    /// ```
    pub fn y_direction(self) -> Vector3D {
        self.y_dir
    }

    /// Returns the unit z direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Vector3D};
    /// assert_eq!(Frame3D::WORLD.z_direction(), Vector3D::Z);
    /// ```
    pub fn z_direction(self) -> Vector3D {
        self.z_dir
    }

    /// Returns the frame's main axis (origin and z direction).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Point3D, Vector3D};
    /// let axis = Frame3D::WORLD.axis();
    /// assert_eq!(axis.origin(), Point3D::ORIGIN);
    /// assert_eq!(axis.direction(), Vector3D::Z);
    /// ```
    pub fn axis(self) -> Axis3D {
        Axis3D {
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
    /// use geomcore::{Frame3D, Point3D};
    /// let (u, v, w) = Frame3D::WORLD.local_coordinates(Point3D::new(1.0, 2.0, 3.0));
    /// assert_eq!((u, v, w), (1.0, 2.0, 3.0));
    /// ```
    pub fn local_coordinates(self, p: Point3D) -> (f64, f64, f64) {
        let d = p - self.origin;
        (d.dot(self.x_dir), d.dot(self.y_dir), d.dot(self.z_dir))
    }

    /// Returns the point at `origin + u*x + v*y + w*z`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame3D, Point3D};
    /// let p = Frame3D::WORLD.point_at(1.0, 2.0, 3.0);
    /// assert_eq!(p, Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn point_at(self, u: f64, v: f64, w: f64) -> Point3D {
        self.origin + u * self.x_dir + v * self.y_dir + w * self.z_dir
    }
}

/// A 2D axis: an origin point and a unit direction.
///
/// # Examples
///
/// ```
/// use geomcore::{Axis2D, Point2D, Vector2D};
/// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 3.0)).unwrap();
/// assert_eq!(axis.direction(), Vector2D::Y);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis2D {
    origin: Point2D,
    direction: Vector2D,
}

impl Axis2D {
    /// Creates a new axis from an origin and a direction.
    ///
    /// The direction is normalized. Returns
    /// [`FrameConstructionError::NullDirection`] if `direction` cannot be
    /// normalized (zero length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::X).unwrap();
    /// assert_eq!(axis.origin(), Point2D::ORIGIN);
    /// assert_eq!(axis.direction(), Vector2D::X);
    /// ```
    pub fn new(origin: Point2D, direction: Vector2D) -> Result<Axis2D, FrameConstructionError> {
        let direction = direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        Ok(Axis2D { origin, direction })
    }

    /// Returns the origin point of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::new(1.0, 2.0), Vector2D::X).unwrap();
    /// assert_eq!(axis.origin(), Point2D::new(1.0, 2.0));
    /// ```
    pub fn origin(self) -> Point2D {
        self.origin
    }

    /// Returns the unit direction of the axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Axis2D, Point2D, Vector2D};
    /// let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::new(5.0, 0.0)).unwrap();
    /// assert_eq!(axis.direction(), Vector2D::X);
    /// ```
    pub fn direction(self) -> Vector2D {
        self.direction
    }
}

/// A 2D coordinate frame: an origin and two orthogonal unit directions
/// (x, y). The pair may be either direct (counterclockwise, `x × y > 0`) or
/// indirect (clockwise); see [`Frame2D::is_direct`].
///
/// # Examples
///
/// ```
/// use geomcore::Frame2D;
/// let f = Frame2D::WORLD;
/// assert!(f.is_direct());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame2D {
    origin: Point2D,
    x_dir: Vector2D,
    y_dir: Vector2D,
}

impl Frame2D {
    /// The world frame: origin at [`Point2D::ORIGIN`], axes aligned with
    /// [`Vector2D::X`], [`Vector2D::Y`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame2D, Point2D, Vector2D};
    /// assert_eq!(Frame2D::WORLD.origin(), Point2D::ORIGIN);
    /// assert_eq!(Frame2D::WORLD.x_direction(), Vector2D::X);
    /// ```
    pub const WORLD: Frame2D = Frame2D {
        origin: Point2D::ORIGIN,
        x_dir: Vector2D::X,
        y_dir: Vector2D::Y,
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
    /// use geomcore::{Frame2D, Point2D, Vector2D};
    /// let f = Frame2D::new(Point2D::ORIGIN, Vector2D::X, Vector2D::Y).unwrap();
    /// assert_eq!(f.x_direction(), Vector2D::X);
    /// assert_eq!(f.y_direction(), Vector2D::Y);
    /// ```
    pub fn new(
        origin: Point2D,
        x_direction: Vector2D,
        y_direction: Vector2D,
    ) -> Result<Frame2D, FrameConstructionError> {
        let x_dir = x_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        let y_dir = y_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        if x_dir.dot(y_dir).abs() > tol::P_CONFUSION {
            return Err(FrameConstructionError::NotOrthogonal);
        }
        Ok(Frame2D {
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
    /// use geomcore::{Frame2D, Point2D, Vector2D};
    /// let f = Frame2D::from_x(Point2D::ORIGIN, Vector2D::new(0.6, 0.8)).unwrap();
    /// assert_eq!(f.y_direction(), Vector2D::new(-0.8, 0.6));
    /// assert!(f.is_direct());
    /// ```
    pub fn from_x(
        origin: Point2D,
        x_direction: Vector2D,
    ) -> Result<Frame2D, FrameConstructionError> {
        let x_dir = x_direction
            .normalized()
            .ok_or(FrameConstructionError::NullDirection)?;
        Ok(Frame2D {
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
    /// use geomcore::{Frame2D, Point2D};
    /// assert_eq!(Frame2D::WORLD.origin(), Point2D::ORIGIN);
    /// ```
    pub fn origin(self) -> Point2D {
        self.origin
    }

    /// Returns the unit x direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame2D, Vector2D};
    /// assert_eq!(Frame2D::WORLD.x_direction(), Vector2D::X);
    /// ```
    pub fn x_direction(self) -> Vector2D {
        self.x_dir
    }

    /// Returns the unit y direction of the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Frame2D, Vector2D};
    /// assert_eq!(Frame2D::WORLD.y_direction(), Vector2D::Y);
    /// ```
    pub fn y_direction(self) -> Vector2D {
        self.y_dir
    }

    /// Returns whether the frame is direct (counterclockwise, `x × y > 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Frame2D;
    /// assert!(Frame2D::WORLD.is_direct());
    /// ```
    pub fn is_direct(self) -> bool {
        self.x_dir.cross(self.y_dir) > 0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Axis2D, Axis3D, Frame2D, Frame3D, FrameConstructionError, Point2D, Point3D, Vector2D,
        Vector3D,
    };

    #[test]
    fn test_axis3_new_unit_direction() {
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::new(2.0, 0.0, 0.0)).unwrap();
        assert_eq!(axis.origin(), Point3D::ORIGIN);
        assert_eq!(axis.direction(), Vector3D::X);
    }

    #[test]
    fn test_axis3_new_zero_direction_errors() {
        assert_eq!(
            Axis3D::new(Point3D::ORIGIN, Vector3D::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame3_world_const() {
        let f = Frame3D::WORLD;
        assert_eq!(f.origin(), Point3D::ORIGIN);
        assert_eq!(f.x_direction(), Vector3D::X);
        assert_eq!(f.y_direction(), Vector3D::Y);
        assert_eq!(f.z_direction(), Vector3D::Z);
    }

    #[test]
    fn test_frame3_new_axis_aligned_hint() {
        let f = Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X).unwrap();
        assert_eq!(f.x_direction(), Vector3D::X);
        assert_eq!(f.y_direction(), Vector3D::Y);
        assert_eq!(f.z_direction(), Vector3D::Z);
    }

    #[test]
    fn test_frame3_new_diagonal_hint_projects_perpendicular() {
        let f = Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::new(1.0, 1.0, 0.0)).unwrap();
        let expected_x = Vector3D::new(1.0, 1.0, 0.0).normalized().unwrap();
        assert!((f.x_direction().x - expected_x.x).abs() < 1e-10);
        assert!((f.x_direction().y - expected_x.y).abs() < 1e-10);
        assert!((f.x_direction().z - expected_x.z).abs() < 1e-10);
        let expected_y = Vector3D::Z.cross(f.x_direction());
        assert!((f.y_direction().x - expected_y.x).abs() < 1e-10);
        assert!((f.y_direction().y - expected_y.y).abs() < 1e-10);
        assert!((f.y_direction().z - expected_y.z).abs() < 1e-10);
    }

    #[test]
    fn test_frame3_new_parallel_hint_errors() {
        assert_eq!(
            Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::Z),
            Err(FrameConstructionError::ParallelDirections)
        );
        assert_eq!(
            Frame3D::new(Point3D::ORIGIN, Vector3D::Z, -Vector3D::Z),
            Err(FrameConstructionError::ParallelDirections)
        );
    }

    #[test]
    fn test_frame3_new_null_z_errors() {
        assert_eq!(
            Frame3D::new(Point3D::ORIGIN, Vector3D::ZERO, Vector3D::X),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame3_new_null_hint_errors() {
        assert_eq!(
            Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    fn assert_orthonormal_right_handed(f: Frame3D) {
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
        let f = Frame3D::from_z(Point3D::ORIGIN, Vector3D::Z).unwrap();
        assert_orthonormal_right_handed(f);
    }

    #[test]
    fn test_frame3_from_z_x_axis_aligned() {
        let f = Frame3D::from_z(Point3D::ORIGIN, Vector3D::X).unwrap();
        assert_orthonormal_right_handed(f);
    }

    #[test]
    fn test_frame3_from_z_diagonal() {
        let z = Vector3D::new(1.0, 1.0, 1.0).normalized().unwrap();
        let f = Frame3D::from_z(Point3D::ORIGIN, z).unwrap();
        assert_orthonormal_right_handed(f);
    }

    #[test]
    fn test_frame3_from_z_null_errors() {
        assert_eq!(
            Frame3D::from_z(Point3D::ORIGIN, Vector3D::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame3_axis() {
        let f = Frame3D::WORLD;
        let axis = f.axis();
        assert_eq!(axis.origin(), Point3D::ORIGIN);
        assert_eq!(axis.direction(), Vector3D::Z);
    }

    #[test]
    fn test_frame3_local_coordinates_and_point_at_round_trip() {
        let origin = Point3D::new(1.0, 2.0, 3.0);
        let f = Frame3D::new(
            origin,
            Vector3D::new(0.0, 0.0, 1.0),
            Vector3D::new(1.0, 1.0, 0.0),
        )
        .unwrap();
        let p = Point3D::new(5.0, -4.0, 7.0);
        let (u, v, w) = f.local_coordinates(p);
        let round_tripped = f.point_at(u, v, w);
        assert!((round_tripped.x - p.x).abs() < 1e-10);
        assert!((round_tripped.y - p.y).abs() < 1e-10);
        assert!((round_tripped.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_frame3_local_coordinates_of_origin_is_zero() {
        let f = Frame3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::Z, Vector3D::X).unwrap();
        let (u, v, w) = f.local_coordinates(f.origin());
        assert_eq!((u, v, w), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_axis2_new_unit_direction() {
        let axis = Axis2D::new(Point2D::ORIGIN, Vector2D::new(0.0, 3.0)).unwrap();
        assert_eq!(axis.origin(), Point2D::ORIGIN);
        assert_eq!(axis.direction(), Vector2D::Y);
    }

    #[test]
    fn test_axis2_new_zero_direction_errors() {
        assert_eq!(
            Axis2D::new(Point2D::ORIGIN, Vector2D::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame2_world_const() {
        let f = Frame2D::WORLD;
        assert_eq!(f.origin(), Point2D::ORIGIN);
        assert_eq!(f.x_direction(), Vector2D::X);
        assert_eq!(f.y_direction(), Vector2D::Y);
    }

    #[test]
    fn test_frame2_new_orthogonal_ok() {
        let f = Frame2D::new(Point2D::ORIGIN, Vector2D::X, Vector2D::Y).unwrap();
        assert_eq!(f.x_direction(), Vector2D::X);
        assert_eq!(f.y_direction(), Vector2D::Y);
    }

    #[test]
    fn test_frame2_new_non_orthogonal_errors() {
        assert_eq!(
            Frame2D::new(Point2D::ORIGIN, Vector2D::X, Vector2D::new(1.0, 1.0)),
            Err(FrameConstructionError::NotOrthogonal)
        );
    }

    #[test]
    fn test_frame2_new_null_direction_errors() {
        assert_eq!(
            Frame2D::new(Point2D::ORIGIN, Vector2D::ZERO, Vector2D::Y),
            Err(FrameConstructionError::NullDirection)
        );
        assert_eq!(
            Frame2D::new(Point2D::ORIGIN, Vector2D::X, Vector2D::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame2_from_x() {
        let f = Frame2D::from_x(Point2D::ORIGIN, Vector2D::new(0.6, 0.8)).unwrap();
        assert!((f.x_direction().x - 0.6).abs() < 1e-10);
        assert!((f.x_direction().y - 0.8).abs() < 1e-10);
        assert!((f.y_direction().x - (-0.8)).abs() < 1e-10);
        assert!((f.y_direction().y - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_frame2_from_x_null_errors() {
        assert_eq!(
            Frame2D::from_x(Point2D::ORIGIN, Vector2D::ZERO),
            Err(FrameConstructionError::NullDirection)
        );
    }

    #[test]
    fn test_frame2_is_direct_true_for_world() {
        assert!(Frame2D::WORLD.is_direct());
    }

    #[test]
    fn test_frame2_is_direct_false_for_left_handed() {
        let f = Frame2D::new(Point2D::ORIGIN, Vector2D::X, -Vector2D::Y).unwrap();
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
