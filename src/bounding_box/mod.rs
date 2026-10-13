//! Axis-aligned bounding boxes of curves and surfaces (see `CONTEXT.md`).
//!
//! [`BoundingBox3D`] is the box; [`Bound3D`] asks a curve or surface for
//! it. Elementary shapes get their tightest box, touching the shape on
//! every side: a circle or ellipse arc from its ends and the angles where a
//! coordinate peaks within its range, a trimmed parabola or hyperbola from
//! its ends and the parameters where a coordinate is stationary, a sphere
//! or torus from its radii and the tilt of its axis. B-splines get the box
//! of their control points, which holds the shape (weights are positive,
//! so the shape lies in the convex hull of its poles) but may be loose; a
//! trimmed B-spline gets the box of the poles of the knot spans its range
//! covers and of the Bézier control points of its pieces of the spans it
//! cuts. Unbounded shapes (lines, parabolas, hyperbolas, planes,
//! cylinders, cones) have no box; trimmed curves always have one.
//!
//! ```
//! use geomcore::{Bound3D, Circle3D, Line3D, Point3D, TrimmedCurve3D, Vector3D};
//!
//! let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
//! let quarter = TrimmedCurve3D::new(circle, 0.0, std::f64::consts::FRAC_PI_2).unwrap();
//! let b = quarter.bounding_box().unwrap();
//! // From (2, 0, 0) to (0, 2, 0): the arc's own extreme points, not the circle's.
//! assert_eq!((b.min().y, b.max().y), (0.0, 2.0));
//! assert_eq!(b.max().x, 2.0);
//! assert!(b.min().x.abs() < 1e-15);
//!
//! let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
//! assert_eq!(line.bounding_box(), None);
//! ```

mod curves;
mod surfaces;

use crate::{Point3D, Vector3D};

/// An axis-aligned box: the points whose coordinates all lie between
/// those of its [`min`](BoundingBox3D::min) and
/// [`max`](BoundingBox3D::max) corners, faces included.
///
/// A box is never empty: `min` is at most `max` on every axis. A box of a
/// single point is flat on every axis.
///
/// # Examples
///
/// ```
/// use geomcore::{BoundingBox3D, Point3D};
///
/// let a = BoundingBox3D::new(Point3D::new(0.0, 0.0, 0.0), Point3D::new(2.0, 1.0, 1.0));
/// let b = BoundingBox3D::new(Point3D::new(3.0, 0.0, 0.0), Point3D::new(4.0, 1.0, 1.0));
/// assert!(!a.overlaps(b));
/// // Within a distance of 1 of each other: the boxes enlarged by half of it touch.
/// assert!(a.enlarged(0.5).overlaps(b.enlarged(0.5)));
/// assert_eq!(a.union(b).max(), Point3D::new(4.0, 1.0, 1.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox3D {
    min: Point3D,
    max: Point3D,
}

fn coordinates(p: Point3D) -> [f64; 3] {
    [p.x, p.y, p.z]
}

fn point([x, y, z]: [f64; 3]) -> Point3D {
    Point3D::new(x, y, z)
}

impl BoundingBox3D {
    /// The box with opposite corners `a` and `b`, in any order.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::new(1.0, 0.0, 5.0), Point3D::new(0.0, 2.0, 3.0));
    /// assert_eq!(b.min(), Point3D::new(0.0, 0.0, 3.0));
    /// assert_eq!(b.max(), Point3D::new(1.0, 2.0, 5.0));
    /// ```
    pub fn new(a: Point3D, b: Point3D) -> BoundingBox3D {
        BoundingBox3D {
            min: Point3D::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
            max: Point3D::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
        }
    }

    /// The smallest box holding every point of `points`, or `None` if there
    /// are none.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let points = [Point3D::new(1.0, 0.0, 0.0), Point3D::new(-1.0, 2.0, 0.5)];
    /// let b = BoundingBox3D::from_points(points).unwrap();
    /// assert_eq!(b.min(), Point3D::new(-1.0, 0.0, 0.0));
    /// assert_eq!(b.max(), Point3D::new(1.0, 2.0, 0.5));
    /// assert_eq!(BoundingBox3D::from_points([]), None);
    /// ```
    pub fn from_points(points: impl IntoIterator<Item = Point3D>) -> Option<BoundingBox3D> {
        let mut points = points.into_iter();
        let first = points.next()?;
        Some(points.fold(BoundingBox3D::new(first, first), BoundingBox3D::including))
    }

    /// The corner with the smallest coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::new(1.0, 2.0, 3.0), Point3D::ORIGIN);
    /// assert_eq!(b.min(), Point3D::ORIGIN);
    /// ```
    pub fn min(self) -> Point3D {
        self.min
    }

    /// The corner with the largest coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::new(1.0, 2.0, 3.0), Point3D::ORIGIN);
    /// assert_eq!(b.max(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn max(self) -> Point3D {
        self.max
    }

    /// The middle of the box.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(2.0, 4.0, 6.0));
    /// assert_eq!(b.center(), Point3D::new(1.0, 2.0, 3.0));
    /// ```
    pub fn center(self) -> Point3D {
        self.min + 0.5 * (self.max - self.min)
    }

    /// The vector from the [`min`](Self::min) corner to the
    /// [`max`](Self::max) corner: the box's size along each axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D, Vector3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(3.0, 4.0, 0.0));
    /// assert_eq!(b.diagonal(), Vector3D::new(3.0, 4.0, 0.0));
    /// assert_eq!(b.diagonal().magnitude(), 5.0);
    /// ```
    pub fn diagonal(self) -> Vector3D {
        self.max - self.min
    }

    /// The smallest box holding both boxes.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let a = BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(1.0, 1.0, 1.0));
    /// let b = BoundingBox3D::new(Point3D::new(2.0, -1.0, 0.0), Point3D::new(3.0, 0.0, 0.5));
    /// let both = a.union(b);
    /// assert_eq!(both.min(), Point3D::new(0.0, -1.0, 0.0));
    /// assert_eq!(both.max(), Point3D::new(3.0, 1.0, 1.0));
    /// ```
    pub fn union(self, other: BoundingBox3D) -> BoundingBox3D {
        self.including(other.min).including(other.max)
    }

    /// The box grown by `distance` on every side, for instance by a
    /// tolerance before an overlap test. A negative `distance` shrinks it,
    /// an axis shorter than twice the shrink collapsing to its middle.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(4.0, 1.0, 1.0));
    /// let grown = b.enlarged(0.5);
    /// assert_eq!(grown.min(), Point3D::new(-0.5, -0.5, -0.5));
    /// assert_eq!(grown.max(), Point3D::new(4.5, 1.5, 1.5));
    ///
    /// let shrunk = b.enlarged(-1.0);
    /// assert_eq!(shrunk.min(), Point3D::new(1.0, 0.5, 0.5));
    /// assert_eq!(shrunk.max(), Point3D::new(3.0, 0.5, 0.5));
    /// ```
    pub fn enlarged(self, distance: f64) -> BoundingBox3D {
        let (min, max) = (coordinates(self.min), coordinates(self.max));
        let (mut lo, mut hi) = (min, max);
        for k in 0..3 {
            lo[k] = min[k] - distance;
            hi[k] = max[k] + distance;
            if lo[k] > hi[k] {
                let middle = 0.5 * (min[k] + max[k]);
                (lo[k], hi[k]) = (middle, middle);
            }
        }
        BoundingBox3D {
            min: point(lo),
            max: point(hi),
        }
    }

    /// Whether `point` is in the box, faces included.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let b = BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(1.0, 1.0, 1.0));
    /// assert!(b.encloses(Point3D::new(0.5, 0.5, 0.5)));
    /// assert!(b.encloses(Point3D::new(1.0, 0.0, 0.5)));
    /// assert!(!b.encloses(Point3D::new(1.5, 0.5, 0.5)));
    /// ```
    pub fn encloses(self, point: Point3D) -> bool {
        let (p, min, max) = (
            coordinates(point),
            coordinates(self.min),
            coordinates(self.max),
        );
        (0..3).all(|k| min[k] <= p[k] && p[k] <= max[k])
    }

    /// Whether the two boxes share a point, faces included: boxes that
    /// only touch overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BoundingBox3D, Point3D};
    ///
    /// let a = BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(1.0, 1.0, 1.0));
    /// let touching = BoundingBox3D::new(Point3D::new(1.0, 0.0, 0.0), Point3D::new(2.0, 1.0, 1.0));
    /// let apart = BoundingBox3D::new(Point3D::new(1.5, 0.0, 0.0), Point3D::new(2.0, 1.0, 1.0));
    /// assert!(a.overlaps(touching));
    /// assert!(!a.overlaps(apart));
    /// ```
    pub fn overlaps(self, other: BoundingBox3D) -> bool {
        let (a_min, a_max) = (coordinates(self.min), coordinates(self.max));
        let (b_min, b_max) = (coordinates(other.min), coordinates(other.max));
        (0..3).all(|k| a_min[k] <= b_max[k] && b_min[k] <= a_max[k])
    }

    /// The smallest box holding this box and `point`.
    pub(crate) fn including(self, p: Point3D) -> BoundingBox3D {
        BoundingBox3D {
            min: Point3D::new(
                self.min.x.min(p.x),
                self.min.y.min(p.y),
                self.min.z.min(p.z),
            ),
            max: Point3D::new(
                self.max.x.max(p.x),
                self.max.y.max(p.y),
                self.max.z.max(p.z),
            ),
        }
    }
}

/// The axis-aligned box bounding a curve or surface, or `None` for an
/// unbounded one.
///
/// Elementary curves and surfaces report their tightest box, which touches
/// the shape on every side (up to the rounding of evaluating it).
/// B-splines report a box holding the whole shape, from their control
/// points, which may be larger than the tightest. Lines, parabolas,
/// hyperbolas, planes, cylinders and cones are unbounded and report
/// `None`; a trimmed curve always has a box, over its range.
///
/// # Examples
///
/// ```
/// use geomcore::{Bound3D, Plane, Point3D, Sphere, Vector3D};
///
/// let sphere = Sphere::new(Point3D::new(1.0, 0.0, 0.0), 2.0).unwrap();
/// let b = sphere.bounding_box().unwrap();
/// assert_eq!(b.min(), Point3D::new(-1.0, -2.0, -2.0));
/// assert_eq!(b.max(), Point3D::new(3.0, 2.0, 2.0));
///
/// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
/// assert_eq!(plane.bounding_box(), None);
/// ```
pub trait Bound3D {
    /// The axis-aligned box bounding the shape, or `None` if the shape is
    /// unbounded.
    fn bounding_box(&self) -> Option<BoundingBox3D>;
}

impl<T: Bound3D + ?Sized> Bound3D for &T {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        (**self).bounding_box()
    }
}

pub(crate) mod sealed {
    use super::BoundingBox3D;

    /// The box of a range of a curve: what a trimmed curve needs from its
    /// basis. Implemented by every curve, bounded or not.
    pub trait BoundRange3D: crate::curves::ParametricCurve3D {
        /// The box bounding the curve over `[first, last]`, a finite range
        /// with `first < last`, at most one period long on a periodic
        /// curve and anywhere on it.
        fn range_box(&self, first: f64, last: f64) -> BoundingBox3D;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit() -> BoundingBox3D {
        BoundingBox3D::new(Point3D::ORIGIN, Point3D::new(1.0, 1.0, 1.0))
    }

    #[test]
    fn test_new_orders_the_corners() {
        let b = BoundingBox3D::new(Point3D::new(1.0, -1.0, 2.0), Point3D::new(-3.0, 4.0, 2.0));
        assert_eq!(b.min(), Point3D::new(-3.0, -1.0, 2.0));
        assert_eq!(b.max(), Point3D::new(1.0, 4.0, 2.0));
    }

    #[test]
    fn test_from_points_of_one_point_is_flat() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let b = BoundingBox3D::from_points([p]).unwrap();
        assert_eq!((b.min(), b.max()), (p, p));
        assert_eq!(b.diagonal(), Vector3D::ZERO);
        assert!(b.encloses(p));
    }

    #[test]
    fn test_union_is_symmetric_and_holds_both() {
        let a = unit();
        let b = BoundingBox3D::new(Point3D::new(-2.0, 0.5, 0.5), Point3D::new(-1.0, 3.0, 0.7));
        assert_eq!(a.union(b), b.union(a));
        let both = a.union(b);
        for corner in [a.min(), a.max(), b.min(), b.max()] {
            assert!(both.encloses(corner));
        }
        assert_eq!(a.union(a), a);
    }

    #[test]
    fn test_enlarged_grows_and_shrinks() {
        let b = unit().enlarged(0.25);
        assert_eq!(b.min(), Point3D::new(-0.25, -0.25, -0.25));
        assert_eq!(b.max(), Point3D::new(1.25, 1.25, 1.25));
        assert_eq!(b.enlarged(-0.25), unit());
        let collapsed = unit().enlarged(-2.0);
        assert_eq!(collapsed.min(), Point3D::new(0.5, 0.5, 0.5));
        assert_eq!(collapsed.max(), Point3D::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_encloses_counts_faces_and_rejects_each_axis() {
        let b = unit();
        assert!(b.encloses(Point3D::ORIGIN));
        assert!(b.encloses(Point3D::new(1.0, 1.0, 1.0)));
        for outside in [
            Point3D::new(-1e-12, 0.5, 0.5),
            Point3D::new(0.5, 1.0 + 1e-12, 0.5),
            Point3D::new(0.5, 0.5, 2.0),
        ] {
            assert!(!b.encloses(outside));
        }
    }

    #[test]
    fn test_overlaps_needs_every_axis() {
        let b = unit();
        assert!(b.overlaps(b));
        let inner = BoundingBox3D::new(Point3D::new(0.2, 0.2, 0.2), Point3D::new(0.3, 0.3, 0.3));
        assert!(b.overlaps(inner) && inner.overlaps(b));
        let corner = BoundingBox3D::new(Point3D::new(1.0, 1.0, 1.0), Point3D::new(2.0, 2.0, 2.0));
        assert!(b.overlaps(corner));
        // Overlapping on x and y but not z.
        let above = BoundingBox3D::new(Point3D::new(0.5, 0.5, 1.5), Point3D::new(2.0, 2.0, 2.0));
        assert!(!b.overlaps(above) && !above.overlaps(b));
    }

    #[test]
    fn test_center_and_diagonal() {
        let b = BoundingBox3D::new(Point3D::new(-1.0, 0.0, 2.0), Point3D::new(3.0, 2.0, 2.0));
        assert_eq!(b.center(), Point3D::new(1.0, 1.0, 2.0));
        assert_eq!(b.diagonal(), Vector3D::new(4.0, 2.0, 0.0));
    }
}
