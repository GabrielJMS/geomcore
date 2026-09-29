use crate::vector::{Vector2D, Vector3D};

/// Two-dimensional point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2D {
    /// x-coordinate
    pub x: f64,
    /// y-coordinate
    pub y: f64,
}

impl Point2D {
    /// Origin (0, 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point2D;
    /// assert_eq!(Point2D::ORIGIN, Point2D::new(0.0, 0.0));
    /// ```
    pub const ORIGIN: Point2D = Point2D { x: 0.0, y: 0.0 };

    /// Create a new 2D point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point2D;
    /// let p = Point2D::new(1.0, 2.0);
    /// assert_eq!(p.x, 1.0);
    /// assert_eq!(p.y, 2.0);
    /// ```
    pub const fn new(x: f64, y: f64) -> Point2D {
        Point2D { x, y }
    }

    /// Distance to another point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point2D;
    /// assert_eq!(Point2D::new(0.0, 0.0).distance(Point2D::new(3.0, 4.0)), 5.0);
    /// ```
    pub fn distance(self, other: Point2D) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Three-dimensional point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point3D {
    /// x-coordinate
    pub x: f64,
    /// y-coordinate
    pub y: f64,
    /// z-coordinate
    pub z: f64,
}

impl Point3D {
    /// Origin (0, 0, 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point3D;
    /// assert_eq!(Point3D::ORIGIN, Point3D::new(0.0, 0.0, 0.0));
    /// ```
    pub const ORIGIN: Point3D = Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Create a new 3D point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point3D;
    /// let p = Point3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(p.x, 1.0);
    /// assert_eq!(p.y, 2.0);
    /// assert_eq!(p.z, 3.0);
    /// ```
    pub const fn new(x: f64, y: f64, z: f64) -> Point3D {
        Point3D { x, y, z }
    }

    /// Distance to another point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point3D;
    /// assert_eq!(Point3D::new(1.0, 2.0, 3.0).distance(Point3D::new(4.0, 6.0, 3.0)), 5.0);
    /// ```
    pub fn distance(self, other: Point3D) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// Point3D + Vector3D -> Point3D
impl std::ops::Add<Vector3D> for Point3D {
    type Output = Point3D;

    fn add(self, vector: Vector3D) -> Point3D {
        Point3D {
            x: self.x + vector.x,
            y: self.y + vector.y,
            z: self.z + vector.z,
        }
    }
}

// Point3D - Vector3D -> Point3D
impl std::ops::Sub<Vector3D> for Point3D {
    type Output = Point3D;

    fn sub(self, vector: Vector3D) -> Point3D {
        Point3D {
            x: self.x - vector.x,
            y: self.y - vector.y,
            z: self.z - vector.z,
        }
    }
}

// Point3D - Point3D -> Vector3D
impl std::ops::Sub for Point3D {
    type Output = Vector3D;

    fn sub(self, other: Point3D) -> Vector3D {
        Vector3D {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

// Point2D + Vector2D -> Point2D
impl std::ops::Add<Vector2D> for Point2D {
    type Output = Point2D;

    fn add(self, vector: Vector2D) -> Point2D {
        Point2D {
            x: self.x + vector.x,
            y: self.y + vector.y,
        }
    }
}

// Point2D - Vector2D -> Point2D
impl std::ops::Sub<Vector2D> for Point2D {
    type Output = Point2D;

    fn sub(self, vector: Vector2D) -> Point2D {
        Point2D {
            x: self.x - vector.x,
            y: self.y - vector.y,
        }
    }
}

// Point2D - Point2D -> Vector2D
impl std::ops::Sub for Point2D {
    type Output = Vector2D;

    fn sub(self, other: Point2D) -> Vector2D {
        Vector2D {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3_origin() {
        assert_eq!(Point3D::ORIGIN, Point3D::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_point3_new() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_point3_distance() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(4.0, 6.0, 3.0);
        assert_eq!(p1.distance(p2), 5.0);
    }

    #[test]
    fn test_point3_add_vector() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let v = Vector3D::new(4.0, 5.0, 6.0);
        assert_eq!(p + v, Point3D::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_point3_sub_vector() {
        let p = Point3D::new(5.0, 7.0, 9.0);
        let v = Vector3D::new(4.0, 5.0, 6.0);
        assert_eq!(p - v, Point3D::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_point3_sub_point() {
        let p1 = Point3D::new(4.0, 6.0, 8.0);
        let p2 = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p1 - p2, Vector3D::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn test_point3_op_identity() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let q = Point3D::new(4.0, 5.0, 6.0);
        let diff = q - p;
        assert_eq!(p + diff, q);
    }

    #[test]
    fn test_point2_origin() {
        assert_eq!(Point2D::ORIGIN, Point2D::new(0.0, 0.0));
    }

    #[test]
    fn test_point2_new() {
        let p = Point2D::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_point2_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert_eq!(p1.distance(p2), 5.0);
    }

    #[test]
    fn test_point2_add_vector() {
        let p = Point2D::new(1.0, 2.0);
        let v = Vector2D::new(3.0, 4.0);
        assert_eq!(p + v, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_point2_sub_vector() {
        let p = Point2D::new(4.0, 6.0);
        let v = Vector2D::new(1.0, 2.0);
        assert_eq!(p - v, Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_point2_sub_point() {
        let p1 = Point2D::new(4.0, 6.0);
        let p2 = Point2D::new(1.0, 2.0);
        assert_eq!(p1 - p2, Vector2D::new(3.0, 4.0));
    }

    #[test]
    fn test_point2_op_identity() {
        let p = Point2D::new(1.0, 2.0);
        let q = Point2D::new(4.0, 6.0);
        let diff = q - p;
        assert_eq!(p + diff, q);
    }
}
