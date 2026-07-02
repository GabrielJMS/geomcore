use crate::vector::{Vector2, Vector3};

/// Two-dimensional point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2 {
    /// x-coordinate
    pub x: f64,
    /// y-coordinate
    pub y: f64,
}

impl Point2 {
    /// Origin (0, 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point2;
    /// assert_eq!(Point2::ORIGIN, Point2::new(0.0, 0.0));
    /// ```
    pub const ORIGIN: Point2 = Point2 { x: 0.0, y: 0.0 };

    /// Create a new 2D point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point2;
    /// let p = Point2::new(1.0, 2.0);
    /// assert_eq!(p.x, 1.0);
    /// assert_eq!(p.y, 2.0);
    /// ```
    pub const fn new(x: f64, y: f64) -> Point2 {
        Point2 { x, y }
    }

    /// Distance to another point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point2;
    /// assert_eq!(Point2::new(0.0, 0.0).distance(Point2::new(3.0, 4.0)), 5.0);
    /// ```
    pub fn distance(self, other: Point2) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Three-dimensional point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point3 {
    /// x-coordinate
    pub x: f64,
    /// y-coordinate
    pub y: f64,
    /// z-coordinate
    pub z: f64,
}

impl Point3 {
    /// Origin (0, 0, 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point3;
    /// assert_eq!(Point3::ORIGIN, Point3::new(0.0, 0.0, 0.0));
    /// ```
    pub const ORIGIN: Point3 = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Create a new 3D point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point3;
    /// let p = Point3::new(1.0, 2.0, 3.0);
    /// assert_eq!(p.x, 1.0);
    /// assert_eq!(p.y, 2.0);
    /// assert_eq!(p.z, 3.0);
    /// ```
    pub const fn new(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    /// Distance to another point.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Point3;
    /// assert_eq!(Point3::new(1.0, 2.0, 3.0).distance(Point3::new(4.0, 6.0, 3.0)), 5.0);
    /// ```
    pub fn distance(self, other: Point3) -> f64 {
        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// Point3 + Vector3 -> Point3
impl std::ops::Add<Vector3> for Point3 {
    type Output = Point3;

    fn add(self, vector: Vector3) -> Point3 {
        Point3 {
            x: self.x + vector.x,
            y: self.y + vector.y,
            z: self.z + vector.z,
        }
    }
}

// Point3 - Vector3 -> Point3
impl std::ops::Sub<Vector3> for Point3 {
    type Output = Point3;

    fn sub(self, vector: Vector3) -> Point3 {
        Point3 {
            x: self.x - vector.x,
            y: self.y - vector.y,
            z: self.z - vector.z,
        }
    }
}

// Point3 - Point3 -> Vector3
impl std::ops::Sub for Point3 {
    type Output = Vector3;

    fn sub(self, other: Point3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

// Point2 + Vector2 -> Point2
impl std::ops::Add<Vector2> for Point2 {
    type Output = Point2;

    fn add(self, vector: Vector2) -> Point2 {
        Point2 {
            x: self.x + vector.x,
            y: self.y + vector.y,
        }
    }
}

// Point2 - Vector2 -> Point2
impl std::ops::Sub<Vector2> for Point2 {
    type Output = Point2;

    fn sub(self, vector: Vector2) -> Point2 {
        Point2 {
            x: self.x - vector.x,
            y: self.y - vector.y,
        }
    }
}

// Point2 - Point2 -> Vector2
impl std::ops::Sub for Point2 {
    type Output = Vector2;

    fn sub(self, other: Point2) -> Vector2 {
        Vector2 {
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
        assert_eq!(Point3::ORIGIN, Point3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_point3_new() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_point3_distance() {
        let p1 = Point3::new(1.0, 2.0, 3.0);
        let p2 = Point3::new(4.0, 6.0, 3.0);
        assert_eq!(p1.distance(p2), 5.0);
    }

    #[test]
    fn test_point3_add_vector() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let v = Vector3::new(4.0, 5.0, 6.0);
        assert_eq!(p + v, Point3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_point3_sub_vector() {
        let p = Point3::new(5.0, 7.0, 9.0);
        let v = Vector3::new(4.0, 5.0, 6.0);
        assert_eq!(p - v, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_point3_sub_point() {
        let p1 = Point3::new(4.0, 6.0, 8.0);
        let p2 = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p1 - p2, Vector3::new(3.0, 4.0, 5.0));
    }

    #[test]
    fn test_point3_op_identity() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let q = Point3::new(4.0, 5.0, 6.0);
        let diff = q - p;
        assert_eq!(p + diff, q);
    }

    #[test]
    fn test_point2_origin() {
        assert_eq!(Point2::ORIGIN, Point2::new(0.0, 0.0));
    }

    #[test]
    fn test_point2_new() {
        let p = Point2::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_point2_distance() {
        let p1 = Point2::new(0.0, 0.0);
        let p2 = Point2::new(3.0, 4.0);
        assert_eq!(p1.distance(p2), 5.0);
    }

    #[test]
    fn test_point2_add_vector() {
        let p = Point2::new(1.0, 2.0);
        let v = Vector2::new(3.0, 4.0);
        assert_eq!(p + v, Point2::new(4.0, 6.0));
    }

    #[test]
    fn test_point2_sub_vector() {
        let p = Point2::new(4.0, 6.0);
        let v = Vector2::new(1.0, 2.0);
        assert_eq!(p - v, Point2::new(3.0, 4.0));
    }

    #[test]
    fn test_point2_sub_point() {
        let p1 = Point2::new(4.0, 6.0);
        let p2 = Point2::new(1.0, 2.0);
        assert_eq!(p1 - p2, Vector2::new(3.0, 4.0));
    }

    #[test]
    fn test_point2_op_identity() {
        let p = Point2::new(1.0, 2.0);
        let q = Point2::new(4.0, 6.0);
        let diff = q - p;
        assert_eq!(p + diff, q);
    }
}
