/// Two-dimensional vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector2D {
    /// x-component
    pub x: f64,
    /// y-component
    pub y: f64,
}

impl Vector2D {
    /// Zero vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::ZERO, Vector2D::new(0.0, 0.0));
    /// ```
    pub const ZERO: Vector2D = Vector2D { x: 0.0, y: 0.0 };

    /// Unit vector along x-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::X, Vector2D::new(1.0, 0.0));
    /// ```
    pub const X: Vector2D = Vector2D { x: 1.0, y: 0.0 };

    /// Unit vector along y-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::Y, Vector2D::new(0.0, 1.0));
    /// ```
    pub const Y: Vector2D = Vector2D { x: 0.0, y: 1.0 };

    /// Create a new 2D vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// let v = Vector2D::new(3.0, 4.0);
    /// assert_eq!(v.x, 3.0);
    /// assert_eq!(v.y, 4.0);
    /// ```
    pub const fn new(x: f64, y: f64) -> Vector2D {
        Vector2D { x, y }
    }

    /// Dot product.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::X.dot(Vector2D::Y), 0.0);
    /// assert_eq!(Vector2D::X.dot(Vector2D::X), 1.0);
    /// ```
    pub fn dot(self, other: Vector2D) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Cross product (scalar z-component).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::X.cross(Vector2D::Y), 1.0);
    /// assert_eq!(Vector2D::Y.cross(Vector2D::X), -1.0);
    /// ```
    pub fn cross(self, other: Vector2D) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Perpendicular vector (counterclockwise 90 degree rotation).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::X.perp(), Vector2D::Y);
    /// assert_eq!(Vector2D::Y.perp(), Vector2D::new(-1.0, 0.0));
    /// ```
    pub fn perp(self) -> Vector2D {
        Vector2D {
            x: -self.y,
            y: self.x,
        }
    }

    /// Magnitude (Euclidean length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::new(3.0, 4.0).magnitude(), 5.0);
    /// ```
    pub fn magnitude(self) -> f64 {
        self.square_magnitude().sqrt()
    }

    /// Squared magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// assert_eq!(Vector2D::new(3.0, 4.0).square_magnitude(), 25.0);
    /// ```
    pub fn square_magnitude(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Normalized vector.
    ///
    /// Returns `None` if the squared magnitude is below `f64::MIN_POSITIVE`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector2D;
    /// let v = Vector2D::new(3.0, 4.0);
    /// let normalized = v.normalized().unwrap();
    /// assert!((normalized.x - 0.6).abs() < 1e-10);
    /// assert!((normalized.y - 0.8).abs() < 1e-10);
    /// ```
    pub fn normalized(self) -> Option<Vector2D> {
        let sq_mag = self.square_magnitude();
        if sq_mag <= f64::MIN_POSITIVE {
            None
        } else {
            let mag = sq_mag.sqrt();
            Some(Vector2D {
                x: self.x / mag,
                y: self.y / mag,
            })
        }
    }
}

/// Three-dimensional vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector3D {
    /// x-component
    pub x: f64,
    /// y-component
    pub y: f64,
    /// z-component
    pub z: f64,
}

impl Vector3D {
    /// Zero vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::ZERO, Vector3D::new(0.0, 0.0, 0.0));
    /// ```
    pub const ZERO: Vector3D = Vector3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit vector along x-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::X, Vector3D::new(1.0, 0.0, 0.0));
    /// ```
    pub const X: Vector3D = Vector3D {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit vector along y-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::Y, Vector3D::new(0.0, 1.0, 0.0));
    /// ```
    pub const Y: Vector3D = Vector3D {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Unit vector along z-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::Z, Vector3D::new(0.0, 0.0, 1.0));
    /// ```
    pub const Z: Vector3D = Vector3D {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Create a new 3D vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// let v = Vector3D::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 2.0);
    /// assert_eq!(v.z, 3.0);
    /// ```
    pub const fn new(x: f64, y: f64, z: f64) -> Vector3D {
        Vector3D { x, y, z }
    }

    /// Dot product.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::X.dot(Vector3D::Y), 0.0);
    /// assert_eq!(Vector3D::X.dot(Vector3D::X), 1.0);
    /// ```
    pub fn dot(self, other: Vector3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::X.cross(Vector3D::Y), Vector3D::Z);
    /// assert_eq!(Vector3D::Y.cross(Vector3D::Z), Vector3D::X);
    /// assert_eq!(Vector3D::Z.cross(Vector3D::X), Vector3D::Y);
    /// ```
    pub fn cross(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Magnitude (Euclidean length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert!((Vector3D::new(1.0, 2.0, 2.0).magnitude() - 3.0).abs() < 1e-10);
    /// ```
    pub fn magnitude(self) -> f64 {
        self.square_magnitude().sqrt()
    }

    /// Squared magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// assert_eq!(Vector3D::new(1.0, 2.0, 2.0).square_magnitude(), 9.0);
    /// ```
    pub fn square_magnitude(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalized vector.
    ///
    /// Returns `None` if the squared magnitude is below `f64::MIN_POSITIVE`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Vector3D;
    /// let v = Vector3D::new(3.0, 4.0, 0.0);
    /// let normalized = v.normalized().unwrap();
    /// assert!((normalized.x - 0.6).abs() < 1e-10);
    /// assert!((normalized.y - 0.8).abs() < 1e-10);
    /// assert_eq!(normalized.z, 0.0);
    /// ```
    pub fn normalized(self) -> Option<Vector3D> {
        let sq_mag = self.square_magnitude();
        if sq_mag <= f64::MIN_POSITIVE {
            None
        } else {
            let mag = sq_mag.sqrt();
            Some(Vector3D {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            })
        }
    }

    /// Signed angle from self to other around reference (right-hand rule), in (-π, π].
    ///
    /// Uses `atan2(reference.normalized · (self × other), self · other)`.
    ///
    /// # Preconditions
    ///
    /// The `reference` vector must be non-zero. If `reference` is a zero vector (squared magnitude
    /// ≤ `f64::MIN_POSITIVE`), the function cannot determine the sign and returns the unsigned angle
    /// in [0, π]: `atan2(0.0, self · other)`. This collapses to 0 for `self · other > 0` and π for
    /// `self · other < 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use geomcore::Vector3D;
    /// let angle = Vector3D::X.angle_with_ref(Vector3D::Y, Vector3D::Z);
    /// assert!((angle - PI / 2.0).abs() < 1e-10);
    /// ```
    pub fn angle_with_ref(self, other: Vector3D, reference: Vector3D) -> f64 {
        let cross_prod = self.cross(other);
        let dot_prod = self.dot(other);

        match reference.normalized() {
            Some(ref_normalized) => {
                let signed_mag = ref_normalized.dot(cross_prod);
                signed_mag.atan2(dot_prod)
            }
            None => {
                // Degenerate reference: sign is unknowable; collapse to the unsigned 0/π case
                f64::atan2(0.0, dot_prod)
            }
        }
    }
}

// Implement arithmetic operations for Vector3D
impl std::ops::Add for Vector3D {
    type Output = Vector3D;

    fn add(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for Vector3D {
    type Output = Vector3D;

    fn sub(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Neg for Vector3D {
    type Output = Vector3D;

    fn neg(self) -> Vector3D {
        Vector3D {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul<f64> for Vector3D {
    type Output = Vector3D;

    fn mul(self, scalar: f64) -> Vector3D {
        Vector3D {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Mul<Vector3D> for f64 {
    type Output = Vector3D;

    fn mul(self, vector: Vector3D) -> Vector3D {
        Vector3D {
            x: self * vector.x,
            y: self * vector.y,
            z: self * vector.z,
        }
    }
}

// Implement arithmetic operations for Vector2D
impl std::ops::Add for Vector2D {
    type Output = Vector2D;

    fn add(self, other: Vector2D) -> Vector2D {
        Vector2D {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vector2D {
    type Output = Vector2D;

    fn sub(self, other: Vector2D) -> Vector2D {
        Vector2D {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Neg for Vector2D {
    type Output = Vector2D;

    fn neg(self) -> Vector2D {
        Vector2D {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl std::ops::Mul<f64> for Vector2D {
    type Output = Vector2D;

    fn mul(self, scalar: f64) -> Vector2D {
        Vector2D {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl std::ops::Mul<Vector2D> for f64 {
    type Output = Vector2D;

    fn mul(self, vector: Vector2D) -> Vector2D {
        Vector2D {
            x: self * vector.x,
            y: self * vector.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_vector3_constants() {
        assert_eq!(Vector3D::ZERO, Vector3D::new(0.0, 0.0, 0.0));
        assert_eq!(Vector3D::X, Vector3D::new(1.0, 0.0, 0.0));
        assert_eq!(Vector3D::Y, Vector3D::new(0.0, 1.0, 0.0));
        assert_eq!(Vector3D::Z, Vector3D::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vector3_new() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vector3_dot() {
        assert_eq!(Vector3D::X.dot(Vector3D::Y), 0.0);
        assert_eq!(Vector3D::X.dot(Vector3D::X), 1.0);
        assert_eq!(Vector3D::X.dot(Vector3D::Z), 0.0);
    }

    #[test]
    fn test_vector3_cross() {
        assert_eq!(Vector3D::X.cross(Vector3D::Y), Vector3D::Z);
        assert_eq!(Vector3D::Y.cross(Vector3D::Z), Vector3D::X);
        assert_eq!(Vector3D::Z.cross(Vector3D::X), Vector3D::Y);
    }

    #[test]
    fn test_vector3_magnitude() {
        assert_eq!(Vector3D::new(3.0, 4.0, 0.0).magnitude(), 5.0);
        assert!((Vector3D::new(1.0, 2.0, 2.0).magnitude() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_square_magnitude() {
        assert_eq!(Vector3D::new(3.0, 4.0, 0.0).square_magnitude(), 25.0);
        assert_eq!(Vector3D::new(1.0, 2.0, 2.0).square_magnitude(), 9.0);
    }

    #[test]
    fn test_vector3_normalized() {
        let v = Vector3D::new(3.0, 4.0, 0.0);
        let norm = v.normalized().unwrap();
        assert!((norm.x - 0.6).abs() < 1e-10);
        assert!((norm.y - 0.8).abs() < 1e-10);
        assert_eq!(norm.z, 0.0);
    }

    #[test]
    fn test_vector3_normalized_zero_returns_none() {
        assert_eq!(Vector3D::ZERO.normalized(), None);
    }

    #[test]
    fn test_vector3_normalized_min_positive_boundary() {
        // Boundary test: exactly at f64::MIN_POSITIVE should return None
        let v_at_boundary = Vector3D::new(f64::MIN_POSITIVE.sqrt(), 0.0, 0.0);
        assert_eq!(v_at_boundary.normalized(), None);

        // Boundary test: slightly above f64::MIN_POSITIVE should return Some
        let v_above_boundary = Vector3D::new((f64::MIN_POSITIVE * 4.0).sqrt(), 0.0, 0.0);
        let normalized = v_above_boundary.normalized();
        assert!(normalized.is_some());
        let n = normalized.unwrap();
        assert!((n.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_angle_with_ref() {
        let angle = Vector3D::X.angle_with_ref(Vector3D::Y, Vector3D::Z);
        assert!((angle - PI / 2.0).abs() < 1e-10);

        let angle2 = Vector3D::Y.angle_with_ref(Vector3D::X, Vector3D::Z);
        assert!((angle2 + PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_angle_with_ref_degenerate_reference() {
        // Degenerate reference (zero vector): should return unsigned angle
        // For orthogonal vectors (dot product = 0), result should be 0
        let angle1 = Vector3D::X.angle_with_ref(Vector3D::Y, Vector3D::ZERO);
        assert!((angle1 - 0.0).abs() < 1e-10);

        // For opposite vectors (dot product < 0), result should be π
        let angle2 = Vector3D::X.angle_with_ref(-Vector3D::X, Vector3D::ZERO);
        assert!((angle2 - PI).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_add() {
        let a = Vector3D::new(1.0, 2.0, 3.0);
        let b = Vector3D::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vector3D::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector3_sub() {
        let a = Vector3D::new(4.0, 5.0, 6.0);
        let b = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(a - b, Vector3D::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector3_neg() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(-v, Vector3D::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_vector3_mul_scalar() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(v * 2.0, Vector3D::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector3_scalar_mul() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(2.0 * v, Vector3D::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector2_constants() {
        assert_eq!(Vector2D::ZERO, Vector2D::new(0.0, 0.0));
        assert_eq!(Vector2D::X, Vector2D::new(1.0, 0.0));
        assert_eq!(Vector2D::Y, Vector2D::new(0.0, 1.0));
    }

    #[test]
    fn test_vector2_new() {
        let v = Vector2D::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vector2_dot() {
        assert_eq!(Vector2D::X.dot(Vector2D::Y), 0.0);
        assert_eq!(Vector2D::X.dot(Vector2D::X), 1.0);
    }

    #[test]
    fn test_vector2_cross() {
        assert_eq!(Vector2D::X.cross(Vector2D::Y), 1.0);
        assert_eq!(Vector2D::Y.cross(Vector2D::X), -1.0);
    }

    #[test]
    fn test_vector2_perp() {
        assert_eq!(Vector2D::X.perp(), Vector2D::Y);
        assert_eq!(Vector2D::Y.perp(), Vector2D::new(-1.0, 0.0));
    }

    #[test]
    fn test_vector2_magnitude() {
        assert_eq!(Vector2D::new(3.0, 4.0).magnitude(), 5.0);
    }

    #[test]
    fn test_vector2_square_magnitude() {
        assert_eq!(Vector2D::new(3.0, 4.0).square_magnitude(), 25.0);
    }

    #[test]
    fn test_vector2_normalized() {
        let v = Vector2D::new(3.0, 4.0);
        let norm = v.normalized().unwrap();
        assert!((norm.x - 0.6).abs() < 1e-10);
        assert!((norm.y - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_normalized_zero_returns_none() {
        assert_eq!(Vector2D::ZERO.normalized(), None);
    }

    #[test]
    fn test_vector2_normalized_min_positive_boundary() {
        // Boundary test: exactly at f64::MIN_POSITIVE should return None
        let v_at_boundary = Vector2D::new(f64::MIN_POSITIVE.sqrt(), 0.0);
        assert_eq!(v_at_boundary.normalized(), None);

        // Boundary test: slightly above f64::MIN_POSITIVE should return Some
        let v_above_boundary = Vector2D::new((f64::MIN_POSITIVE * 4.0).sqrt(), 0.0);
        let normalized = v_above_boundary.normalized();
        assert!(normalized.is_some());
        let n = normalized.unwrap();
        assert!((n.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_add() {
        let a = Vector2D::new(1.0, 2.0);
        let b = Vector2D::new(3.0, 4.0);
        assert_eq!(a + b, Vector2D::new(4.0, 6.0));
    }

    #[test]
    fn test_vector2_sub() {
        let a = Vector2D::new(4.0, 6.0);
        let b = Vector2D::new(1.0, 2.0);
        assert_eq!(a - b, Vector2D::new(3.0, 4.0));
    }

    #[test]
    fn test_vector2_neg() {
        let v = Vector2D::new(1.0, 2.0);
        assert_eq!(-v, Vector2D::new(-1.0, -2.0));
    }

    #[test]
    fn test_vector2_mul_scalar() {
        let v = Vector2D::new(1.0, 2.0);
        assert_eq!(v * 2.0, Vector2D::new(2.0, 4.0));
    }

    #[test]
    fn test_vector2_scalar_mul() {
        let v = Vector2D::new(1.0, 2.0);
        assert_eq!(2.0 * v, Vector2D::new(2.0, 4.0));
    }
}
