/// Two-dimensional vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector2 {
    /// x-component
    pub x: f64,
    /// y-component
    pub y: f64,
}

impl Vector2 {
    /// Zero vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::ZERO, Vector2::new(0.0, 0.0));
    /// ```
    pub const ZERO: Vector2 = Vector2 { x: 0.0, y: 0.0 };

    /// Unit vector along x-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::X, Vector2::new(1.0, 0.0));
    /// ```
    pub const X: Vector2 = Vector2 { x: 1.0, y: 0.0 };

    /// Unit vector along y-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::Y, Vector2::new(0.0, 1.0));
    /// ```
    pub const Y: Vector2 = Vector2 { x: 0.0, y: 1.0 };

    /// Create a new 2D vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// let v = Vector2::new(3.0, 4.0);
    /// assert_eq!(v.x, 3.0);
    /// assert_eq!(v.y, 4.0);
    /// ```
    pub const fn new(x: f64, y: f64) -> Vector2 {
        Vector2 { x, y }
    }

    /// Dot product.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::X.dot(Vector2::Y), 0.0);
    /// assert_eq!(Vector2::X.dot(Vector2::X), 1.0);
    /// ```
    pub fn dot(self, other: Vector2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Cross product (scalar z-component).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::X.cross(Vector2::Y), 1.0);
    /// assert_eq!(Vector2::Y.cross(Vector2::X), -1.0);
    /// ```
    pub fn cross(self, other: Vector2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Perpendicular vector (counterclockwise 90 degree rotation).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::X.perp(), Vector2::Y);
    /// assert_eq!(Vector2::Y.perp(), Vector2::new(-1.0, 0.0));
    /// ```
    pub fn perp(self) -> Vector2 {
        Vector2 {
            x: -self.y,
            y: self.x,
        }
    }

    /// Magnitude (Euclidean length).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::new(3.0, 4.0).magnitude(), 5.0);
    /// ```
    pub fn magnitude(self) -> f64 {
        self.square_magnitude().sqrt()
    }

    /// Squared magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector2;
    /// assert_eq!(Vector2::new(3.0, 4.0).square_magnitude(), 25.0);
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
    /// use geomrust::Vector2;
    /// let v = Vector2::new(3.0, 4.0);
    /// let normalized = v.normalized().unwrap();
    /// assert!((normalized.x - 0.6).abs() < 1e-10);
    /// assert!((normalized.y - 0.8).abs() < 1e-10);
    /// ```
    pub fn normalized(self) -> Option<Vector2> {
        let sq_mag = self.square_magnitude();
        if sq_mag <= f64::MIN_POSITIVE {
            None
        } else {
            let mag = sq_mag.sqrt();
            Some(Vector2 {
                x: self.x / mag,
                y: self.y / mag,
            })
        }
    }
}

/// Three-dimensional vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector3 {
    /// x-component
    pub x: f64,
    /// y-component
    pub y: f64,
    /// z-component
    pub z: f64,
}

impl Vector3 {
    /// Zero vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::ZERO, Vector3::new(0.0, 0.0, 0.0));
    /// ```
    pub const ZERO: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit vector along x-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::X, Vector3::new(1.0, 0.0, 0.0));
    /// ```
    pub const X: Vector3 = Vector3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// Unit vector along y-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::Y, Vector3::new(0.0, 1.0, 0.0));
    /// ```
    pub const Y: Vector3 = Vector3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Unit vector along z-axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::Z, Vector3::new(0.0, 0.0, 1.0));
    /// ```
    pub const Z: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Create a new 3D vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// let v = Vector3::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 2.0);
    /// assert_eq!(v.z, 3.0);
    /// ```
    pub const fn new(x: f64, y: f64, z: f64) -> Vector3 {
        Vector3 { x, y, z }
    }

    /// Dot product.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::X.dot(Vector3::Y), 0.0);
    /// assert_eq!(Vector3::X.dot(Vector3::X), 1.0);
    /// ```
    pub fn dot(self, other: Vector3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::X.cross(Vector3::Y), Vector3::Z);
    /// assert_eq!(Vector3::Y.cross(Vector3::Z), Vector3::X);
    /// assert_eq!(Vector3::Z.cross(Vector3::X), Vector3::Y);
    /// ```
    pub fn cross(self, other: Vector3) -> Vector3 {
        Vector3 {
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
    /// use geomrust::Vector3;
    /// assert!((Vector3::new(1.0, 2.0, 2.0).magnitude() - 3.0).abs() < 1e-10);
    /// ```
    pub fn magnitude(self) -> f64 {
        self.square_magnitude().sqrt()
    }

    /// Squared magnitude.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::Vector3;
    /// assert_eq!(Vector3::new(1.0, 2.0, 2.0).square_magnitude(), 9.0);
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
    /// use geomrust::Vector3;
    /// let v = Vector3::new(3.0, 4.0, 0.0);
    /// let normalized = v.normalized().unwrap();
    /// assert!((normalized.x - 0.6).abs() < 1e-10);
    /// assert!((normalized.y - 0.8).abs() < 1e-10);
    /// assert_eq!(normalized.z, 0.0);
    /// ```
    pub fn normalized(self) -> Option<Vector3> {
        let sq_mag = self.square_magnitude();
        if sq_mag <= f64::MIN_POSITIVE {
            None
        } else {
            let mag = sq_mag.sqrt();
            Some(Vector3 {
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
    /// use geomrust::Vector3;
    /// let angle = Vector3::X.angle_with_ref(Vector3::Y, Vector3::Z);
    /// assert!((angle - PI / 2.0).abs() < 1e-10);
    /// ```
    pub fn angle_with_ref(self, other: Vector3, reference: Vector3) -> f64 {
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

// Implement arithmetic operations for Vector3
impl std::ops::Add for Vector3 {
    type Output = Vector3;

    fn add(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for Vector3 {
    type Output = Vector3;

    fn sub(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Neg for Vector3 {
    type Output = Vector3;

    fn neg(self) -> Vector3 {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Mul<f64> for Vector3 {
    type Output = Vector3;

    fn mul(self, scalar: f64) -> Vector3 {
        Vector3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Mul<Vector3> for f64 {
    type Output = Vector3;

    fn mul(self, vector: Vector3) -> Vector3 {
        Vector3 {
            x: self * vector.x,
            y: self * vector.y,
            z: self * vector.z,
        }
    }
}

// Implement arithmetic operations for Vector2
impl std::ops::Add for Vector2 {
    type Output = Vector2;

    fn add(self, other: Vector2) -> Vector2 {
        Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vector2 {
    type Output = Vector2;

    fn sub(self, other: Vector2) -> Vector2 {
        Vector2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Neg for Vector2 {
    type Output = Vector2;

    fn neg(self) -> Vector2 {
        Vector2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl std::ops::Mul<f64> for Vector2 {
    type Output = Vector2;

    fn mul(self, scalar: f64) -> Vector2 {
        Vector2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl std::ops::Mul<Vector2> for f64 {
    type Output = Vector2;

    fn mul(self, vector: Vector2) -> Vector2 {
        Vector2 {
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
        assert_eq!(Vector3::ZERO, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(Vector3::X, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(Vector3::Y, Vector3::new(0.0, 1.0, 0.0));
        assert_eq!(Vector3::Z, Vector3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vector3_new() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vector3_dot() {
        assert_eq!(Vector3::X.dot(Vector3::Y), 0.0);
        assert_eq!(Vector3::X.dot(Vector3::X), 1.0);
        assert_eq!(Vector3::X.dot(Vector3::Z), 0.0);
    }

    #[test]
    fn test_vector3_cross() {
        assert_eq!(Vector3::X.cross(Vector3::Y), Vector3::Z);
        assert_eq!(Vector3::Y.cross(Vector3::Z), Vector3::X);
        assert_eq!(Vector3::Z.cross(Vector3::X), Vector3::Y);
    }

    #[test]
    fn test_vector3_magnitude() {
        assert_eq!(Vector3::new(3.0, 4.0, 0.0).magnitude(), 5.0);
        assert!((Vector3::new(1.0, 2.0, 2.0).magnitude() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_square_magnitude() {
        assert_eq!(Vector3::new(3.0, 4.0, 0.0).square_magnitude(), 25.0);
        assert_eq!(Vector3::new(1.0, 2.0, 2.0).square_magnitude(), 9.0);
    }

    #[test]
    fn test_vector3_normalized() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let norm = v.normalized().unwrap();
        assert!((norm.x - 0.6).abs() < 1e-10);
        assert!((norm.y - 0.8).abs() < 1e-10);
        assert_eq!(norm.z, 0.0);
    }

    #[test]
    fn test_vector3_normalized_zero_returns_none() {
        assert_eq!(Vector3::ZERO.normalized(), None);
    }

    #[test]
    fn test_vector3_normalized_min_positive_boundary() {
        // Boundary test: exactly at f64::MIN_POSITIVE should return None
        let v_at_boundary = Vector3::new(f64::MIN_POSITIVE.sqrt(), 0.0, 0.0);
        assert_eq!(v_at_boundary.normalized(), None);

        // Boundary test: slightly above f64::MIN_POSITIVE should return Some
        let v_above_boundary = Vector3::new((f64::MIN_POSITIVE * 4.0).sqrt(), 0.0, 0.0);
        let normalized = v_above_boundary.normalized();
        assert!(normalized.is_some());
        let n = normalized.unwrap();
        assert!((n.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_angle_with_ref() {
        let angle = Vector3::X.angle_with_ref(Vector3::Y, Vector3::Z);
        assert!((angle - PI / 2.0).abs() < 1e-10);

        let angle2 = Vector3::Y.angle_with_ref(Vector3::X, Vector3::Z);
        assert!((angle2 + PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_angle_with_ref_degenerate_reference() {
        // Degenerate reference (zero vector): should return unsigned angle
        // For orthogonal vectors (dot product = 0), result should be 0
        let angle1 = Vector3::X.angle_with_ref(Vector3::Y, Vector3::ZERO);
        assert!((angle1 - 0.0).abs() < 1e-10);

        // For opposite vectors (dot product < 0), result should be π
        let angle2 = Vector3::X.angle_with_ref(-Vector3::X, Vector3::ZERO);
        assert!((angle2 - PI).abs() < 1e-10);
    }

    #[test]
    fn test_vector3_add() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vector3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector3_sub() {
        let a = Vector3::new(4.0, 5.0, 6.0);
        let b = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(a - b, Vector3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector3_neg() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(-v, Vector3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_vector3_mul_scalar() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v * 2.0, Vector3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector3_scalar_mul() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(2.0 * v, Vector3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector2_constants() {
        assert_eq!(Vector2::ZERO, Vector2::new(0.0, 0.0));
        assert_eq!(Vector2::X, Vector2::new(1.0, 0.0));
        assert_eq!(Vector2::Y, Vector2::new(0.0, 1.0));
    }

    #[test]
    fn test_vector2_new() {
        let v = Vector2::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vector2_dot() {
        assert_eq!(Vector2::X.dot(Vector2::Y), 0.0);
        assert_eq!(Vector2::X.dot(Vector2::X), 1.0);
    }

    #[test]
    fn test_vector2_cross() {
        assert_eq!(Vector2::X.cross(Vector2::Y), 1.0);
        assert_eq!(Vector2::Y.cross(Vector2::X), -1.0);
    }

    #[test]
    fn test_vector2_perp() {
        assert_eq!(Vector2::X.perp(), Vector2::Y);
        assert_eq!(Vector2::Y.perp(), Vector2::new(-1.0, 0.0));
    }

    #[test]
    fn test_vector2_magnitude() {
        assert_eq!(Vector2::new(3.0, 4.0).magnitude(), 5.0);
    }

    #[test]
    fn test_vector2_square_magnitude() {
        assert_eq!(Vector2::new(3.0, 4.0).square_magnitude(), 25.0);
    }

    #[test]
    fn test_vector2_normalized() {
        let v = Vector2::new(3.0, 4.0);
        let norm = v.normalized().unwrap();
        assert!((norm.x - 0.6).abs() < 1e-10);
        assert!((norm.y - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_normalized_zero_returns_none() {
        assert_eq!(Vector2::ZERO.normalized(), None);
    }

    #[test]
    fn test_vector2_normalized_min_positive_boundary() {
        // Boundary test: exactly at f64::MIN_POSITIVE should return None
        let v_at_boundary = Vector2::new(f64::MIN_POSITIVE.sqrt(), 0.0);
        assert_eq!(v_at_boundary.normalized(), None);

        // Boundary test: slightly above f64::MIN_POSITIVE should return Some
        let v_above_boundary = Vector2::new((f64::MIN_POSITIVE * 4.0).sqrt(), 0.0);
        let normalized = v_above_boundary.normalized();
        assert!(normalized.is_some());
        let n = normalized.unwrap();
        assert!((n.x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector2_add() {
        let a = Vector2::new(1.0, 2.0);
        let b = Vector2::new(3.0, 4.0);
        assert_eq!(a + b, Vector2::new(4.0, 6.0));
    }

    #[test]
    fn test_vector2_sub() {
        let a = Vector2::new(4.0, 6.0);
        let b = Vector2::new(1.0, 2.0);
        assert_eq!(a - b, Vector2::new(3.0, 4.0));
    }

    #[test]
    fn test_vector2_neg() {
        let v = Vector2::new(1.0, 2.0);
        assert_eq!(-v, Vector2::new(-1.0, -2.0));
    }

    #[test]
    fn test_vector2_mul_scalar() {
        let v = Vector2::new(1.0, 2.0);
        assert_eq!(v * 2.0, Vector2::new(2.0, 4.0));
    }

    #[test]
    fn test_vector2_scalar_mul() {
        let v = Vector2::new(1.0, 2.0);
        assert_eq!(2.0 * v, Vector2::new(2.0, 4.0));
    }
}
