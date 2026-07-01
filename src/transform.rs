//! Rigid and affine transformations of points and vectors.

use crate::{Axis3, Frame3, Point3, Vector3};

/// An affine transformation of 3D space: a linear map plus a translation.
///
/// Represented as a linear part `linear` (row-major 3x3 matrix, which may
/// encode rotation, scale, and/or reflection) and a translation vector
/// `translation`. A point `p` maps to `linear * p + translation`; a vector
/// `v` maps to `linear * v` (translation-invariant).
///
/// # Examples
///
/// ```
/// use geomrust::{Point3, Transform, Vector3};
/// let t = Transform::translation(Vector3::new(1.0, 2.0, 3.0));
/// assert_eq!(t.apply_point(Point3::ORIGIN), Point3::new(1.0, 2.0, 3.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    linear: [[f64; 3]; 3],
    translation: Vector3,
}

const IDENTITY_LINEAR: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn mat_vec(m: [[f64; 3]; 3], v: Vector3) -> Vector3 {
    Vector3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}

fn scalar_mul(s: f64, m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = s * m[i][j];
        }
    }
    result
}

fn mat_sub(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    result
}

fn mat_add(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

/// Outer product `d ⊗ d` of a vector with itself, as a 3x3 matrix.
fn outer(d: Vector3) -> [[f64; 3]; 3] {
    [
        [d.x * d.x, d.x * d.y, d.x * d.z],
        [d.y * d.x, d.y * d.y, d.y * d.z],
        [d.z * d.x, d.z * d.y, d.z * d.z],
    ]
}

/// Cross-product matrix `[d]×` such that `[d]× * v == d.cross(v)`.
fn cross_matrix(d: Vector3) -> [[f64; 3]; 3] {
    [[0.0, -d.z, d.y], [d.z, 0.0, -d.x], [-d.y, d.x, 0.0]]
}

/// Translation `t = P - L·P` that keeps point `p_fixed` fixed under the
/// linear part `linear`.
fn translation_fixing(linear: [[f64; 3]; 3], p_fixed: Point3) -> Vector3 {
    let p = Vector3::new(p_fixed.x, p_fixed.y, p_fixed.z);
    p - mat_vec(linear, p)
}

impl Transform {
    /// The identity transformation: leaves every point and vector unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Transform};
    /// assert_eq!(Transform::IDENTITY.apply_point(Point3::new(1.0, 2.0, 3.0)), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub const IDENTITY: Transform = Transform {
        linear: IDENTITY_LINEAR,
        translation: Vector3::ZERO,
    };

    /// A pure translation by `offset`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Transform, Vector3};
    /// let t = Transform::translation(Vector3::new(1.0, 0.0, 0.0));
    /// assert_eq!(t.apply_point(Point3::ORIGIN), Point3::new(1.0, 0.0, 0.0));
    /// ```
    pub fn translation(offset: Vector3) -> Transform {
        Transform {
            linear: IDENTITY_LINEAR,
            translation: offset,
        }
    }

    /// Rotation by `angle` radians about `axis` (right-hand rule).
    ///
    /// Uses the Rodrigues rotation formula
    /// `R = c*I + s*[d]x + (1-c)*d⊗d` for the unit axis direction `d`, then
    /// derives the translation that keeps the axis origin fixed.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Point3, Transform, Vector3};
    /// use std::f64::consts::PI;
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// let r = Transform::rotation(axis, PI / 2.0);
    /// let p = r.apply_point(Point3::new(1.0, 0.0, 0.0));
    /// assert!((p.x - 0.0).abs() < 1e-10);
    /// assert!((p.y - 1.0).abs() < 1e-10);
    /// ```
    pub fn rotation(axis: Axis3, angle: f64) -> Transform {
        let d = axis.direction();
        let c = angle.cos();
        let s = angle.sin();
        // Rodrigues' rotation formula: R = c*I + s*[d]x + (1-c)*(d⊗d).
        let linear = mat_add(
            mat_add(
                scalar_mul(c, IDENTITY_LINEAR),
                scalar_mul(s, cross_matrix(d)),
            ),
            scalar_mul(1.0 - c, outer(d)),
        );
        let translation = translation_fixing(linear, axis.origin());
        Transform {
            linear,
            translation,
        }
    }

    /// Uniform scaling by `factor` about `center` (`factor` may be negative
    /// or zero).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Transform};
    /// let s = Transform::scaling(Point3::new(1.0, 1.0, 1.0), 2.0);
    /// let p = s.apply_point(Point3::new(2.0, 2.0, 2.0));
    /// assert!((p.x - 3.0).abs() < 1e-10);
    /// ```
    pub fn scaling(center: Point3, factor: f64) -> Transform {
        let linear = scalar_mul(factor, IDENTITY_LINEAR);
        let translation = translation_fixing(linear, center);
        Transform {
            linear,
            translation,
        }
    }

    /// Point reflection (inversion) through `center`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Transform};
    /// let m = Transform::mirror_point(Point3::new(1.0, 1.0, 1.0));
    /// let p = m.apply_point(Point3::ORIGIN);
    /// assert!((p.x - 2.0).abs() < 1e-10);
    /// assert!((p.y - 2.0).abs() < 1e-10);
    /// assert!((p.z - 2.0).abs() < 1e-10);
    /// ```
    pub fn mirror_point(center: Point3) -> Transform {
        let linear = scalar_mul(-1.0, IDENTITY_LINEAR);
        let translation = 2.0 * Vector3::new(center.x, center.y, center.z);
        Transform {
            linear,
            translation,
        }
    }

    /// Mirror across `axis` (reflection that fixes the axis line).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Point3, Transform, Vector3};
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::X).unwrap();
    /// let m = Transform::mirror_axis(axis);
    /// let p = m.apply_point(Point3::new(0.0, 1.0, 0.0));
    /// assert!((p.y - (-1.0)).abs() < 1e-10);
    /// ```
    pub fn mirror_axis(axis: Axis3) -> Transform {
        let d = axis.direction();
        let linear = mat_sub(scalar_mul(2.0, outer(d)), IDENTITY_LINEAR);
        let translation = translation_fixing(linear, axis.origin());
        Transform {
            linear,
            translation,
        }
    }

    /// Mirror across the plane through `frame.origin()` with normal
    /// `frame.z_direction()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Frame3, Point3, Transform};
    /// let m = Transform::mirror_plane(Frame3::WORLD);
    /// let p = m.apply_point(Point3::new(0.0, 0.0, 5.0));
    /// assert!((p.z - (-5.0)).abs() < 1e-10);
    /// ```
    pub fn mirror_plane(frame: Frame3) -> Transform {
        let n = frame.z_direction();
        let linear = mat_sub(IDENTITY_LINEAR, scalar_mul(2.0, outer(n)));
        let translation = translation_fixing(linear, frame.origin());
        Transform {
            linear,
            translation,
        }
    }

    /// Composes `self` with `next`: `self` is applied first, then `next`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Axis3, Point3, Transform, Vector3};
    /// use std::f64::consts::PI;
    /// let translate = Transform::translation(Vector3::new(1.0, 0.0, 0.0));
    /// let axis = Axis3::new(Point3::ORIGIN, Vector3::Z).unwrap();
    /// let rotate = Transform::rotation(axis, PI / 2.0);
    /// let composed = translate.then(rotate);
    /// let p = composed.apply_point(Point3::ORIGIN);
    /// assert!((p.x - 0.0).abs() < 1e-10);
    /// assert!((p.y - 1.0).abs() < 1e-10);
    /// ```
    pub fn then(self, next: Transform) -> Transform {
        let linear = mat_mul(next.linear, self.linear);
        let translation = mat_vec(next.linear, self.translation) + next.translation;
        Transform {
            linear,
            translation,
        }
    }

    /// Applies this transformation to a point: `linear * p + translation`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Point3, Transform, Vector3};
    /// let t = Transform::translation(Vector3::new(1.0, 2.0, 3.0));
    /// assert_eq!(t.apply_point(Point3::ORIGIN), Point3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn apply_point(self, p: Point3) -> Point3 {
        let v = mat_vec(self.linear, Vector3::new(p.x, p.y, p.z)) + self.translation;
        Point3::new(v.x, v.y, v.z)
    }

    /// Applies this transformation to a vector (translation-invariant):
    /// `linear * v`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::{Transform, Vector3};
    /// let t = Transform::translation(Vector3::new(1.0, 2.0, 3.0));
    /// let v = Vector3::new(4.0, 5.0, 6.0);
    /// assert_eq!(t.apply_vector(v), v);
    /// ```
    pub fn apply_vector(self, v: Vector3) -> Vector3 {
        mat_vec(self.linear, v)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Axis3, Frame3, Point3, Transform, Vector3};
    use std::f64::consts::PI;

    #[test]
    fn test_identity_leaves_point_unchanged() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(Transform::IDENTITY.apply_point(p), p);
    }

    #[test]
    fn test_identity_leaves_vector_unchanged() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(Transform::IDENTITY.apply_vector(v), v);
    }

    #[test]
    fn test_translation_moves_point_by_offset() {
        let t = Transform::translation(Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(t.apply_point(Point3::ORIGIN), Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_translation_leaves_vector_unchanged() {
        let t = Transform::translation(Vector3::new(1.0, 2.0, 3.0));
        let v = Vector3::new(4.0, 5.0, 6.0);
        assert_eq!(t.apply_vector(v), v);
    }

    #[test]
    fn test_rotation_of_x_about_z_by_half_pi_is_y() {
        let axis = Axis3::new(Point3::ORIGIN, Vector3::Z).unwrap();
        let r = Transform::rotation(axis, PI / 2.0);
        let rotated = r.apply_point(Point3::new(1.0, 0.0, 0.0));
        assert!((rotated.x - 0.0).abs() < 1e-10, "x = {}", rotated.x);
        assert!((rotated.y - 1.0).abs() < 1e-10, "y = {}", rotated.y);
        assert!((rotated.z - 0.0).abs() < 1e-10, "z = {}", rotated.z);
    }

    #[test]
    fn test_rotation_about_off_origin_axis_fixes_axis_points() {
        let axis = Axis3::new(Point3::new(1.0, 0.0, 0.0), Vector3::Z).unwrap();
        let r = Transform::rotation(axis, PI / 2.0);
        let fixed = r.apply_point(Point3::new(1.0, 0.0, 5.0));
        assert!((fixed.x - 1.0).abs() < 1e-10);
        assert!((fixed.y - 0.0).abs() < 1e-10);
        assert!((fixed.z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_about_center() {
        let s = Transform::scaling(Point3::new(1.0, 1.0, 1.0), 2.0);
        let scaled = s.apply_point(Point3::new(2.0, 2.0, 2.0));
        assert!((scaled.x - 3.0).abs() < 1e-10);
        assert!((scaled.y - 3.0).abs() < 1e-10);
        assert!((scaled.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mirror_point_is_involution() {
        let m = Transform::mirror_point(Point3::new(1.0, 2.0, 3.0));
        let p = Point3::new(4.0, -1.0, 2.5);
        let twice = m.apply_point(m.apply_point(p));
        assert!((twice.x - p.x).abs() < 1e-10);
        assert!((twice.y - p.y).abs() < 1e-10);
        assert!((twice.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_mirror_axis_is_involution() {
        let axis = Axis3::new(Point3::new(0.0, 1.0, 0.0), Vector3::X).unwrap();
        let m = Transform::mirror_axis(axis);
        let p = Point3::new(4.0, -1.0, 2.5);
        let twice = m.apply_point(m.apply_point(p));
        assert!((twice.x - p.x).abs() < 1e-10);
        assert!((twice.y - p.y).abs() < 1e-10);
        assert!((twice.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_mirror_plane_is_involution() {
        let frame = Frame3::new(Point3::new(0.0, 0.0, 2.0), Vector3::Z, Vector3::X).unwrap();
        let m = Transform::mirror_plane(frame);
        let p = Point3::new(4.0, -1.0, 2.5);
        let twice = m.apply_point(m.apply_point(p));
        assert!((twice.x - p.x).abs() < 1e-10);
        assert!((twice.y - p.y).abs() < 1e-10);
        assert!((twice.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_then_applies_self_before_next() {
        // translate(1,0,0) then rotate about Z by pi/2 at origin: ORIGIN -> (1,0,0) -> (0,1,0)
        let translate = Transform::translation(Vector3::new(1.0, 0.0, 0.0));
        let axis = Axis3::new(Point3::ORIGIN, Vector3::Z).unwrap();
        let rotate = Transform::rotation(axis, PI / 2.0);
        let composed = translate.then(rotate);
        let result = composed.apply_point(Point3::ORIGIN);
        assert!((result.x - 0.0).abs() < 1e-10, "x = {}", result.x);
        assert!((result.y - 1.0).abs() < 1e-10, "y = {}", result.y);
        assert!((result.z - 0.0).abs() < 1e-10, "z = {}", result.z);
    }
}
