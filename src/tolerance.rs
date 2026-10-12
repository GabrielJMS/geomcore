/// Distance and angle tolerances shared by all geometric queries.
///
/// Containment checks, projections, and intersections are meaningless without
/// an agreed notion of "close enough". `Tolerance` carries the three scales
/// the kernel works with:
///
/// - [`Tolerance::confusion`]: distance below which two points coincide.
/// - [`Tolerance::angular`]: angle (radians) below which directions are
///   parallel/orthogonal.
/// - [`Tolerance::parametric`]: distance in parameter space.
///
/// The defaults match the tolerances the construction and parametrization
/// code has always used internally, so existing behavior is unchanged when
/// [`Tolerance::DEFAULT`] is passed.
///
/// # Precision system
///
/// Models are assumed to fit in a cube of side [`Tolerance::MODEL_SIZE`]
/// centred on the origin, in one length unit. Inside it an `f64`
/// coordinate is exact to about 1e-11, four orders of magnitude below the
/// default `confusion`, so computed points converge well inside it.
///
/// Every geometric decision is a distance decision against `confusion`.
/// Two directions are parallel for a configuration when turning one onto
/// the other moves the geometry by at most `confusion`: an angle below
/// `confusion / extent`, the extent being how far the geometry reaches
/// ([`Tolerance::angular_for_extent`]). Unbounded geometry, such as a
/// line or a plane, matters only inside the cube, across which it
/// reaches; that gives `angular`, by default `confusion / MODEL_SIZE`. The
/// cube bounds the model, not its carriers: without it the only
/// consistent angular tolerance would be zero. `angular` also guards dimensionless numbers,
/// such as a normalized coefficient near zero.
///
/// # Examples
///
/// ```
/// use geomcore::Tolerance;
///
/// let tol = Tolerance::DEFAULT;
/// assert_eq!(tol.confusion, 1e-7);
/// assert_eq!(tol, Tolerance::default());
///
/// let loose = Tolerance {
///     confusion: 1e-3,
///     ..Tolerance::DEFAULT
/// };
/// assert_eq!(loose.angular, tol.angular);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Distance below which two points are considered coincident.
    pub confusion: f64,
    /// Angular tolerance for parallelism/orthogonality checks (radians).
    pub angular: f64,
    /// Parametric-space tolerance.
    pub parametric: f64,
}

impl Tolerance {
    /// The side of the cube, centred on the origin, that models are
    /// assumed to fit in: 1e5 model units (100 m in millimetres).
    pub const MODEL_SIZE: f64 = 1e5;

    /// Default tolerances: `confusion = 1e-7`, `angular = 1e-12` (that is,
    /// `confusion / MODEL_SIZE`), `parametric = 1e-9`.
    pub const DEFAULT: Tolerance = Tolerance {
        confusion: 1e-7,
        angular: 1e-12,
        parametric: 1e-9,
    };

    /// Create tolerances from the three scales explicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Tolerance;
    /// let tol = Tolerance::new(1e-6, 1e-10, 1e-8);
    /// assert_eq!(tol.confusion, 1e-6);
    /// ```
    pub const fn new(confusion: f64, angular: f64, parametric: f64) -> Tolerance {
        Tolerance {
            confusion,
            angular,
            parametric,
        }
    }

    /// The angle (radians) below which two directions are parallel for
    /// geometry reaching `extent` from the pivot of the turn: turning by it
    /// moves that geometry by at most `confusion`. Never below `angular`,
    /// the angle for geometry reaching across the whole model.
    ///
    /// Compare it with the sine of the angle (the magnitude of the cross
    /// product of unit vectors). Geometry smaller than `confusion` is a
    /// point, and every direction is parallel for it.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::Tolerance;
    ///
    /// let tol = Tolerance::DEFAULT;
    /// // A circle of radius 10: normals 1e-9 apart tilt it by 1e-8.
    /// assert_eq!(tol.angular_for_extent(10.0), 1e-8);
    /// // A plane reaches across the model.
    /// assert_eq!(tol.angular_for_extent(f64::INFINITY), tol.angular);
    /// ```
    pub fn angular_for_extent(&self, extent: f64) -> f64 {
        (self.confusion / extent).max(self.angular)
    }
}

impl Default for Tolerance {
    fn default() -> Tolerance {
        Tolerance::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_matches_constants() {
        assert_eq!(Tolerance::default(), Tolerance::DEFAULT);
        assert_eq!(Tolerance::DEFAULT.confusion, 1e-7);
        assert_eq!(Tolerance::DEFAULT.angular, 1e-12);
        assert_eq!(Tolerance::DEFAULT.parametric, 1e-9);
    }

    #[test]
    fn test_default_angular_is_confusion_over_model_size() {
        let derived = Tolerance::DEFAULT.confusion / Tolerance::MODEL_SIZE;
        assert!((Tolerance::DEFAULT.angular / derived - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_angular_for_extent_is_confusion_over_extent_down_to_angular() {
        let tol = Tolerance::DEFAULT;
        assert_eq!(tol.angular_for_extent(1.0), 1e-7);
        assert_eq!(tol.angular_for_extent(1e9), tol.angular);
        assert_eq!(tol.angular_for_extent(0.0), f64::INFINITY);
    }

    #[test]
    fn test_struct_update_syntax() {
        let loose = Tolerance {
            confusion: 1e-3,
            ..Tolerance::DEFAULT
        };
        assert_eq!(loose.confusion, 1e-3);
        assert_eq!(loose.angular, Tolerance::DEFAULT.angular);
        assert_eq!(loose.parametric, Tolerance::DEFAULT.parametric);
    }
}
