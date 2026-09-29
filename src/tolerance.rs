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
    /// Default tolerances: `confusion = 1e-7`, `angular = 1e-12`,
    /// `parametric = 1e-9`.
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
