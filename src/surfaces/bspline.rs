//! B-spline (NURBS) surfaces in 3D: tensor-product de Boor evaluation with
//! first derivatives, rational (weighted) poles, and periodic (closed) surfaces
//! in either direction, a thin wrapper over [`crate::surface_math::bspline`].
//!
//! Construction packs the pole grid (and, for rational surfaces, the weight
//! grid) into a cached homogeneous flat buffer laid out row-major as
//! `[i_u][i_v][h]`, together with the flattened knot sequences for each
//! direction, so evaluation never re-derives them.

use crate::curve_math::bspline as curve;
use crate::surface_math::bspline as math;
use crate::surfaces::ParametricSurface;
use crate::{Point3, Vector3};

pub use crate::curves::BSplineConstructionError;

/// A tensor-product B-spline (optionally rational, optionally periodic in
/// either direction) surface in 3D, evaluated by de Boor's algorithm applied
/// once per parametric direction.
///
/// The pole grid `poles[i_u][i_v]` defines the control net; `i_u` indexes the
/// `u` direction and `i_v` the `v` direction. Each direction carries its own
/// degree, `knots`/`multiplicities`, and periodic flag, validated exactly as a
/// [`crate::BSplineCurve3D`] direction is. A rational surface additionally
/// carries a `weight` per pole and is evaluated in homogeneous coordinates,
/// then projected back to Euclidean space via the quotient rule.
///
/// Only first derivatives are supported (this is a proof-of-concept); see
/// [`BSplineSurface::eval_derivative`].
///
/// # Examples
///
/// A bilinear (degree-1 × degree-1) patch over the unit square:
///
/// ```
/// use geomrust::surfaces::BSplineSurface;
/// use geomrust::Point3;
///
/// let poles = vec![
///     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
///     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
/// ];
/// let surface = BSplineSurface::new(
///     1,
///     1,
///     poles,
///     vec![0.0, 1.0],
///     vec![2, 2],
///     vec![0.0, 1.0],
///     vec![2, 2],
///     false,
///     false,
/// )
/// .unwrap();
///
/// assert_eq!(surface.eval_point(0.5, 0.5), Point3::new(0.5, 0.5, 0.25));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineSurface {
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    poles: Vec<Vec<Point3>>,
    weights: Option<Vec<Vec<f64>>>,
    u_knots: Vec<f64>,
    u_mults: Vec<u32>,
    v_knots: Vec<f64>,
    v_mults: Vec<u32>,
    /// Cached flat (expanded) knot sequence for each direction.
    u_flat: Vec<f64>,
    v_flat: Vec<f64>,
    /// Cached flat pole buffer, row-major `[i_u][i_v][h]`, with `h = 4`
    /// (homogeneous `(x*w, y*w, z*w, w)`) if rational, else `h = 3`.
    flat_poles: Vec<f64>,
    /// Grid dimensions: `n_u` rows (u direction), `n_v` columns (v direction).
    n_u: usize,
    n_v: usize,
    /// Per-pole coordinate stride (`4` rational, `3` otherwise).
    h: usize,
}

impl BSplineSurface {
    /// Creates a non-rational (polynomial) tensor-product B-spline surface.
    ///
    /// `poles[i_u][i_v]` is the control net: `i_u` indexes the `u` direction
    /// (`poles.len()` rows) and `i_v` the `v` direction (row length).
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConstructionError::PoleCountMismatch`] if the pole grid
    /// is empty or ragged (rows of differing length), since a well-formed
    /// tensor-product net must be rectangular. Otherwise returns a
    /// [`BSplineConstructionError`] if either direction's `degree`, `knots`,
    /// `multiplicities`, and pole count (grid rows for `u`, columns for `v`)
    /// are not mutually consistent (see the error variants for the exact
    /// conditions checked, applied per direction).
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::surfaces::BSplineSurface;
    /// use geomrust::Point3;
    ///
    /// let poles = vec![
    ///     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    ///     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles,
    ///     vec![0.0, 1.0], vec![2, 2],
    ///     vec![0.0, 1.0], vec![2, 2],
    ///     false, false,
    /// )
    /// .unwrap();
    /// assert_eq!(surface.eval_point(0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
    /// ```
    #[allow(clippy::too_many_arguments)] // mirrors the two-direction knot/degree data model
    pub fn new(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<Point3>>,
        u_knots: Vec<f64>,
        u_multiplicities: Vec<u32>,
        v_knots: Vec<f64>,
        v_multiplicities: Vec<u32>,
        u_periodic: bool,
        v_periodic: bool,
    ) -> Result<BSplineSurface, BSplineConstructionError> {
        Self::build(
            u_degree,
            v_degree,
            poles,
            None,
            u_knots,
            u_multiplicities,
            v_knots,
            v_multiplicities,
            u_periodic,
            v_periodic,
        )
    }

    /// Creates a rational (NURBS) tensor-product B-spline surface with one
    /// weight per pole, laid out `weights[i_u][i_v]` matching `poles`.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConstructionError::WeightCountMismatch`] if the weight
    /// grid does not match the pole grid shape, or
    /// [`BSplineConstructionError::NonPositiveWeight`] if any weight is `<= 0`.
    /// Otherwise validates the pole grid and both directions exactly as
    /// [`BSplineSurface::new`] does.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomrust::surfaces::BSplineSurface;
    /// use geomrust::Point3;
    ///
    /// let poles = vec![
    ///     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    ///     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// ];
    /// let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    /// let surface = BSplineSurface::new_rational(
    ///     1, 1, poles, weights,
    ///     vec![0.0, 1.0], vec![2, 2],
    ///     vec![0.0, 1.0], vec![2, 2],
    ///     false, false,
    /// )
    /// .unwrap();
    /// // All-unit weights reproduce the polynomial patch.
    /// assert_eq!(surface.eval_point(0.5, 0.5), Point3::new(0.5, 0.5, 0.25));
    /// ```
    #[allow(clippy::too_many_arguments)] // mirrors the two-direction knot/degree data model
    pub fn new_rational(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<Point3>>,
        weights: Vec<Vec<f64>>,
        u_knots: Vec<f64>,
        u_multiplicities: Vec<u32>,
        v_knots: Vec<f64>,
        v_multiplicities: Vec<u32>,
        u_periodic: bool,
        v_periodic: bool,
    ) -> Result<BSplineSurface, BSplineConstructionError> {
        Self::build(
            u_degree,
            v_degree,
            poles,
            Some(weights),
            u_knots,
            u_multiplicities,
            v_knots,
            v_multiplicities,
            u_periodic,
            v_periodic,
        )
    }

    #[allow(clippy::too_many_arguments)] // internal shared builder
    fn build(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<Point3>>,
        weights: Option<Vec<Vec<f64>>>,
        u_knots: Vec<f64>,
        u_multiplicities: Vec<u32>,
        v_knots: Vec<f64>,
        v_multiplicities: Vec<u32>,
        u_periodic: bool,
        v_periodic: bool,
    ) -> Result<BSplineSurface, BSplineConstructionError> {
        // Rectangular, non-empty grid. A ragged or empty net is a shape error,
        // reported as PoleCountMismatch.
        let n_u = poles.len();
        if n_u == 0 || poles[0].is_empty() {
            return Err(BSplineConstructionError::PoleCountMismatch);
        }
        let n_v = poles[0].len();
        if poles.iter().any(|row| row.len() != n_v) {
            return Err(BSplineConstructionError::PoleCountMismatch);
        }

        // Weight grid, if present, must match the pole grid shape, and be
        // strictly positive.
        if let Some(ws) = &weights {
            if ws.len() != n_u || ws.iter().any(|row| row.len() != n_v) {
                return Err(BSplineConstructionError::WeightCountMismatch);
            }
            if ws.iter().flatten().any(|&w| w <= 0.0) {
                return Err(BSplineConstructionError::NonPositiveWeight);
            }
        }

        // Validate each direction: u over grid rows, v over grid columns.
        curve::validate_direction(u_degree, n_u, &u_knots, &u_multiplicities, u_periodic)?;
        curve::validate_direction(v_degree, n_v, &v_knots, &v_multiplicities, v_periodic)?;

        let u_flat = curve::flat_knots(&u_knots, &u_multiplicities);
        let v_flat = curve::flat_knots(&v_knots, &v_multiplicities);
        let h = if weights.is_some() { 4 } else { 3 };
        let flat_poles = pack_flat_poles(&poles, weights.as_ref(), h);

        Ok(BSplineSurface {
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            poles,
            weights,
            u_knots,
            u_mults: u_multiplicities,
            v_knots,
            v_mults: v_multiplicities,
            u_flat,
            v_flat,
            flat_poles,
            n_u,
            n_v,
            h,
        })
    }

    /// Returns the surface's degree in the `u` direction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert_eq!(surface.u_degree(), 1);
    /// ```
    pub fn u_degree(&self) -> usize {
        self.u_degree
    }

    /// Returns the surface's degree in the `v` direction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert_eq!(surface.v_degree(), 1);
    /// ```
    pub fn v_degree(&self) -> usize {
        self.v_degree
    }

    /// Returns whether the surface is periodic (closed) in the `u` direction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert!(!surface.is_u_periodic());
    /// ```
    pub fn is_u_periodic(&self) -> bool {
        self.u_periodic
    }

    /// Returns whether the surface is periodic (closed) in the `v` direction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert!(!surface.is_v_periodic());
    /// ```
    pub fn is_v_periodic(&self) -> bool {
        self.v_periodic
    }

    /// Returns whether the surface is rational (has per-pole weights).
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert!(!surface.is_rational());
    /// ```
    pub fn is_rational(&self) -> bool {
        self.weights.is_some()
    }

    /// Evaluates the point on the surface at parameters `(u, v)`.
    ///
    /// For periodic directions, a parameter outside the bounds is wrapped into
    /// the period before evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert_eq!(surface.eval_point(0.5, 0.5), Point3::new(0.5, 0.5, 0.25));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3 {
        math::surface_d0(
            u,
            v,
            self.u_degree,
            self.v_degree,
            self.u_periodic,
            self.v_periodic,
            &self.u_flat,
            &self.v_flat,
            &self.flat_poles,
            self.n_u,
            self.n_v,
            self.h,
        )
    }

    /// Evaluates the points on the surface at each `(u, v)` in `uvs`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::Point3;
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// let points = surface.eval_points(&[(0.0, 0.0), (0.5, 0.5)]);
    /// assert_eq!(points[1], Point3::new(0.5, 0.5, 0.25));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the first partial derivative of order `(du, dv)` at `(u, v)`.
    ///
    /// Exactly one of `du`, `dv` must be `1` and the other `0`: `(1, 0)`
    /// returns `∂S/∂u`, `(0, 1)` returns `∂S/∂v`.
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`BSplineSurface::eval_point`] for the
    /// position itself), or if `du + dv >= 2`, with a message stating that
    /// only first derivatives are supported (this is a proof-of-concept).
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomrust::surfaces::BSplineSurface;
    /// # use geomrust::{Point3, Vector3};
    /// # let poles = vec![
    /// #     vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// // Bilinear patch: Su = (1, 0, v), Sv = (0, 1, u).
    /// assert_eq!(surface.eval_derivative(0.3, 0.7, 1, 0), Vector3::new(1.0, 0.0, 0.7));
    /// assert_eq!(surface.eval_derivative(0.3, 0.7, 0, 1), Vector3::new(0.0, 1.0, 0.3));
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
        match (du, dv) {
            (0, 0) => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            (1, 0) | (0, 1) => {
                let (_s, su, sv) = self.eval_surface_d1(u, v);
                if du == 1 { su } else { sv }
            }
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (only first derivatives are supported)"
            ),
        }
    }

    /// Shared first-derivative evaluation, returning `(S, Su, Sv)`.
    fn eval_surface_d1(&self, u: f64, v: f64) -> (Point3, Vector3, Vector3) {
        math::surface_d1(
            u,
            v,
            self.u_degree,
            self.v_degree,
            self.u_periodic,
            self.v_periodic,
            &self.u_flat,
            &self.v_flat,
            &self.flat_poles,
            self.n_u,
            self.n_v,
            self.h,
        )
    }

    /// `u` parameter bounds `(first, last)`: for a clamped direction the active
    /// flat span, for a periodic direction the full period `(knots[0],
    /// knots[last])`.
    fn u_param_range(&self) -> (f64, f64) {
        let sum_mults: u32 = self.u_mults.iter().sum();
        curve::param_range(
            &self.u_flat,
            self.u_degree,
            self.u_periodic,
            sum_mults as usize,
        )
    }

    /// `v` parameter bounds `(first, last)`, analogous to [`Self::u_param_range`].
    fn v_param_range(&self) -> (f64, f64) {
        let sum_mults: u32 = self.v_mults.iter().sum();
        curve::param_range(
            &self.v_flat,
            self.v_degree,
            self.v_periodic,
            sum_mults as usize,
        )
    }
}

impl ParametricSurface for BSplineSurface {
    fn eval_point(&self, u: f64, v: f64) -> Point3 {
        BSplineSurface::eval_point(self, u, v)
    }

    /// See [`BSplineSurface::eval_derivative`] for the supported orders and the
    /// panic conditions (only first derivatives are supported).
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3 {
        BSplineSurface::eval_derivative(self, u, v, du, dv)
    }

    fn u_bounds(&self) -> (f64, f64) {
        self.u_param_range()
    }

    fn v_bounds(&self) -> (f64, f64) {
        self.v_param_range()
    }

    fn u_period(&self) -> Option<f64> {
        if self.u_periodic {
            let (first, last) = self.u_param_range();
            Some(last - first)
        } else {
            None
        }
    }

    fn v_period(&self) -> Option<f64> {
        if self.v_periodic {
            let (first, last) = self.v_param_range();
            Some(last - first)
        } else {
            None
        }
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3> {
        BSplineSurface::eval_points(self, uvs)
    }
}

/// Packs the pole grid (and, if rational, weights) into a flat, row-major
/// `[i_u][i_v][h]` coordinate buffer: `(x, y, z)` per pole if `weights` is
/// `None`, else the homogeneous `(x*w, y*w, z*w, w)`.
fn pack_flat_poles(poles: &[Vec<Point3>], weights: Option<&Vec<Vec<f64>>>, h: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(poles.len() * poles[0].len() * h);
    for (iu, row) in poles.iter().enumerate() {
        for (iv, p) in row.iter().enumerate() {
            match weights {
                Some(ws) => {
                    let w = ws[iu][iv];
                    out.extend_from_slice(&[p.x * w, p.y * w, p.z * w, w]);
                }
                None => out.extend_from_slice(&[p.x, p.y, p.z]),
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- bilinear patch by hand ----

    fn bilinear() -> BSplineSurface {
        // Degree 1x1 over the unit square:
        //   P00=(0,0,0) P01=(0,1,0)
        //   P10=(1,0,0) P11=(1,1,1)
        // S(u,v) = (u, v, u*v).
        let poles = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
        ];
        BSplineSurface::new(
            1,
            1,
            poles,
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_accessors() {
        let s = bilinear();
        assert_eq!(s.u_degree(), 1);
        assert_eq!(s.v_degree(), 1);
        assert!(!s.is_u_periodic());
        assert!(!s.is_v_periodic());
        assert!(!s.is_rational());
    }

    #[test]
    fn test_bilinear_point_by_hand() {
        let s = bilinear();
        assert_eq!(s.eval_point(0.5, 0.5), Point3::new(0.5, 0.5, 0.25));
        assert_eq!(s.eval_point(0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        assert_eq!(s.eval_point(1.0, 1.0), Point3::new(1.0, 1.0, 1.0));
        assert_eq!(s.eval_point(0.3, 0.7), Point3::new(0.3, 0.7, 0.3 * 0.7));
    }

    #[test]
    fn test_bilinear_derivatives_by_hand() {
        let s = bilinear();
        // Su = (1, 0, v), Sv = (0, 1, u).
        assert_eq!(
            s.eval_derivative(0.3, 0.7, 1, 0),
            Vector3::new(1.0, 0.0, 0.7)
        );
        assert_eq!(
            s.eval_derivative(0.3, 0.7, 0, 1),
            Vector3::new(0.0, 1.0, 0.3)
        );
    }

    #[test]
    fn test_eval_points_matches_mapped_eval_point() {
        let s = bilinear();
        let uvs = [(0.0, 0.0), (0.2, 0.4), (0.5, 0.5), (1.0, 1.0)];
        let expected: Vec<Point3> = uvs.iter().map(|&(u, v)| s.eval_point(u, v)).collect();
        assert_eq!(s.eval_points(&uvs), expected);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        bilinear().eval_derivative(0.5, 0.5, 0, 0);
    }

    #[test]
    #[should_panic(expected = "only first derivatives are supported")]
    fn test_eval_derivative_second_order_uu_panics() {
        bilinear().eval_derivative(0.5, 0.5, 2, 0);
    }

    #[test]
    #[should_panic(expected = "only first derivatives are supported")]
    fn test_eval_derivative_mixed_second_order_panics() {
        bilinear().eval_derivative(0.5, 0.5, 1, 1);
    }

    // ---- bounds / periods ----

    #[test]
    fn test_bounds_clamped() {
        let s = bilinear();
        assert_eq!(ParametricSurface::u_bounds(&s), (0.0, 1.0));
        assert_eq!(ParametricSurface::v_bounds(&s), (0.0, 1.0));
        assert_eq!(ParametricSurface::u_period(&s), None);
        assert_eq!(ParametricSurface::v_period(&s), None);
    }

    // ---- validation error variants ----

    fn ok_grid() -> Vec<Vec<Point3>> {
        vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
        ]
    }

    #[test]
    fn test_empty_grid_is_pole_count_mismatch() {
        let err = BSplineSurface::new(
            1,
            1,
            vec![],
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::PoleCountMismatch);
    }

    #[test]
    fn test_ragged_grid_is_pole_count_mismatch() {
        let poles = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0)], // short row
        ];
        let err = BSplineSurface::new(
            1,
            1,
            poles,
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::PoleCountMismatch);
    }

    #[test]
    fn test_invalid_u_degree() {
        let err = BSplineSurface::new(
            0,
            1,
            ok_grid(),
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::InvalidDegree);
    }

    #[test]
    fn test_u_pole_count_mismatch_from_knots() {
        // Degree 1: expected u rows = sum(u_mults) - degree - 1. Knots
        // [0,1,2] mults [2,1,2] (valid for degree 1) expect 3 rows, but the
        // grid has only 2 => pole-count mismatch in the u direction.
        let err = BSplineSurface::new(
            1,
            1,
            ok_grid(),
            vec![0.0, 1.0, 2.0],
            vec![2, 1, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::PoleCountMismatch);
    }

    #[test]
    fn test_v_knots_not_increasing() {
        let err = BSplineSurface::new(
            1,
            1,
            ok_grid(),
            vec![0.0, 1.0],
            vec![2, 2],
            vec![1.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::KnotsNotIncreasing);
    }

    #[test]
    fn test_weight_count_mismatch_ragged() {
        let err = BSplineSurface::new_rational(
            1,
            1,
            ok_grid(),
            vec![vec![1.0, 1.0], vec![1.0]], // ragged weights
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::WeightCountMismatch);
    }

    #[test]
    fn test_weight_count_mismatch_wrong_rows() {
        let err = BSplineSurface::new_rational(
            1,
            1,
            ok_grid(),
            vec![vec![1.0, 1.0]], // one row instead of two
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::WeightCountMismatch);
    }

    #[test]
    fn test_non_positive_weight() {
        let err = BSplineSurface::new_rational(
            1,
            1,
            ok_grid(),
            vec![vec![1.0, 0.0], vec![1.0, 1.0]],
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::NonPositiveWeight);

        let err = BSplineSurface::new_rational(
            1,
            1,
            ok_grid(),
            vec![vec![1.0, 1.0], vec![-2.0, 1.0]],
            vec![0.0, 1.0],
            vec![2, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, BSplineConstructionError::NonPositiveWeight);
    }

    // ---- periodic seam through the public type ----

    fn periodic_tube() -> BSplineSurface {
        // Fixture uperiodic_tube: u_degree 2 periodic (6 pole rows), v_degree 1
        // clamped (2 columns).
        let poles = vec![
            vec![Point3::new(2.0, 0.0, 0.0), Point3::new(2.2, 0.0, 3.0)],
            vec![
                Point3::new(1.0000000000000002, 1.7320508075688772, 0.0),
                Point3::new(1.1000000000000003, 1.905255888325765, 3.0),
            ],
            vec![
                Point3::new(-0.9999999999999996, 1.7320508075688774, 0.0),
                Point3::new(-1.0999999999999996, 1.9052558883257653, 3.0),
            ],
            vec![
                Point3::new(-2.0, 2.4492935982947064e-16, 0.0),
                Point3::new(-2.2, 2.6942229581241775e-16, 3.0),
            ],
            vec![
                Point3::new(-1.0000000000000009, -1.7320508075688767, 0.0),
                Point3::new(-1.100000000000001, -1.9052558883257646, 3.0),
            ],
            vec![
                Point3::new(1.0000000000000002, -1.7320508075688772, 0.0),
                Point3::new(1.1000000000000003, -1.905255888325765, 3.0),
            ],
        ];
        BSplineSurface::new(
            2,
            1,
            poles,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 1, 1, 1, 1, 1],
            vec![0.0, 1.0],
            vec![2, 2],
            true,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_periodic_bounds_and_period() {
        let s = periodic_tube();
        assert_eq!(ParametricSurface::u_bounds(&s), (0.0, 6.0));
        assert_eq!(ParametricSurface::v_bounds(&s), (0.0, 1.0));
        assert_eq!(ParametricSurface::u_period(&s), Some(6.0));
        assert_eq!(ParametricSurface::v_period(&s), None);
    }

    #[test]
    fn test_periodic_seam_matches_across_period() {
        let s = periodic_tube();
        // u = 0 and u = period (6) must give the same point and u-derivative.
        for v in [0.0, 0.3, 0.7, 1.0] {
            let p0 = s.eval_point(0.0, v);
            let p1 = s.eval_point(6.0, v);
            assert!((p0.x - p1.x).abs() < 1e-9, "x seam v={v}");
            assert!((p0.y - p1.y).abs() < 1e-9, "y seam v={v}");
            assert!((p0.z - p1.z).abs() < 1e-9, "z seam v={v}");

            let d0 = s.eval_derivative(0.0, v, 1, 0);
            let d1 = s.eval_derivative(6.0, v, 1, 0);
            assert!((d0.x - d1.x).abs() < 1e-9, "du.x seam v={v}");
            assert!((d0.y - d1.y).abs() < 1e-9, "du.y seam v={v}");
            assert!((d0.z - d1.z).abs() < 1e-9, "du.z seam v={v}");
        }
    }

    #[test]
    fn test_periodic_out_of_window_u_matches_golden() {
        let s = periodic_tube();
        // Golden from fixtures/surfaces_bspline.json (uperiodic_tube), sample
        // with out-of-window u = -0.8.
        let p = s.eval_point(-0.8, 0.6);
        assert!((p.x - 1.7596000000000003).abs() < 1e-7);
        assert!((p.y - -0.5507921568069025).abs() < 1e-7);
        assert!((p.z - 1.7999999999999998).abs() < 1e-7);
    }
}
