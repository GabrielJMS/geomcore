//! B-spline (NURBS) surfaces in 3D: tensor-product de Boor evaluation with
//! first derivatives, rational (weighted) poles, and periodic (closed) surfaces
//! in either direction, a thin wrapper over [`crate::surface_math::bspline`].
//!
//! Construction packs the pole grid (and, for rational surfaces, the weight
//! grid) into a cached homogeneous flat buffer laid out row-major as
//! `[i_u][i_v][h]`, together with the flattened knot sequences for each
//! direction, so evaluation never re-derives them.

use crate::curve_math::bspline as curve;
use crate::math::boxtree::{BoxTree, Lazy};
use crate::math::newton_min_2d;
use crate::projection::SurfaceProjection;
use crate::surface_math::bspline as math;
use crate::surfaces::ParametricSurface;
use crate::{Point3D, Tolerance, Transform, Vector3D};

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
/// Derivatives up to second order are supported; see
/// [`BSplineSurface::eval_derivative`].
///
/// # Examples
///
/// A bilinear (degree-1 × degree-1) patch over the unit square:
///
/// ```
/// use geomcore::surfaces::BSplineSurface;
/// use geomcore::Point3D;
///
/// let poles = vec![
///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
///     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
/// assert_eq!(surface.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 0.25));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BSplineSurface {
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    poles: Vec<Vec<Point3D>>,
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
    /// Bounding-box hierarchy of the knot-span patches, built on the first
    /// projection.
    patches: Lazy<PatchTree>,
}

/// The knot-span boundaries of each direction and the box hierarchy over
/// the patches (`i_u * (v_breaks.len() - 1) + i_v`).
#[derive(Clone)]
struct PatchTree {
    u_breaks: Vec<f64>,
    v_breaks: Vec<f64>,
    tree: BoxTree,
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
    /// use geomcore::surfaces::BSplineSurface;
    /// use geomcore::Point3D;
    ///
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    ///     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles,
    ///     vec![0.0, 1.0], vec![2, 2],
    ///     vec![0.0, 1.0], vec![2, 2],
    ///     false, false,
    /// )
    /// .unwrap();
    /// assert_eq!(surface.eval_point(0.0, 0.0), Point3D::new(0.0, 0.0, 0.0));
    /// ```
    #[allow(clippy::too_many_arguments)] // mirrors the two-direction knot/degree data model
    pub fn new(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<Point3D>>,
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
    /// use geomcore::surfaces::BSplineSurface;
    /// use geomcore::Point3D;
    ///
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    ///     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
    /// assert_eq!(surface.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 0.25));
    /// ```
    #[allow(clippy::too_many_arguments)] // mirrors the two-direction knot/degree data model
    pub fn new_rational(
        u_degree: usize,
        v_degree: usize,
        poles: Vec<Vec<Point3D>>,
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
        poles: Vec<Vec<Point3D>>,
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
            patches: Lazy::default(),
        })
    }

    /// Returns the surface's degree in the `u` direction.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert!(!surface.is_rational());
    /// ```
    pub fn is_rational(&self) -> bool {
        self.weights.is_some()
    }

    /// Returns the image of this surface under `transform`: every control
    /// pole maps point-wise while degrees, knots, multiplicities,
    /// periodicity, and weights are unchanged.
    ///
    /// B-spline evaluation is affinely invariant, so mapping the poles is
    /// exact for every rigid, mirrored, or uniformly scaled transform.
    /// Infallible: pole images always form a valid net over the unchanged
    /// knot vectors, even for a collapsing (zero-scale) transform.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::{Point3D, Transform, Vector3D};
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// let moved = surface.transformed(Transform::translation(Vector3D::new(0.0, 0.0, 4.0)));
    /// assert_eq!(moved.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 4.25));
    /// ```
    pub fn transformed(&self, transform: Transform) -> BSplineSurface {
        let poles: Vec<Vec<Point3D>> = self
            .poles
            .iter()
            .map(|row| row.iter().map(|&p| transform.apply_point(p)).collect())
            .collect();
        let flat_poles = pack_flat_poles(&poles, self.weights.as_ref(), self.h);
        BSplineSurface {
            u_degree: self.u_degree,
            v_degree: self.v_degree,
            u_periodic: self.u_periodic,
            v_periodic: self.v_periodic,
            poles,
            weights: self.weights.clone(),
            u_knots: self.u_knots.clone(),
            u_mults: self.u_mults.clone(),
            v_knots: self.v_knots.clone(),
            v_mults: self.v_mults.clone(),
            u_flat: self.u_flat.clone(),
            v_flat: self.v_flat.clone(),
            flat_poles,
            n_u: self.n_u,
            n_v: self.n_v,
            h: self.h,
            patches: Lazy::default(),
        }
    }

    /// Evaluates the point on the surface at parameters `(u, v)`.
    ///
    /// For periodic directions, a parameter outside the bounds is wrapped into
    /// the period before evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// assert_eq!(surface.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 0.25));
    /// ```
    pub fn eval_point(&self, u: f64, v: f64) -> Point3D {
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
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::Point3D;
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// let points = surface.eval_points(&[(0.0, 0.0), (0.5, 0.5)]);
    /// assert_eq!(points[1], Point3D::new(0.5, 0.5, 0.25));
    /// ```
    pub fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        uvs.iter().map(|&(u, v)| self.eval_point(u, v)).collect()
    }

    /// Evaluates the partial derivative of order `(du, dv)` at `(u, v)`,
    /// for `1 <= du + dv <= 2`: `(1, 0)` is `∂S/∂u`, `(1, 1)` is
    /// `∂²S/∂u∂v`, and so on.
    ///
    /// # Panics
    ///
    /// Panics if `du + dv == 0` (use [`BSplineSurface::eval_point`] for the
    /// position itself), or if `du + dv > 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use geomcore::surfaces::BSplineSurface;
    /// # use geomcore::{Point3D, Vector3D};
    /// # let poles = vec![
    /// #     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
    /// #     vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
    /// # ];
    /// # let surface = BSplineSurface::new(1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false).unwrap();
    /// // Bilinear patch: Su = (1, 0, v), Sv = (0, 1, u).
    /// assert_eq!(surface.eval_derivative(0.3, 0.7, 1, 0), Vector3D::new(1.0, 0.0, 0.7));
    /// assert_eq!(surface.eval_derivative(0.3, 0.7, 0, 1), Vector3D::new(0.0, 1.0, 0.3));
    /// ```
    pub fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
        match (du, dv) {
            (0, 0) => panic!(
                "eval_derivative: du + dv must be >= 1 (use eval_point for the (0, 0) order)"
            ),
            (1, 0) | (0, 1) => {
                let (_s, su, sv) = self.eval_surface_d1(u, v);
                if du == 1 { su } else { sv }
            }
            (2, 0) | (1, 1) | (0, 2) => {
                let [_, _, _, suu, suv, svv] = self.eval_surface_d2(u, v);
                match du {
                    2 => suu,
                    1 => suv,
                    _ => svv,
                }
            }
            _ => panic!(
                "eval_derivative: order du={du}, dv={dv} is not supported (du + dv must be <= 2)"
            ),
        }
    }

    /// Returns whether `point` lies on the surface: the surface is
    /// projected with [`BSplineSurface::project_point`] and the point
    /// counts as contained when the projected distance is within
    /// `tol.confusion`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, Tolerance};
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
    ///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false,
    /// )
    /// .unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// assert!(surface.contains(Point3D::new(1.0, 1.0, 0.0), tol));
    /// assert!(!surface.contains(Point3D::new(1.0, 1.0, 1.0), tol));
    /// ```
    pub fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        self.project_point(point, tol).distance <= tol.confusion
    }

    /// All local minima of the distance from `point` to the surface,
    /// ordered by ascending distance.
    ///
    /// Seeding follows the knot structure: each direction is sampled at least `max(degree, 2)` times per
    /// knot span (16 samples minimum), boundaries included, and the
    /// discrete local minima of the sampled distance (wrapping across
    /// periodic seams) seed a box-bounded Newton minimisation of
    /// `|S(u,v) - P|^2` with exact second derivatives, within the C1
    /// region around each seed (C0 crease knots bound it). Minima on an
    /// edge, corner or crease are reached exactly. Seeds converging to
    /// the same point merge; the first entry is the global closest point
    /// ([`BSplineSurface::project_point`] is always a candidate).
    /// Periodic parameters are reported within their period.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, Tolerance};
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
    ///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false,
    /// )
    /// .unwrap();
    /// let extrema = surface.extrema(Point3D::new(1.0, 1.0, 1.0), Tolerance::DEFAULT);
    /// assert!(!extrema.is_empty());
    /// assert_eq!(extrema[0].distance, 1.0);
    /// ```
    pub fn extrema(&self, point: Point3D, tol: Tolerance) -> Vec<SurfaceProjection> {
        use crate::projection::snap_distance;
        // Refined seeds beyond this many (by sampled distance) are dropped.
        const MAX_SEEDS: usize = 32;
        let (u0, u1) = self.u_param_range();
        let (v0, v1) = self.v_param_range();
        let ub = curve::span_breaks(&self.u_knots, u0, u1);
        let vb = curve::span_breaks(&self.v_knots, v0, v1);
        let us = curve::span_samples(
            &ub,
            samples_per_span(self.u_degree, ub.len() - 1),
            self.u_periodic,
        );
        let vs = curve::span_samples(
            &vb,
            samples_per_span(self.v_degree, vb.len() - 1),
            self.v_periodic,
        );
        let (nu, nv) = (us.len(), vs.len());
        let dist: Vec<f64> = math::surface_grid(
            &us,
            &vs,
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
        .into_iter()
        .map(|p| p.distance(point))
        .collect();
        let mut seeds: Vec<(f64, usize, usize)> =
            grid_minima(&dist, nu, nv, [self.u_periodic, self.v_periodic])
                .map(|(i, j)| (dist[i * nv + j], i, j))
                .collect();
        seeds.sort_by(|a, b| a.0.total_cmp(&b.0));
        seeds.truncate(MAX_SEEDS);
        let u_regions = curve::smooth_regions(
            &self.u_knots,
            &self.u_mults,
            self.u_degree,
            self.u_periodic,
            u0,
            u1,
        );
        let v_regions = curve::smooth_regions(
            &self.v_knots,
            &self.v_mults,
            self.v_degree,
            self.v_periodic,
            v0,
            v1,
        );
        let (u_period, v_period) = (
            self.u_periodic.then_some(u1 - u0),
            self.v_periodic.then_some(v1 - v0),
        );
        let global = self.project_point(point, tol);
        let mut candidates = vec![(global.u, global.v)];
        for (_, i, j) in seeds {
            for (ubounds, u) in curve::regions_at(&u_regions, us[i], u_period) {
                for (vbounds, v) in curve::regions_at(&v_regions, vs[j], v_period) {
                    candidates.push(self.refine_projection(point, u, v, [ubounds, vbounds]));
                }
            }
        }
        let mut out: Vec<SurfaceProjection> = Vec::new();
        let mut kept: Vec<Point3D> = Vec::new();
        for (u, v) in candidates {
            let p = self.eval_point(u, v);
            if kept.iter().any(|q| q.distance(p) <= tol.confusion) {
                continue;
            }
            kept.push(p);
            out.push(SurfaceProjection {
                u,
                v,
                distance: snap_distance(p.distance(point), tol.confusion),
            });
        }
        out.sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Projects `point` onto the surface: the global minimum of the
    /// distance, as the first entry of [`BSplineSurface::extrema`] but
    /// without visiting the whole surface. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`BSplineSurface::contains`].
    ///
    /// Branch and bound over the knot-span patches: each patch lies in the
    /// bounding box of its control poles (convex hull property), so
    /// patches are visited by increasing box distance, each seeded from
    /// its best sample (same density as [`BSplineSurface::extrema`]) and
    /// refined by the same Newton minimisation bounded to the patch (a
    /// polynomial piece, so minima on its creases and edges are reached
    /// exactly), until no remaining
    /// box can hold a closer point. The boxes form a hierarchy built once
    /// per surface (on the first projection), so the cost tracks the
    /// patches near `point`, not the size of the control net.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, Tolerance};
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
    ///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false,
    /// )
    /// .unwrap();
    /// let proj = surface.project_point(Point3D::new(1.0, 1.0, 1.0), Tolerance::DEFAULT);
    /// assert!((proj.u - 0.5).abs() < 1e-9);
    /// assert!((proj.v - 0.5).abs() < 1e-9);
    /// assert_eq!(proj.distance, 1.0);
    /// ```
    pub fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        use crate::projection::snap_distance;
        let patches = self.patch_tree();
        let (ub, vb) = (&patches.u_breaks, &patches.v_breaks);
        let (per_u, per_v) = (
            samples_per_span(self.u_degree, ub.len() - 1),
            samples_per_span(self.v_degree, vb.len() - 1),
        );
        let n_vspans = vb.len() - 1;
        // (distance, u, v) of the closest point so far.
        let mut best = (f64::INFINITY, ub[0], vb[0]);
        patches.tree.nearest_first(point, |k| {
            let (a, b) = (k / n_vspans, k % n_vspans);
            let us = curve::span_samples(&ub[a..a + 2], per_u, false);
            let vs = curve::span_samples(&vb[b..b + 2], per_v, false);
            let samples = math::surface_grid(
                &us,
                &vs,
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
            );
            let dist: Vec<f64> = samples.iter().map(|p| p.distance(point)).collect();
            // A patch can hold several basins: refine from each discrete
            // local minimum of its samples.
            for (i, j) in grid_minima(&dist, us.len(), vs.len(), [false, false]) {
                let (u, v) = self.refine_projection(
                    point,
                    us[i],
                    vs[j],
                    [Some((ub[a], ub[a + 1])), Some((vb[b], vb[b + 1]))],
                );
                let d = self.eval_point(u, v).distance(point);
                if d < best.0 {
                    best = (d, u, v);
                }
            }
            best.0
        });
        SurfaceProjection {
            u: best.1,
            v: best.2,
            distance: snap_distance(best.0, tol.confusion),
        }
    }

    /// Projects each point in `points` onto the surface.
    ///
    /// Default-style batch wrapper over [`BSplineSurface::project_point`]:
    /// one native call per batch, mirroring [`BSplineSurface::eval_points`].
    pub fn project_points(&self, points: &[Point3D], tol: Tolerance) -> Vec<SurfaceProjection> {
        points.iter().map(|&p| self.project_point(p, tol)).collect()
    }

    /// The patch bounding-box hierarchy, built on first use.
    fn patch_tree(&self) -> &PatchTree {
        self.patches.get_or_init(|| {
            let (u0, u1) = self.u_param_range();
            let (v0, v1) = self.v_param_range();
            let u_breaks = curve::span_breaks(&self.u_knots, u0, u1);
            let v_breaks = curve::span_breaks(&self.v_knots, v0, v1);
            let boxes = math::patch_boxes(
                &u_breaks,
                &v_breaks,
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
            );
            PatchTree {
                tree: BoxTree::new(&boxes, v_breaks.len() - 1),
                u_breaks,
                v_breaks,
            }
        })
    }

    /// Shared first-derivative evaluation, returning `(S, Su, Sv)`.
    fn eval_surface_d1(&self, u: f64, v: f64) -> (Point3D, Vector3D, Vector3D) {
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

    /// Shared second-derivative evaluation, returning
    /// `[S, Su, Sv, Suu, Suv, Svv]` (the point as a vector).
    fn eval_surface_d2(&self, u: f64, v: f64) -> [Vector3D; 6] {
        math::surface_d2(
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

    /// Box-bounded Newton minimisation of `|S(u,v) - point|^2` from the
    /// seed `(u, v)`, with exact second derivatives, within `bounds` per
    /// direction (`None`: unbounded, for a smooth periodic direction);
    /// periodic parameters are wrapped into their period on return.
    fn refine_projection(
        &self,
        point: Point3D,
        u: f64,
        v: f64,
        bounds: [Option<(f64, f64)>; 2],
    ) -> (f64, f64) {
        let (u0, u1) = self.u_param_range();
        let (v0, v1) = self.v_param_range();
        let target = Vector3D::new(point.x, point.y, point.z);
        let objective = |u: f64, v: f64| {
            let [s, su, sv, suu, suv, svv] =
                self.eval_surface_d2(curve::inward(u, bounds[0]), curve::inward(v, bounds[1]));
            let r = s - target;
            let huv = su.dot(sv) + suv.dot(r);
            (
                0.5 * r.dot(r),
                [su.dot(r), sv.dot(r)],
                [
                    [su.dot(su) + suu.dot(r), huv],
                    [huv, sv.dot(sv) + svv.dot(r)],
                ],
            )
        };
        let [u, v] = newton_min_2d(objective, [u, v], bounds, 1e-14, 100);
        let wrap = |x: f64, periodic: bool, lo: f64, hi: f64| {
            if periodic {
                lo + (x - lo).rem_euclid(hi - lo)
            } else {
                x
            }
        };
        (
            wrap(u, self.u_periodic, u0, u1),
            wrap(v, self.v_periodic, v0, v1),
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
    fn eval_point(&self, u: f64, v: f64) -> Point3D {
        BSplineSurface::eval_point(self, u, v)
    }

    fn project_point(&self, point: Point3D, tol: Tolerance) -> SurfaceProjection {
        BSplineSurface::project_point(self, point, tol)
    }

    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        BSplineSurface::contains(self, point, tol)
    }

    /// See [`BSplineSurface::eval_derivative`] for the supported orders and the
    /// panic conditions (`du + dv` must be 1 or 2).
    fn eval_derivative(&self, u: f64, v: f64, du: u32, dv: u32) -> Vector3D {
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

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        BSplineSurface::eval_points(self, uvs)
    }
}

/// The discrete local minima `(i, j)` of a row-major `nu x nv` grid of
/// values, against all 8 neighbours (wrapping in a periodic direction),
/// ties broken by index so a plateau yields one minimum.
fn grid_minima(
    values: &[f64],
    nu: usize,
    nv: usize,
    periodic: [bool; 2],
) -> impl Iterator<Item = (usize, usize)> + '_ {
    let step = |k: usize, delta: isize, n: usize, periodic: bool| {
        let m = k as isize + delta;
        if periodic {
            Some(m.rem_euclid(n as isize) as usize)
        } else {
            (0..n as isize).contains(&m).then_some(m as usize)
        }
    };
    (0..nu)
        .flat_map(move |i| (0..nv).map(move |j| (i, j)))
        .filter(move |&(i, j)| {
            let k = i * nv + j;
            let d = values[k];
            (-1..=1).all(|di| {
                (-1..=1).all(|dj| {
                    match (step(i, di, nu, periodic[0]), step(j, dj, nv, periodic[1])) {
                        (Some(a), Some(b)) => {
                            let n = a * nv + b;
                            n == k || values[n] > d || (values[n] == d && n > k)
                        }
                        _ => true,
                    }
                })
            })
        })
}

/// Samples per knot span: at least `max(degree, 2)`, and 16 over all
/// `spans`, so dense control nets are sampled as finely as their spans
/// demand.
fn samples_per_span(degree: usize, spans: usize) -> usize {
    const MIN_SAMPLES: usize = 16;
    degree.max(2).max(MIN_SAMPLES.div_ceil(spans))
}

/// Packs the pole grid (and, if rational, weights) into a flat, row-major
/// `[i_u][i_v][h]` coordinate buffer: `(x, y, z)` per pole if `weights` is
/// `None`, else the homogeneous `(x*w, y*w, z*w, w)`.
fn pack_flat_poles(poles: &[Vec<Point3D>], weights: Option<&Vec<Vec<f64>>>, h: usize) -> Vec<f64> {
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
            vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
            vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
        assert_eq!(s.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 0.25));
        assert_eq!(s.eval_point(0.0, 0.0), Point3D::new(0.0, 0.0, 0.0));
        assert_eq!(s.eval_point(1.0, 1.0), Point3D::new(1.0, 1.0, 1.0));
        assert_eq!(s.eval_point(0.3, 0.7), Point3D::new(0.3, 0.7, 0.3 * 0.7));
    }

    #[test]
    fn test_bilinear_derivatives_by_hand() {
        let s = bilinear();
        // Su = (1, 0, v), Sv = (0, 1, u).
        assert_eq!(
            s.eval_derivative(0.3, 0.7, 1, 0),
            Vector3D::new(1.0, 0.0, 0.7)
        );
        assert_eq!(
            s.eval_derivative(0.3, 0.7, 0, 1),
            Vector3D::new(0.0, 1.0, 0.3)
        );
    }

    #[test]
    fn test_eval_points_matches_mapped_eval_point() {
        let s = bilinear();
        let uvs = [(0.0, 0.0), (0.2, 0.4), (0.5, 0.5), (1.0, 1.0)];
        let expected: Vec<Point3D> = uvs.iter().map(|&(u, v)| s.eval_point(u, v)).collect();
        assert_eq!(s.eval_points(&uvs), expected);
    }

    #[test]
    #[should_panic(expected = "du + dv must be >= 1")]
    fn test_eval_derivative_zero_order_panics() {
        bilinear().eval_derivative(0.5, 0.5, 0, 0);
    }

    #[test]
    fn test_eval_derivative_second_order_bilinear() {
        // S = (u, v, u*v): Suu = Svv = 0, Suv = (0, 0, 1).
        let s = bilinear();
        assert_eq!(
            s.eval_derivative(0.5, 0.5, 2, 0),
            Vector3D::new(0.0, 0.0, 0.0)
        );
        assert_eq!(
            s.eval_derivative(0.5, 0.5, 1, 1),
            Vector3D::new(0.0, 0.0, 1.0)
        );
        assert_eq!(
            s.eval_derivative(0.5, 0.5, 0, 2),
            Vector3D::new(0.0, 0.0, 0.0)
        );
    }

    #[test]
    #[should_panic(expected = "du + dv must be <= 2")]
    fn test_eval_derivative_third_order_panics() {
        bilinear().eval_derivative(0.5, 0.5, 2, 1);
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

    fn ok_grid() -> Vec<Vec<Point3D>> {
        vec![
            vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
            vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
            vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
            vec![Point3D::new(1.0, 0.0, 0.0)], // short row
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
            vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.2, 0.0, 3.0)],
            vec![
                Point3D::new(1.0000000000000002, 1.7320508075688772, 0.0),
                Point3D::new(1.1000000000000003, 1.905255888325765, 3.0),
            ],
            vec![
                Point3D::new(-0.9999999999999996, 1.7320508075688774, 0.0),
                Point3D::new(-1.0999999999999996, 1.9052558883257653, 3.0),
            ],
            vec![
                Point3D::new(-2.0, 2.4492935982947064e-16, 0.0),
                Point3D::new(-2.2, 2.6942229581241775e-16, 3.0),
            ],
            vec![
                Point3D::new(-1.0000000000000009, -1.7320508075688767, 0.0),
                Point3D::new(-1.100000000000001, -1.9052558883257646, 3.0),
            ],
            vec![
                Point3D::new(1.0000000000000002, -1.7320508075688772, 0.0),
                Point3D::new(1.1000000000000003, -1.905255888325765, 3.0),
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

    fn bilinear_patch() -> BSplineSurface {
        let poles = vec![
            vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
            vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
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
    fn test_bspline_surface_project_point_planar() {
        let s = bilinear_patch();
        let tol = Tolerance::DEFAULT;
        let proj = s.project_point(Point3D::new(1.0, 1.0, 1.0), tol);
        assert!((proj.u - 0.5).abs() < 1e-9);
        assert!((proj.v - 0.5).abs() < 1e-9);
        assert!((proj.distance - 1.0).abs() < 1e-9);
        // Beyond the edge: boundary foot wins.
        let edge = s.project_point(Point3D::new(3.0, 1.0, 0.0), tol);
        assert!((edge.u - 1.0).abs() < 1e-9);
        assert!((edge.v - 0.5).abs() < 1e-9);
        assert!((edge.distance - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bspline_surface_contains_planar() {
        let s = bilinear_patch();
        let tol = Tolerance::DEFAULT;
        assert!(s.contains(Point3D::new(1.0, 1.0, 0.0), tol));
        assert!(s.contains(s.eval_point(0.25, 0.75), tol));
        assert!(!s.contains(Point3D::new(1.0, 1.0, 1.0), tol));
    }

    #[test]
    fn test_bspline_surface_project_point_tube() {
        let s = periodic_tube();
        let tol = Tolerance::DEFAULT;
        // On-surface point projects to (near-)zero distance.
        let on = s.eval_point(0.5, 0.5);
        assert_eq!(s.project_point(on, tol).distance, 0.0);
        assert!(s.contains(on, tol));
        // Off-surface: beat against dense brute-force sampling.
        let query = Point3D::new(0.0, 0.0, 5.0);
        let proj = s.project_point(query, tol);
        let (u0, u1) = ParametricSurface::u_bounds(&s);
        let (v0, v1) = ParametricSurface::v_bounds(&s);
        let mut best = f64::INFINITY;
        for j in 0..=60 {
            for i in 0..=60 {
                let d = s
                    .eval_point(
                        u0 + (u1 - u0) * i as f64 / 60.0,
                        v0 + (v1 - v0) * j as f64 / 60.0,
                    )
                    .distance(query);
                best = best.min(d);
            }
        }
        assert!(proj.distance <= best + 1e-9);
        assert!(best - proj.distance < 1e-2);
    }

    // ---- transformed ----

    fn transformed_test_patch() -> BSplineSurface {
        let poles = vec![
            vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
            vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 1.0)],
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
    fn test_bspline_surface_transformed_translation_spot_check() {
        let surface = transformed_test_patch();
        let t = Transform::translation(Vector3D::new(0.0, 0.0, 4.0));
        let moved = surface.transformed(t);
        // Hand-computed patch midpoint, shifted.
        assert_eq!(moved.eval_point(0.5, 0.5), Point3D::new(0.5, 0.5, 4.25));
        assert_eq!(moved.u_degree(), surface.u_degree());
        assert_eq!(moved.v_degree(), surface.v_degree());
    }

    #[test]
    fn test_bspline_surface_transformed_round_trip_and_mirror() {
        use crate::{Axis3D, Frame3D};
        use std::f64::consts::PI;
        let surface = transformed_test_patch();
        let axis = Axis3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
        let t = Transform::rotation(axis, PI / 2.0).then(Transform::scaling(Point3D::ORIGIN, 2.0));
        let moved = surface.transformed(t);
        let tol = Tolerance::DEFAULT;
        for (u, v) in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.25)] {
            assert!(moved.contains(t.apply_point(surface.eval_point(u, v)), tol));
        }
        let inv =
            Transform::scaling(Point3D::ORIGIN, 0.5).then(Transform::rotation(axis, -PI / 2.0));
        let back = moved.transformed(inv);
        for (u, v) in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.25)] {
            assert!(back.eval_point(u, v).distance(surface.eval_point(u, v)) < 1e-9);
        }
        // Mirror preserves the point set (as a fresh pole net).
        let m = Transform::mirror_plane(Frame3D::WORLD);
        let mirrored = surface.transformed(m);
        for (u, v) in [(0.0, 0.0), (0.5, 0.5), (1.0, 0.25)] {
            assert!(mirrored.contains(m.apply_point(surface.eval_point(u, v)), tol));
        }
        assert_eq!(mirrored.transformed(m), surface);
    }
}
