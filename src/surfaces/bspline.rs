//! B-spline (NURBS) surfaces in 3D: tensor-product de Boor evaluation with
//! first derivatives, rational (weighted) poles, and periodic (closed) surfaces
//! in either direction, a thin wrapper over [`crate::surface_math::bspline`].
//!
//! Construction stores the pole grid row-major, with a separate weight per
//! pole if rational, and each direction's knot vector with its flattened
//! knot sequence, so evaluation never re-derives them.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::curve_math::bspline as curve;
use crate::math::boxtree::{BoxTree, Lazy, box_distance};
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
    u_knots: curve::KnotVector,
    v_knots: curve::KnotVector,
    /// The pole grid, row-major `[i_u][i_v]`: a row per `u` pole.
    poles: Vec<Point3D>,
    /// One weight per pole, laid out as `poles`, if rational.
    weights: Option<Vec<f64>>,
    /// Bounding-box hierarchy of the knot-span patches, built on the first
    /// projection.
    patches: Lazy<PatchTree>,
}

/// The knot-span boundaries of each direction, whether the surface is a
/// crease or ends there ([`curve::crease_breaks`]), each span's Bezier
/// matrix ([`curve::span_bezier_matrices`]), and the box hierarchy over
/// the patches (`i_u * (v_breaks.len() - 1) + i_v`).
#[derive(Clone)]
struct PatchTree {
    u_breaks: Vec<f64>,
    v_breaks: Vec<f64>,
    u_creases: Vec<bool>,
    v_creases: Vec<bool>,
    u_spans: (Vec<usize>, Vec<f64>),
    v_spans: (Vec<usize>, Vec<f64>),
    tree: BoxTree,
}

impl PatchTree {
    /// The parameter rectangle `[u0, u1, v0, v1]` of patch `k` and whether
    /// each of its edges is a crease or a boundary.
    fn patch(&self, k: usize) -> ([f64; 4], [bool; 4]) {
        let n_vspans = self.v_breaks.len() - 1;
        let (a, b) = (k / n_vspans, k % n_vspans);
        (
            [
                self.u_breaks[a],
                self.u_breaks[a + 1],
                self.v_breaks[b],
                self.v_breaks[b + 1],
            ],
            [
                self.u_creases[a],
                self.u_creases[a + 1],
                self.v_creases[b],
                self.v_creases[b + 1],
            ],
        )
    }

    /// The number of patches.
    fn len(&self) -> usize {
        (self.u_breaks.len() - 1) * (self.v_breaks.len() - 1)
    }

    /// Distances below this much of the best one found are not worth
    /// telling apart: a relative `1e-12` of the surface's extent and of
    /// its distance to `point`.
    fn resolution(&self, point: Point3D) -> f64 {
        let b = self.tree.bounds();
        let diagonal =
            ((b[3] - b[0]).powi(2) + (b[4] - b[1]).powi(2) + (b[5] - b[2]).powi(2)).sqrt();
        1e-12 * (diagonal + box_distance(point, b))
    }
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
        let u_knots = curve::KnotVector::new(u_degree, u_knots, u_multiplicities, u_periodic, n_u)?;
        let v_knots = curve::KnotVector::new(v_degree, v_knots, v_multiplicities, v_periodic, n_v)?;

        Ok(BSplineSurface {
            u_knots,
            v_knots,
            poles: poles.concat(),
            weights: weights.map(|w| w.concat()),
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
        self.u_knots.degree()
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
        self.v_knots.degree()
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
        self.u_knots.is_periodic()
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
        self.v_knots.is_periodic()
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
        BSplineSurface {
            u_knots: self.u_knots.clone(),
            v_knots: self.v_knots.clone(),
            poles: self
                .poles
                .iter()
                .map(|&p| transform.apply_point(p))
                .collect(),
            weights: self.weights.clone(),
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
        math::surface_point(&self.u_knots, &self.v_knots, self.pole_view(), u, v)
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

    /// Evaluates the point and the first derivatives `[Su, Sv]` at
    /// parameters `(u, v)`, from one evaluation of the basis functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, Vector3D};
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
    ///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 0.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false,
    /// )
    /// .unwrap();
    /// let (point, [su, sv]) = surface.eval_d1(0.5, 0.5);
    /// assert_eq!(point, Point3D::new(1.0, 1.0, 0.0));
    /// assert_eq!(su, Vector3D::new(2.0, 0.0, 0.0));
    /// assert_eq!(sv, Vector3D::new(0.0, 2.0, 0.0));
    /// ```
    pub fn eval_d1(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2]) {
        let (s, su, sv) = self.eval_surface_d1(u, v);
        (s, [su, sv])
    }

    /// Evaluates the point, the first derivatives `[Su, Sv]` and the second
    /// derivatives `[Suu, Suv, Svv]` at parameters `(u, v)`, from one
    /// evaluation of the basis functions.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, Vector3D};
    /// let poles = vec![
    ///     vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 2.0, 0.0)],
    ///     vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 2.0, 1.0)],
    /// ];
    /// let surface = BSplineSurface::new(
    ///     1, 1, poles, vec![0.0, 1.0], vec![2, 2], vec![0.0, 1.0], vec![2, 2], false, false,
    /// )
    /// .unwrap();
    /// let (_, _, [suu, suv, svv]) = surface.eval_d2(0.5, 0.5);
    /// assert_eq!(suu, Vector3D::ZERO);
    /// assert_eq!(suv, Vector3D::new(0.0, 0.0, 1.0));
    /// assert_eq!(svv, Vector3D::ZERO);
    /// ```
    pub fn eval_d2(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2], [Vector3D; 3]) {
        let [s, su, sv, suu, suv, svv] = self.eval_surface_d2(u, v);
        (Point3D::ORIGIN + s, [su, sv], [suu, suv, svv])
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
    /// ordered by ascending distance, the first being
    /// [`BSplineSurface::project_point`].
    ///
    /// Every knot-span patch is searched as in
    /// [`BSplineSurface::project_point`], but keeping each minimum found
    /// rather than only the closest: a subpatch is dropped only when it
    /// provably holds no local minimum. Each minimum is polished by the
    /// same Newton minimisation within the C1 region around it (C0 crease
    /// knots bound it), so minima on an edge, corner or crease are reached
    /// exactly. Minima within `tol.confusion` of each other merge.
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
        let global = self.project_point(point, tol);
        let patches = self.patch_tree();
        let mut search = PatchSearch::new(self, point, patches.resolution(point), true);
        for k in 0..patches.len() {
            let (rect, creases) = patches.patch(k);
            if let Some(root) = search.root(patches, k) {
                search.search(root, rect, creases);
            }
        }
        let u_regions = self.u_knots.smooth_regions();
        let v_regions = self.v_knots.smooth_regions();
        let (u_period, v_period) = (self.u_knots.period(), self.v_knots.period());
        // A minimum on a subpatch edge inside a smooth region is where the
        // search stopped, not necessarily a minimum of the surface: polish
        // across subpatches.
        let mut candidates = vec![(global.u, global.v)];
        for (u, v) in search.minima {
            for (ubounds, u) in curve::regions_at(&u_regions, u, u_period) {
                for (vbounds, v) in curve::regions_at(&v_regions, v, v_period) {
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
        // The projection leads, even past a rounding-level tie.
        out[1..].sort_by(|x, y| x.distance.total_cmp(&y.distance));
        out
    }

    /// Projects `point` onto the surface: the global minimum of the
    /// distance, as the first entry of [`BSplineSurface::extrema`] but
    /// without visiting the whole surface. Distances within
    /// `tol.confusion` snap to `0.0`, matching [`BSplineSurface::contains`].
    ///
    /// Branch and bound, exact up to rounding. The knot-span patches whose
    /// control poles' bounding boxes (convex hull property) may hold the
    /// closest point are found from a hierarchy built once per surface.
    /// Each patch's Bernstein (Bezier) control net is then halved by de
    /// Casteljau subdivision, closest control point first, until each
    /// subpatch is either
    ///
    /// - farther than the closest point so far, its control points' boxes
    ///   being a lower bound (a box fitted to a small subpatch is flat, so
    ///   the bound is tight to second order);
    /// - free of local minima, by interval bounds on the gradient and
    ///   Hessian of the squared distance from the nets of its derivatives
    ///   (a gradient that keeps its sign towards a smooth edge, or a
    ///   direction of negative curvature);
    /// - or holding a single minimum, convex there, which a box-bounded
    ///   Newton minimisation with exact second derivatives then reaches.
    ///
    /// A subpatch whose gradient pins its minima to a crease or boundary
    /// edge is searched as that edge curve. Subpatch corners lie on the
    /// surface and tighten the closest point. The search stops within a
    /// relative `1e-12` of the surface's extent; where the distance is
    /// flat to that resolution (a point on an axis of revolution) a cap on
    /// the subdivision falls back to Newton from the closest subpatches.
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
        let mut search = PatchSearch::new(self, point, patches.resolution(point), false);
        // Every patch that may hold the closest point, by their pole boxes
        // then their Bernstein nets (whose corners tighten the bound),
        // searched closest control point first.
        let mut roots = Vec::new();
        patches.tree.nearest_first(point, |k| {
            if let Some(root) = search.root(patches, k) {
                let (rect, creases) = patches.patch(k);
                roots.push((root, rect, creases));
            }
            search.best.0
        });
        roots.sort_by(|a, b| a.0.key.total_cmp(&b.0.key));
        for (root, rect, creases) in roots {
            search.search(root, rect, creases);
        }
        let (distance, u, v) = search.best;
        SurfaceProjection {
            u,
            v,
            distance: snap_distance(distance, tol.confusion),
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
            let u_breaks = self.u_knots.breaks();
            let v_breaks = self.v_knots.breaks();
            let boxes = math::patch_boxes(
                &self.u_knots,
                &self.v_knots,
                self.pole_view(),
                &u_breaks,
                &v_breaks,
            );
            PatchTree {
                tree: BoxTree::new(&boxes, v_breaks.len() - 1),
                u_spans: self.u_knots.span_bezier_matrices(&u_breaks),
                v_spans: self.v_knots.span_bezier_matrices(&v_breaks),
                u_creases: self.u_knots.crease_breaks(&u_breaks),
                v_creases: self.v_knots.crease_breaks(&v_breaks),
                u_breaks,
                v_breaks,
            }
        })
    }

    /// Shared first-derivative evaluation, returning `(S, Su, Sv)`.
    fn eval_surface_d1(&self, u: f64, v: f64) -> (Point3D, Vector3D, Vector3D) {
        let [s, su, sv, ..] =
            math::surface_derivatives::<1>(&self.u_knots, &self.v_knots, self.pole_view(), u, v);
        (Point3D::ORIGIN + s, su, sv)
    }

    /// Shared second-derivative evaluation, returning
    /// `[S, Su, Sv, Suu, Suv, Svv]` (the point as a vector).
    fn eval_surface_d2(&self, u: f64, v: f64) -> [Vector3D; 6] {
        math::surface_derivatives::<2>(&self.u_knots, &self.v_knots, self.pole_view(), u, v)
    }

    /// The poles as read by evaluation.
    fn pole_view(&self) -> curve::Poles<'_> {
        curve::Poles {
            points: &self.poles,
            weights: self.weights.as_deref(),
        }
    }

    /// Values per point of a Bernstein net: homogeneous if rational.
    fn net_stride(&self) -> usize {
        if self.is_rational() { 4 } else { 3 }
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
            wrap(u, self.is_u_periodic(), u0, u1),
            wrap(v, self.is_v_periodic(), v0, v1),
        )
    }

    /// `u` parameter bounds `(first, last)`: for a clamped direction the active
    /// flat span, for a periodic direction the full period `(knots[0],
    /// knots[last])`.
    fn u_param_range(&self) -> (f64, f64) {
        self.u_knots.bounds()
    }

    /// `v` parameter bounds `(first, last)`, analogous to [`Self::u_param_range`].
    fn v_param_range(&self) -> (f64, f64) {
        self.v_knots.bounds()
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

    fn eval_d1(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2]) {
        BSplineSurface::eval_d1(self, u, v)
    }

    fn eval_d2(&self, u: f64, v: f64) -> (Point3D, [Vector3D; 2], [Vector3D; 3]) {
        BSplineSurface::eval_d2(self, u, v)
    }

    fn u_bounds(&self) -> (f64, f64) {
        self.u_param_range()
    }

    fn v_bounds(&self) -> (f64, f64) {
        self.v_param_range()
    }

    fn u_period(&self) -> Option<f64> {
        self.u_knots.period()
    }

    fn v_period(&self) -> Option<f64> {
        self.v_knots.period()
    }

    fn eval_points(&self, uvs: &[(f64, f64)]) -> Vec<Point3D> {
        BSplineSurface::eval_points(self, uvs)
    }
}

/// Subdivisions per knot-span patch before the search settles for Newton
/// from the closest remaining subpatches: reached only where the distance
/// is flat to the search resolution. Once one patch of a closest-point
/// search reaches it, the later patches get an eighth of it.
const MAX_SPLITS: usize = 256;

/// Newton refinements once [`MAX_SPLITS`] is reached.
const MAX_FALLBACKS: usize = 8;
/// Subdivisions from a patch down to one subpatch: halving below this,
/// parameter steps reach rounding.
const MAX_DEPTH: u32 = 80;

/// Branch and bound over the Bernstein subpatches of knot-span patches for
/// the minima of the distance to a point (see
/// [`BSplineSurface::project_point`]).
struct PatchSearch<'a> {
    surface: &'a BSplineSurface,
    point: Point3D,
    /// Net shape: `rows x cols` points of `h` values, `size` in all.
    rows: usize,
    cols: usize,
    h: usize,
    size: usize,
    /// Storage of the pending subpatches' nets, `size` values per slot.
    nets: Vec<f64>,
    /// Offsets of unused slots in `nets`.
    free: Vec<usize>,
    pending: BinaryHeap<Subpatch>,
    /// Euclidean control points of the subpatch at hand.
    points: Vec<[f64; 3]>,
    /// A copy of the net being split.
    scratch: Vec<f64>,
    /// `(distance, u, v)` of the closest point so far.
    best: (f64, f64, f64),
    /// Subpatches closer than this to `best` need no search.
    resolution: f64,
    /// Whether every local minimum is wanted, not only the closest.
    all: bool,
    /// The minima found, when `all`.
    minima: Vec<(f64, f64)>,
    /// Subdivisions allowed per patch.
    budget: usize,
}

/// A subpatch awaiting its visit: its net at offset `slot`, its parameter
/// rectangle `[u0, u1, v0, v1]`, a lower bound `lower` on its distance,
/// and the visiting order `key`, its closest control point's distance. Once
/// its minima are known to lie on its edge `side` of `direction`, `edge`
/// holds `(direction, side)` and the net is that edge curve's.
#[derive(Clone, Copy)]
struct Subpatch {
    lower: f64,
    key: f64,
    slot: usize,
    rect: [f64; 4],
    depth: u32,
    edge: Option<(usize, usize)>,
}

impl PartialEq for Subpatch {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Subpatch {}

impl PartialOrd for Subpatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Subpatch {
    /// Reversed, so a [`BinaryHeap`] pops the lowest key first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.key.total_cmp(&self.key)
    }
}

/// What a subpatch's derivative bounds say about its minima.
enum Verdict {
    /// No local minimum of the surface: the gradient keeps a sign towards
    /// a smooth edge, or the distance curves down everywhere.
    None,
    /// At the corner `[s, t]` (each `0` or `1`), the gradient pinning it
    /// to two crease or boundary edges.
    Corner([usize; 2]),
    /// On edge `side` (`0` or `1`) of direction `direction` (`0` for `u`),
    /// pinned there by the gradient, convex along it.
    Edge(usize, usize),
    /// On edge `side` of `direction`, pinned there by the gradient, but
    /// anywhere along it as far as known.
    Pinned(usize, usize),
    /// Anywhere, the distance being convex.
    Convex,
    /// Not known yet: split along `u` if true, else along `v`.
    Unknown(bool),
}

impl<'a> PatchSearch<'a> {
    /// An empty search for the minima of the distance from `point` (every
    /// local one if `all`), subpatches within `resolution` of the best
    /// distance so far being settled.
    fn new(surface: &'a BSplineSurface, point: Point3D, resolution: f64, all: bool) -> Self {
        let (rows, cols) = (surface.u_degree() + 1, surface.v_degree() + 1);
        let h = surface.net_stride();
        let size = rows * cols * h;
        PatchSearch {
            surface,
            point,
            rows,
            cols,
            h,
            size,
            nets: Vec::new(),
            free: Vec::new(),
            pending: BinaryHeap::new(),
            points: Vec::with_capacity(rows * cols),
            scratch: vec![0.0; size],
            best: (f64::INFINITY, 0.0, 0.0),
            resolution,
            all,
            minima: Vec::new(),
            budget: MAX_SPLITS,
        }
    }

    /// The whole patch `k` of `patches` as a subpatch to search, unless it
    /// is farther than the closest point so far.
    fn root(&mut self, patches: &PatchTree, k: usize) -> Option<Subpatch> {
        let s = self.surface;
        let slot = self.slot();
        let n_vspans = patches.v_breaks.len() - 1;
        let (a, b) = (k / n_vspans, k % n_vspans);
        let (p, q) = (s.u_degree(), s.v_degree());
        let (mu, mv) = ((p + 1) * (p + 1), (q + 1) * (q + 1));
        math::patch_bezier(
            (
                &patches.u_spans.1[a * mu..(a + 1) * mu],
                patches.u_spans.0[a],
                p,
                s.u_knots.pole_count(),
            ),
            (
                &patches.v_spans.1[b * mv..(b + 1) * mv],
                patches.v_spans.0[b],
                q,
                s.v_knots.pole_count(),
            ),
            s.pole_view(),
            &mut self.scratch,
            &mut self.nets[slot..slot + self.size],
        );
        self.node(slot, patches.patch(k).0, 0, None)
    }

    /// Searches the patch over `rect`, with the subpatch `root` over all of
    /// it, whose edges are creases or boundaries as flagged in `creases`.
    fn search(&mut self, root: Subpatch, rect: [f64; 4], creases: [bool; 4]) {
        self.pending.push(root);
        let (mut splits, mut fallbacks) = (0, 0);
        while let Some(node) = self.pending.pop() {
            if !self.all && node.lower >= self.best.0 - self.resolution {
                self.free.push(node.slot);
                continue;
            }
            // Only the patch's own edges can be creases.
            let edge = std::array::from_fn(|i| creases[i] && node.rect[i] == rect[i]);
            let verdict = self.verdict(&node, edge);
            match verdict {
                Verdict::None => {}
                Verdict::Corner([s, t]) => {
                    let p = self.points[s * (self.rows - 1) * self.cols + t * (self.cols - 1)];
                    let distance = Point3D::new(p[0], p[1], p[2]).distance(self.point);
                    self.found(distance, node.rect[s], node.rect[2 + t]);
                }
                Verdict::Edge(direction, side) => {
                    let start = self.closest_control_point(Some((direction, side)));
                    self.minimise(&node, start);
                }
                Verdict::Pinned(direction, side) => {
                    // Search the edge curve instead, keeping the slot.
                    math::collapse_net(
                        &mut self.nets[node.slot..node.slot + self.size],
                        self.rows,
                        self.cols,
                        self.h,
                        direction == 0,
                        side,
                    );
                    self.push(node.slot, node.rect, node.depth, Some((direction, side)));
                    continue;
                }
                Verdict::Convex => {
                    let start = self.closest_control_point(None);
                    self.minimise(&node, start);
                }
                Verdict::Unknown(along_u) => {
                    if node.depth < MAX_DEPTH && splits < self.budget {
                        splits += 1;
                        self.split(&node, along_u);
                    } else if fallbacks < MAX_FALLBACKS {
                        fallbacks += 1;
                        let start = self.closest_control_point(node.edge);
                        self.minimise(&node, start);
                    }
                }
            }
            self.free.push(node.slot);
        }
        // The distance is flat to the resolution here (a point on an axis
        // of revolution): wherever else it is, other patches can only
        // match it, so they get a smaller budget.
        if splits >= self.budget && !self.all {
            self.budget = MAX_SPLITS / 8;
        }
    }

    /// A free net slot.
    fn slot(&mut self) -> usize {
        self.free.pop().unwrap_or_else(|| {
            let slot = self.nets.len();
            self.nets.resize(slot + self.size, 0.0);
            slot
        })
    }

    /// Queues the subpatch with the net at `slot` (see [`Self::node`]).
    fn push(&mut self, slot: usize, rect: [f64; 4], depth: u32, edge: Option<(usize, usize)>) {
        if let Some(node) = self.node(slot, rect, depth, edge) {
            self.pending.push(node);
        }
    }

    /// The subpatch with the net at `slot`, unless only the closest point
    /// is wanted and the subpatch is farther than the closest so far; its
    /// corners, on the surface, may become that point.
    fn node(
        &mut self,
        slot: usize,
        rect: [f64; 4],
        depth: u32,
        edge: Option<(usize, usize)>,
    ) -> Option<Subpatch> {
        let (rows, cols) = (self.rows, self.cols);
        if self.all {
            // Every subpatch counts, in any order.
            return Some(Subpatch {
                lower: 0.0,
                key: 0.0,
                slot,
                rect,
                depth,
                edge,
            });
        }
        math::net_points(&self.nets[slot..slot + self.size], self.h, &mut self.points);
        for (s, t) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let p = self.points[s * (rows - 1) * cols + t * (cols - 1)];
            let distance = Point3D::new(p[0], p[1], p[2]).distance(self.point);
            if distance < self.best.0 {
                // A collapsed net's corners sit on its edge.
                let [s, t] = match edge {
                    Some((0, side)) => [side, t],
                    Some((_, side)) => [s, side],
                    None => [s, t],
                };
                self.best = (distance, rect[s], rect[2 + t]);
            }
        }
        let q = [self.point.x, self.point.y, self.point.z];
        let lower = math::net_distance_bound(&self.points, rows, cols, q);
        if lower >= self.best.0 - self.resolution {
            self.free.push(slot);
            return None;
        }
        let key = self
            .points
            .iter()
            .map(|x| (0..3).map(|c| (x[c] - q[c]).powi(2)).sum::<f64>())
            .fold(f64::INFINITY, f64::min)
            .sqrt();
        Some(Subpatch {
            lower,
            key,
            slot,
            rect,
            depth,
            edge,
        })
    }

    /// Classifies `node` given its crease or boundary edges `creases`
    /// (`[u0, u1, v0, v1]`), leaving its Euclidean control points in
    /// `points`.
    fn verdict(&mut self, node: &Subpatch, creases: [bool; 4]) -> Verdict {
        let (rows, cols, h) = (self.rows, self.cols, self.h);
        let net = &self.nets[node.slot..node.slot + self.size];
        math::net_points(net, h, &mut self.points);
        let q = [self.point.x, self.point.y, self.point.z];
        // The side each direction's gradient pins the minimum to; a
        // collapsed net stays on its edge.
        let pinned = |gradient: [math::Interval; 2]| -> [Option<usize>; 2] {
            std::array::from_fn(|direction| match node.edge {
                Some((d, side)) if d == direction => Some(side),
                _ => {
                    let g = gradient[direction];
                    if g.lo > 0.0 {
                        Some(0)
                    } else if g.hi < 0.0 {
                        Some(1)
                    } else {
                        None
                    }
                }
            })
        };
        // Visiting every patch, most of which only the gradient rules out:
        // try it alone first.
        if self.all {
            let gradient =
                math::distance_bounds(net, &self.points, rows, cols, h, q, false).gradient;
            let smooth = |direction: usize, side: usize| !creases[2 * direction + side];
            if pinned(gradient)
                .iter()
                .enumerate()
                .any(|(direction, side)| side.is_some_and(|side| smooth(direction, side)))
            {
                return Verdict::None;
            }
        }
        let bounds = math::distance_bounds(net, &self.points, rows, cols, h, q, true);
        let [fss, fst, ftt] = bounds.hessian;
        let pinned = pinned(bounds.gradient);
        for direction in 0..2 {
            let crease = |side: usize| creases[2 * direction + side];
            // Descending through a smooth edge, or curving down towards
            // two of them: the minima are elsewhere.
            let concave = bounds.hessian[2 * direction].hi < 0.0;
            if pinned[direction].is_some_and(|side| !crease(side))
                || (concave && !crease(0) && !crease(1))
            {
                return Verdict::None;
            }
        }
        if !creases.contains(&true) && nowhere_convex([fss, fst, ftt]) {
            return Verdict::None;
        }
        match pinned {
            [Some(s), Some(t)] => Verdict::Corner([s, t]),
            [Some(s), None] if ftt.lo > 0.0 => Verdict::Edge(0, s),
            [None, Some(t)] if fss.lo > 0.0 => Verdict::Edge(1, t),
            // Along the edge, as a curve.
            [Some(s), None] if node.edge.is_none() => Verdict::Pinned(0, s),
            [None, Some(t)] if node.edge.is_none() => Verdict::Pinned(1, t),
            [Some(_), None] => Verdict::Unknown(false),
            [None, Some(_)] => Verdict::Unknown(true),
            [None, None]
                if fss.lo > 0.0 && ftt.lo > 0.0 && fss.lo * ftt.lo > fst.magnitude().powi(2) =>
            {
                Verdict::Convex
            }
            // Along the direction the patch stretches most.
            _ => Verdict::Unknown(bounds.stretch[0] >= bounds.stretch[1]),
        }
    }

    /// The parameters, as fractions `(s, t)` of the subpatch, of its
    /// control point closest to the point, among those on edge `side` of
    /// `direction` if given.
    fn closest_control_point(&self, edge: Option<(usize, usize)>) -> (f64, f64) {
        let (p, q) = (self.rows - 1, self.cols - 1);
        let target = [self.point.x, self.point.y, self.point.z];
        let distance = |i: usize, j: usize| {
            let x = self.points[i * self.cols + j];
            (0..3).map(|c| (x[c] - target[c]).powi(2)).sum::<f64>()
        };
        let candidates: Vec<(usize, usize)> = match edge {
            Some((0, side)) => (0..=q).map(|j| (side * p, j)).collect(),
            Some((_, side)) => (0..=p).map(|i| (i, side * q)).collect(),
            None => (0..=p).flat_map(|i| (0..=q).map(move |j| (i, j))).collect(),
        };
        let (i, j) = candidates
            .into_iter()
            .min_by(|a, b| distance(a.0, a.1).total_cmp(&distance(b.0, b.1)))
            .unwrap_or((0, 0));
        (i as f64 / p as f64, j as f64 / q as f64)
    }

    /// Newton minimisation within the subpatch from the fractions `(s, t)`.
    fn minimise(&mut self, node: &Subpatch, (s, t): (f64, f64)) {
        let r = node.rect;
        let (u, v) = self.surface.refine_projection(
            self.point,
            r[0] + s * (r[1] - r[0]),
            r[2] + t * (r[3] - r[2]),
            [Some((r[0], r[1])), Some((r[2], r[3]))],
        );
        let distance = self.surface.eval_point(u, v).distance(self.point);
        self.found(distance, u, v);
    }

    /// Records a minimum at `(u, v)`, `distance` from the point.
    fn found(&mut self, distance: f64, u: f64, v: f64) {
        if distance < self.best.0 {
            self.best = (distance, u, v);
        }
        if self.all {
            self.minima.push((u, v));
        }
    }

    /// Halves the subpatch along `u` (else `v`) and queues both halves.
    fn split(&mut self, node: &Subpatch, along_u: bool) {
        self.scratch
            .copy_from_slice(&self.nets[node.slot..node.slot + self.size]);
        let (left, right) = (self.slot(), self.slot());
        let (lo, hi) = if left < right {
            let (a, b) = self.nets.split_at_mut(right);
            (&mut a[left..left + self.size], &mut b[..self.size])
        } else {
            let (a, b) = self.nets.split_at_mut(left);
            (&mut b[..self.size], &mut a[right..right + self.size])
        };
        math::halve_net(&self.scratch, self.rows, self.cols, self.h, along_u, lo, hi);
        let r = node.rect;
        let (first, second) = if along_u {
            let mid = 0.5 * (r[0] + r[1]);
            ([r[0], mid, r[2], r[3]], [mid, r[1], r[2], r[3]])
        } else {
            let mid = 0.5 * (r[2] + r[3]);
            ([r[0], r[1], r[2], mid], [r[0], r[1], mid, r[3]])
        };
        self.push(left, first, node.depth + 1, node.edge);
        self.push(right, second, node.depth + 1, node.edge);
    }
}

/// Whether none of the symmetric matrices `[[a, b], [b, c]]` with entries
/// in `[a, b, c]` is positive semidefinite, as the Hessian at a local
/// minimum is: their determinant stays negative, or they all curve down
/// along one direction (the eigenvector of the central matrix's lower
/// eigenvalue).
fn nowhere_convex([a, b, c]: [math::Interval; 3]) -> bool {
    let largest_ac = (a.lo * c.lo)
        .max(a.lo * c.hi)
        .max(a.hi * c.lo)
        .max(a.hi * c.hi);
    let smallest_b = if b.lo > 0.0 {
        b.lo
    } else if b.hi < 0.0 {
        -b.hi
    } else {
        0.0
    };
    if largest_ac < smallest_b * smallest_b {
        return true;
    }
    let (am, bm, cm) = (a.mid(), b.mid(), c.mid());
    let lower = 0.5 * (am + cm) - (0.25 * (am - cm).powi(2) + bm * bm).sqrt();
    let (x, y) = if (am - lower).abs() > (cm - lower).abs() {
        (bm, lower - am)
    } else {
        (lower - cm, bm)
    };
    let along = if x * y >= 0.0 { b.hi } else { b.lo };
    (x != 0.0 || y != 0.0) && x * x * a.hi + 2.0 * x * y * along + y * y * c.hi < 0.0
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

    // ---- exact projection ----

    /// A Bernstein net evaluated by de Casteljau, `[rows][cols][h]`.
    fn bernstein_point(net: &[f64], rows: usize, cols: usize, h: usize, s: f64, t: f64) -> Point3D {
        let lerp = |a: &[f64], n: usize, x: f64| {
            let mut a = a.to_vec();
            for r in 1..n {
                for k in 0..(n - r) * h {
                    a[k] = (1.0 - x) * a[k] + x * a[k + h];
                }
            }
            a[..h].to_vec()
        };
        let column: Vec<f64> = (0..rows)
            .flat_map(|i| lerp(&net[i * cols * h..(i + 1) * cols * h], cols, t))
            .collect();
        let p = lerp(&column, rows, s);
        let w = if h == 4 { p[3] } else { 1.0 };
        Point3D::new(p[0] / w, p[1] / w, p[2] / w)
    }

    /// An extruded degree-6 profile `z(x)`, `x` linear in `u`, with two
    /// dips of nearly equal depth, the right one deeper: one knot span.
    fn two_basins() -> BSplineSurface {
        let z = [1.0, 1.0, -1.0, 1.2, -1.04, 1.0, 1.0];
        let poles = (0..7)
            .map(|i| {
                (0..2)
                    .map(|j| Point3D::new(i as f64 / 6.0, j as f64, z[i]))
                    .collect()
            })
            .collect();
        BSplineSurface::new(
            6,
            1,
            poles,
            vec![0.0, 1.0],
            vec![7, 7],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    /// The distance from `point` to the profile of [`two_basins`] at
    /// `v = 0.5`, by dense sampling then golden-section refinement.
    fn profile_distance(s: &BSplineSurface, point: Point3D) -> f64 {
        let d = |u: f64| s.eval_point(u.clamp(0.0, 1.0), 0.5).distance(point);
        let n = 20_000;
        let best = (0..=n)
            .map(|i| i as f64 / n as f64)
            .min_by(|a, b| d(*a).total_cmp(&d(*b)))
            .unwrap();
        let (mut a, mut b) = (best - 1.0 / n as f64, best + 1.0 / n as f64);
        let g = 0.5 * (5f64.sqrt() - 1.0);
        for _ in 0..80 {
            let (x, y) = (b - g * (b - a), a + g * (b - a));
            if d(x) < d(y) {
                b = y;
            } else {
                a = x;
            }
        }
        d(0.5 * (a + b))
    }

    #[test]
    fn test_project_point_two_basins_in_one_patch() {
        let s = two_basins();
        let tol = Tolerance::DEFAULT;
        // Points above and between the dips, some nearer the shallower one.
        for x in [0.38, 0.42, 0.45, 0.48, 0.5, 0.55] {
            let query = Point3D::new(x, 0.5, 0.6);
            let proj = s.project_point(query, tol);
            let reference = profile_distance(&s, query);
            assert!(
                (proj.distance - reference).abs() < 1e-12,
                "x = {x}: {} vs {reference}",
                proj.distance
            );
            // The foot is a critical point: the residual is normal there.
            let (p, su, sv) = s.eval_surface_d1(proj.u, proj.v);
            let r = p - query;
            assert!(r.dot(su).abs() < 1e-9 * su.magnitude() * r.magnitude());
            assert!(r.dot(sv).abs() < 1e-9 * sv.magnitude() * r.magnitude());
            let extrema = s.extrema(query, tol);
            assert_eq!(extrema[0].distance, proj.distance);
            // Both dips are local minima.
            assert!(extrema.len() >= 2, "x = {x}: {extrema:?}");
        }
    }

    /// A folded net of degree 2 by 5, three `u` spans and two `v` spans,
    /// whose sheets pass close to each other: many points have a second
    /// local minimum a short distance away in the same patch.
    fn folded_net() -> BSplineSurface {
        let row = |p: &[(f64, f64, f64)]| -> Vec<Point3D> {
            p.iter().map(|&(x, y, z)| Point3D::new(x, y, z)).collect()
        };
        #[rustfmt::skip]
        let poles = vec![
            row(&[(4.52, -4.72, -0.97), (0.94, -2.11, -2.98), (0.99, -4.89, 0.49), (4.03, 2.02, 2.08), (2.35, -4.38, -1.07),
                  (-2.22, -2.8, -3.92), (-4.6, 2.81, -3.13), (3.21, 2.17, 2.1), (0.74, 0.06, -3.3), (3.03, 3.51, 1.31)]),
            row(&[(2.04, -4.83, 3.15), (-4.97, 1.53, 2.88), (0.8, 4.4, 3.79), (-3.33, -3.15, -4.61), (3.81, -4.51, 1.64),
                  (0.84, -1.12, -4.38), (-1.41, -3.42, -4.87), (-1.57, -3.53, 0.75), (-4.0, 4.32, -1.94), (-2.98, 4.31, -3.09)]),
            row(&[(-4.47, 0.45, 0.78), (0.81, -0.98, 2.47), (4.44, 2.43, 2.29), (-1.33, -4.42, 0.08), (1.79, -2.52, -1.4),
                  (-0.55, -4.36, -4.45), (-4.85, 3.25, -2.06), (0.55, 3.93, -4.22), (2.3, 0.75, 0.34), (2.87, 3.87, -3.4)]),
            row(&[(-1.7, 2.4, -2.3), (-0.07, -0.04, 1.97), (-4.41, -0.31, -0.26), (0.5, 1.27, 2.44), (4.69, -2.87, -4.96),
                  (3.99, -1.46, 4.44), (-4.07, 2.78, -0.86), (3.23, -2.36, -2.98), (1.69, -2.35, 4.7), (-3.49, -0.13, -0.89)]),
            row(&[(4.62, -2.18, -3.87), (4.1, -0.82, 0.15), (3.88, -2.59, 4.99), (0.8, 3.99, -0.12), (-3.78, 1.26, -0.58),
                  (3.27, -1.56, 0.59), (-0.55, 0.41, -2.35), (-2.7, 3.86, 4.63), (-1.19, -3.46, 2.63), (-1.62, -1.04, 1.81)]),
        ];
        BSplineSurface::new(
            2,
            5,
            poles,
            vec![0.0, 1.0, 2.0, 3.0],
            vec![3, 1, 1, 3],
            vec![0.0, 0.56, 1.0],
            vec![6, 4, 6],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_project_point_close_basins_on_a_folded_net() {
        let s = folded_net();
        let tol = Tolerance::DEFAULT;
        // In patch `[0, 1] x [0, 0.56]`: the foot is the point itself, and a
        // second minimum lies on the edge `u = 0` about 0.08 away.
        let on = s.eval_point(0.05, 0.1);
        let proj = s.project_point(on, tol);
        assert_eq!(proj.distance, 0.0);
        let extrema = s.extrema(on, tol);
        assert_eq!(extrema[0].distance, 0.0);
        assert!(
            extrema[1..]
                .iter()
                .any(|e| e.u == 0.0 && e.distance > 0.05 && e.distance < 0.1),
            "{extrema:?}"
        );
        // Every point of the surface projects onto itself.
        let n = 40;
        for i in 0..=n {
            for j in 0..=n {
                let (u, v) = (3.0 * i as f64 / n as f64, j as f64 / n as f64);
                let on = s.eval_point(u, v);
                let proj = s.project_point(on, tol);
                assert!(proj.distance < 1e-12, "({u}, {v}): {proj:?}");
            }
        }
    }

    #[test]
    fn test_project_point_on_surface_is_zero() {
        let tube = periodic_tube();
        let (u0, u1) = ParametricSurface::u_bounds(&tube);
        let tol = Tolerance::DEFAULT;
        for s in [two_basins(), tube, rational_quarter_cylinder(2.0)] {
            let (u0, u1) = if s.is_u_periodic() {
                (u0, u1)
            } else {
                ParametricSurface::u_bounds(&s)
            };
            let (v0, v1) = ParametricSurface::v_bounds(&s);
            for i in 0..=7 {
                for j in 0..=5 {
                    let (u, v) = (
                        u0 + (u1 - u0) * i as f64 / 7.0,
                        v0 + (v1 - v0) * j as f64 / 5.0,
                    );
                    let on = s.eval_point(u, v);
                    let proj = s.project_point(on, tol);
                    assert_eq!(proj.distance, 0.0);
                    assert!(s.eval_point(proj.u, proj.v).distance(on) < 1e-12);
                }
            }
        }
    }

    /// An exact quarter cylinder of radius `r` about the z axis: a rational
    /// quadratic arc in `u`, extruded over `z` in `[0, 1]` in `v`.
    fn rational_quarter_cylinder(r: f64) -> BSplineSurface {
        let arc = [(r, 0.0), (r, r), (0.0, r)];
        let poles = arc
            .iter()
            .map(|&(x, y)| vec![Point3D::new(x, y, 0.0), Point3D::new(x, y, 1.0)])
            .collect();
        let w = std::f64::consts::FRAC_1_SQRT_2;
        BSplineSurface::new_rational(
            2,
            1,
            poles,
            vec![vec![1.0, 1.0], vec![w, w], vec![1.0, 1.0]],
            vec![0.0, 1.0],
            vec![3, 3],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_project_point_exact_foot_on_rational_cylinder() {
        let radius = 2.0;
        let s = rational_quarter_cylinder(radius);
        let tol = Tolerance::DEFAULT;
        for (rho, theta, z) in [(0.5, 0.3, 0.2), (3.0, 1.1, 0.7), (7.5, 0.05, 0.5)] {
            let query = Point3D::new(rho * f64::cos(theta), rho * f64::sin(theta), z);
            let proj = s.project_point(query, tol);
            assert!((proj.distance - (rho - radius).abs()).abs() < 1e-12);
            let foot = s.eval_point(proj.u, proj.v);
            let expected = Point3D::new(radius * f64::cos(theta), radius * f64::sin(theta), z);
            assert!(foot.distance(expected) < 1e-12);
        }
    }

    #[test]
    fn test_project_point_exact_foot_on_edge_corner_and_crease() {
        let tol = Tolerance::DEFAULT;
        // Bilinear square [0, 2]^2 in z = 0 over [0, 1]^2: an edge foot
        // and a corner foot.
        let s = bilinear_patch();
        let proj = s.project_point(Point3D::new(3.0, 1.0, 1.0), tol);
        assert_eq!((proj.u, proj.v), (1.0, 0.5));
        assert!((proj.distance - 2f64.sqrt()).abs() < 1e-12);
        let proj = s.project_point(Point3D::new(3.0, 3.0, 1.0), tol);
        assert_eq!((proj.u, proj.v), (1.0, 1.0));
        assert!((proj.distance - 3f64.sqrt()).abs() < 1e-12);
        // A ridge: degree 1 across a crease knot, the foot on the crease.
        let ridge = BSplineSurface::new(
            1,
            1,
            vec![
                vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(0.0, 1.0, 0.0)],
                vec![Point3D::new(1.0, 0.0, 1.0), Point3D::new(1.0, 1.0, 1.0)],
                vec![Point3D::new(2.0, 0.0, 0.0), Point3D::new(2.0, 1.0, 0.0)],
            ],
            vec![0.0, 1.0, 2.0],
            vec![2, 1, 2],
            vec![0.0, 1.0],
            vec![2, 2],
            false,
            false,
        )
        .unwrap();
        let proj = ridge.project_point(Point3D::new(1.0, 0.25, 3.0), tol);
        assert_eq!(proj.u, 1.0);
        assert!((proj.v - 0.25).abs() < 1e-12);
        assert!((proj.distance - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_patch_nets_match_the_surface_and_halve() {
        for s in [
            two_basins(),
            periodic_tube(),
            rational_quarter_cylinder(1.5),
        ] {
            let patches = s.patch_tree();
            let mut search = PatchSearch::new(&s, Point3D::ORIGIN, 0.0, true);
            let (rows, cols, h) = (s.u_degree() + 1, s.v_degree() + 1, s.net_stride());
            for k in 0..patches.len() {
                let (rect, _) = patches.patch(k);
                let root = search.root(patches, k).unwrap();
                let net = search.nets[root.slot..root.slot + search.size].to_vec();
                let mut halves = vec![vec![0.0; net.len()]; 2];
                let [left, right] = &mut halves[..] else {
                    unreachable!()
                };
                math::halve_net(&net, rows, cols, h, true, left, right);
                for (a, b) in [(0.0, 0.0), (0.3, 0.8), (1.0, 0.5), (0.7, 1.0)] {
                    let (u, v) = (
                        rect[0] + a * (rect[1] - rect[0]),
                        rect[2] + b * (rect[3] - rect[2]),
                    );
                    let expected = s.eval_point(u, v);
                    let got = bernstein_point(&net, rows, cols, h, a, b);
                    assert!(got.distance(expected) < 1e-12);
                    // The halves along `u` cover `[0, 1/2]` and `[1/2, 1]`.
                    let half = if a < 0.5 {
                        bernstein_point(left, rows, cols, h, 2.0 * a, b)
                    } else {
                        bernstein_point(right, rows, cols, h, 2.0 * a - 1.0, b)
                    };
                    assert!(half.distance(expected) < 1e-12);
                }
            }
        }
    }
}
