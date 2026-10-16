//! Internal B-spline curve evaluation: basis functions with derivatives,
//! rational (NURBS) support, and periodic curves.
//!
//! A [`KnotVector`] holds one parametric direction: its degree, the flat
//! knot sequence (its distinct knots and multiplicities derived from it on
//! first use), and its pole count. Poles are
//! read through a [`Poles`] view: Cartesian points and, for a rational
//! curve, a separate weight per pole.
//!
//! Evaluation computes the `degree + 1` non-zero basis functions and their
//! derivatives at the parameter (The NURBS Book, A2.2) and sums the poles
//! they weight, so its cost depends on the degree only, never on the pole
//! count, and nothing is allocated. A rational curve folds each weight into
//! its basis function, summing the homogeneous point and the weight
//! together, then applies the quotient rule. Periodic curves see one
//! period's flat knots tiled endlessly, each copy shifted by the period,
//! with pole indices wrapping modulo the pole count; tiled knots are
//! computed on demand.

use std::sync::OnceLock;

use super::analytic::in_period;
use crate::{Point3D, Vector3D};

/// Maximum supported B-spline degree.
pub(crate) const MAX_DEGREE: usize = 25;

/// Error type re-exported here for use by [`validate_direction`]; the
/// canonical definition (and its docs) live on [`crate::curves`], next to
/// [`crate::curves::BSplineCurve3D`], the type that actually constructs and
/// reports these errors.
pub(crate) use crate::curves::BSplineConstructionError;

/// Builds the flattened knot sequence: knot `i` repeated `mults[i]` times.
///
/// This is plain repetition with no padding; e.g. `flat_knots(&[0, 1, 2],
/// &[3, 1, 3]) == [0, 0, 0, 1, 2, 2, 2]`.
pub(crate) fn flat_knots(knots: &[f64], mults: &[u32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(mults.iter().map(|&m| m as usize).sum());
    for (&k, &m) in knots.iter().zip(mults.iter()) {
        out.extend(std::iter::repeat_n(k, m as usize));
    }
    out
}

/// The distinct knots of a flat knot sequence, increasing, and their
/// multiplicities.
pub(crate) fn distinct_knots(flat: &[f64]) -> (Vec<f64>, Vec<u32>) {
    flat.chunk_by(|a, b| a == b)
        .map(|run| (run[0], run.len() as u32))
        .unzip()
}

/// One parametric direction of a B-spline: its degree, the flat knot
/// sequence (each distinct knot repeated by its multiplicity), and the
/// pole count they imply.
#[derive(Debug, Clone)]
pub(crate) struct KnotVector {
    degree: usize,
    periodic: bool,
    flat: Vec<f64>,
    /// The distinct knots and their multiplicities, derived from `flat`
    /// when first asked for: editing results seldom need them. Boxed, so
    /// a direction that never derives them stays small.
    distinct: OnceLock<Box<(Vec<f64>, Vec<u32>)>>,
    pole_count: usize,
    /// Multiplicity of the first knot: in a periodic direction, the seam's.
    seam_multiplicity: usize,
}

/// Directions are equal when their degrees, periodicity and flat knots
/// are, whether or not their distinct knots were derived yet.
impl PartialEq for KnotVector {
    fn eq(&self, other: &KnotVector) -> bool {
        (self.degree, self.periodic) == (other.degree, other.periodic) && self.flat == other.flat
    }
}

impl KnotVector {
    /// A direction for `pole_count` poles, validated by
    /// [`validate_direction`].
    pub(crate) fn new(
        degree: usize,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        periodic: bool,
        pole_count: usize,
    ) -> Result<KnotVector, BSplineConstructionError> {
        validate_direction(degree, pole_count, &knots, &multiplicities, periodic)?;
        let flat = flat_knots(&knots, &multiplicities);
        Ok(KnotVector::from_validated(
            degree,
            knots,
            multiplicities,
            flat,
            periodic,
        ))
    }

    /// A direction from parts known to be consistent (as
    /// [`validate_direction`] would find them), `flat` being the flat knots
    /// of `knots` and `multiplicities`: fitting and editing results skip
    /// revalidation.
    pub(crate) fn from_validated(
        degree: usize,
        knots: Vec<f64>,
        multiplicities: Vec<u32>,
        flat: Vec<f64>,
        periodic: bool,
    ) -> KnotVector {
        debug_assert_eq!(flat, flat_knots(&knots, &multiplicities));
        let last = multiplicities[multiplicities.len() - 1] as usize;
        let pole_count = if periodic {
            flat.len() - last
        } else {
            flat.len() - degree - 1
        };
        debug_assert!(
            validate_direction(degree, pole_count, &knots, &multiplicities, periodic).is_ok()
        );
        KnotVector {
            degree,
            periodic,
            seam_multiplicity: multiplicities[0] as usize,
            flat,
            distinct: OnceLock::from(Box::new((knots, multiplicities))),
            pole_count,
        }
    }

    /// A direction from its flat knots, known to be consistent (as
    /// [`validate_direction`] would find their distinct knots and
    /// multiplicities): editing results skip revalidation.
    pub(crate) fn from_flat(degree: usize, flat: Vec<f64>, periodic: bool) -> KnotVector {
        let (first, last) = (flat[0], flat[flat.len() - 1]);
        let seam_multiplicity = flat.iter().take_while(|&&k| k == first).count();
        let last = flat.iter().rev().take_while(|&&k| k == last).count();
        let pole_count = if periodic {
            flat.len() - last
        } else {
            flat.len() - degree - 1
        };
        let direction = KnotVector {
            degree,
            periodic,
            seam_multiplicity,
            flat,
            distinct: OnceLock::new(),
            pole_count,
        };
        debug_assert!(
            validate_direction(
                degree,
                pole_count,
                direction.knots(),
                direction.multiplicities(),
                periodic
            )
            .is_ok()
        );
        direction
    }

    /// The single span `[first, last]` of a Bézier piece of `degree`, both
    /// end knots of multiplicity `degree + 1`.
    pub(crate) fn bezier(degree: usize, first: f64, last: f64) -> KnotVector {
        let mut flat = vec![first; 2 * degree + 2];
        flat[degree + 1..].fill(last);
        KnotVector::from_flat(degree, flat, false)
    }

    /// The degree.
    pub(crate) fn degree(&self) -> usize {
        self.degree
    }

    /// Whether the direction is periodic.
    pub(crate) fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// The distinct knots, increasing.
    pub(crate) fn knots(&self) -> &[f64] {
        &self.distinct().0
    }

    /// The multiplicity of each distinct knot.
    pub(crate) fn multiplicities(&self) -> &[u32] {
        &self.distinct().1
    }

    fn distinct(&self) -> &(Vec<f64>, Vec<u32>) {
        self.distinct
            .get_or_init(|| Box::new(distinct_knots(&self.flat)))
    }

    /// The flat knots: each knot repeated by its multiplicity.
    pub(crate) fn flat(&self) -> &[f64] {
        &self.flat
    }

    /// The number of poles along the direction.
    pub(crate) fn pole_count(&self) -> usize {
        self.pole_count
    }

    /// The parameter range `(first, last)`: the clamped span
    /// `[flat[degree], flat[len - 1 - degree]]`, or a periodic direction's
    /// whole period `[knots[0], knots[last]]`.
    pub(crate) fn bounds(&self) -> (f64, f64) {
        let flat = &self.flat;
        if self.periodic {
            (flat[0], flat[flat.len() - 1])
        } else {
            (flat[self.degree], flat[flat.len() - 1 - self.degree])
        }
    }

    /// The period of a periodic direction.
    pub(crate) fn period(&self) -> Option<f64> {
        let (first, last) = self.bounds();
        self.periodic.then_some(last - first)
    }

    /// Locates `u` and writes the knots around its span relative to it,
    /// `left[j] = u - U[span + 1 - j]` and `right[j] = U[span + j] - u` for
    /// `j` in `1..=degree`; returns the index of the span's first pole.
    ///
    /// A non-periodic direction clamps `u` into its range. A periodic one
    /// wraps it into the period and behaves as if one period's flat knots
    /// (all but the trailing seam run) were tiled endlessly, each copy
    /// shifted by the period; the tiled knot at any index is computed on
    /// demand, so nothing is allocated. A `u` exactly on an interior knot
    /// belongs to the span on its right, and the range end to the last
    /// nonzero-width span; this only matters for derivatives across a knot,
    /// where they are discontinuous.
    #[inline]
    pub(crate) fn span_knots<const N: usize>(
        &self,
        u: f64,
        left: &mut [f64; N],
        right: &mut [f64; N],
    ) -> usize {
        let degree = self.degree;
        debug_assert!(degree < N, "basis table too narrow for degree {degree}");
        if self.periodic {
            return self.periodic_span_knots(u, left, right);
        }
        let (span, u) = self.span(u);
        let flat = &self.flat;
        for j in 1..=degree {
            left[j] = u - flat[span + 1 - j];
            right[j] = flat[span + j] - u;
        }
        span - degree
    }

    /// The index `s` of the flat knot opening the nonzero-width span that
    /// holds `u` in a non-periodic direction, `flat[s] <= u < flat[s + 1]`
    /// (`<=` at the range end), and `u` clamped into the range.
    #[inline(always)]
    fn span(&self, u: f64) -> (usize, f64) {
        let flat = &self.flat;
        let lo = self.degree;
        let hi = flat.len() - 1 - self.degree;
        let u = u.clamp(flat[lo], flat[hi]);
        // Binary search for the rightmost lo <= s < hi with flat[s] <= u
        // (flat[lo] <= u always holds), without data-dependent branches.
        let mut span = lo;
        let mut length = hi - lo;
        while length > 1 {
            let half = length / 2;
            if flat[span + half] <= u {
                span += half;
            }
            length -= half;
        }
        // The rightmost such `s` already opens a nonzero-width span, except
        // at the range end (`u == last`), which steps back to the last real
        // span.
        while span > lo && flat[span] >= flat[span + 1] {
            span -= 1;
        }
        (span, u)
    }

    /// The first pole and the non-zero basis functions at `u` of a
    /// non-periodic direction of degree 1 to 3, unrolled, or `None` for
    /// another direction.
    #[inline(always)]
    pub(crate) fn low_degree_basis(&self, u: f64) -> Option<(usize, [f64; 4])> {
        if self.periodic {
            return None;
        }
        match self.degree {
            1 => Some(self.basis_of_degree::<1>(u)),
            2 => Some(self.basis_of_degree::<2>(u)),
            3 => Some(self.basis_of_degree::<3>(u)),
            _ => None,
        }
    }

    /// [`KnotVector::basis`] of a non-periodic direction of the constant
    /// degree `P` (at most 3): the span's knots sliced once and every loop
    /// unrolled.
    #[inline(always)]
    fn basis_of_degree<const P: usize>(&self, u: f64) -> (usize, [f64; 4]) {
        let (span, u) = self.span(u);
        // The 2P knots around the span, `around[k] = flat[span + 1 - P + k]`.
        let around = &self.flat[span + 1 - P..span + 1 + P];
        // Built as values rather than written in a loop, so that they stay
        // in registers: stored one by one, the unrolled recurrence would
        // reload them in pairs, which waits on the stores.
        let left: [f64; 4] = std::array::from_fn(|j| {
            if (1..=P).contains(&j) {
                u - around[P - j]
            } else {
                0.0
            }
        });
        let right: [f64; 4] = std::array::from_fn(|j| {
            if (1..=P).contains(&j) {
                around[P - 1 + j] - u
            } else {
                0.0
            }
        });
        (span - P, basis_values_of::<P>(&left, &right))
    }

    /// Index of the first of the `degree + 1` poles weighting the span
    /// containing `u`, wrapping modulo the pole count in a periodic
    /// direction (see [`KnotVector::span_knots`]).
    pub(crate) fn span_first_pole(&self, u: f64) -> usize {
        let (mut left, mut right) = ([0.0; LARGE], [0.0; LARGE]);
        self.span_knots(u, &mut left, &mut right)
    }

    /// [`KnotVector::span_knots`] on a periodic direction.
    fn periodic_span_knots<const N: usize>(
        &self,
        u: f64,
        left: &mut [f64; N],
        right: &mut [f64; N],
    ) -> usize {
        let (first, last) = self.bounds();
        let period = last - first;
        // Exactly `last` closes the last span instead of wrapping to the
        // first (they differ only in derivatives across the seam).
        let u = if u == last {
            last
        } else {
            in_period(u, first, last)
        };
        // One period's flat knots: all but the trailing seam run, as many
        // as the poles.
        let block = &self.flat[..self.pole_count];
        // Rightmost knot of the period block at or below `u`; the next tiled
        // knot is then strictly above it, so the span has nonzero width.
        let local = block
            .partition_point(|&k| k <= u)
            .saturating_sub(1)
            .min(block.len() - 1);
        let len = block.len() as isize;
        let knot = |offset: isize| {
            let k = local as isize + offset;
            block[k.rem_euclid(len) as usize] + period * k.div_euclid(len) as f64
        };
        for j in 1..=self.degree {
            left[j] = u - knot(1 - j as isize);
            right[j] = knot(j as isize) - u;
        }
        // Periodic convention: pole 0 starts the span opening at the *last*
        // copy of the first knot, so a seam multiplicity `m` shifts the pole
        // index back by `m - 1`.
        (local + self.pole_count + 1 - self.seam_multiplicity) % self.pole_count
    }

    /// Locates `u` and evaluates into `w` the non-zero basis functions
    /// there, with derivatives up to `order`, touching only the
    /// `2 * degree` knots around the span.
    pub(crate) fn window<const N: usize>(&self, u: f64, order: usize, w: &mut Window<N>) {
        let mut left = [0.0f64; N];
        let mut right = [0.0f64; N];
        w.pole_count = self.pole_count;
        w.periodic = self.periodic;
        w.first = self.span_knots(u, &mut left, &mut right);
        match order {
            0 => basis_values(self.degree, &left, &right, &mut w.ders[0]),
            1 => basis_first(self.degree, &left, &right, &mut w.ders),
            _ => basis_second(self.degree, &left, &right, &mut w.ders),
        }
    }

    /// Locates `u` and writes into `values[..=degree]` the non-zero basis
    /// functions there, without derivatives; returns the index of the pole
    /// the first one weights (see [`KnotVector::pole_index`]).
    #[inline]
    pub(crate) fn basis<const N: usize>(&self, u: f64, values: &mut [f64; N]) -> usize {
        let mut left = [0.0f64; N];
        let mut right = [0.0f64; N];
        let first = self.span_knots(u, &mut left, &mut right);
        basis_values(self.degree, &left, &right, values);
        first
    }

    /// Index of the pole weighted by the `j`-th non-zero basis function of
    /// a span whose first pole is `first`, wrapped in a periodic direction.
    #[inline]
    pub(crate) fn pole_index(&self, first: usize, j: usize) -> usize {
        wrap(first + j, self.pole_count, self.periodic)
    }

    /// Whether the poles of a span whose first pole is `first` run past the
    /// last pole, wrapping to the first ones (periodic directions only).
    #[inline]
    pub(crate) fn wraps(&self, first: usize) -> bool {
        first + self.degree >= self.pole_count
    }

    /// The bounds, then the knots strictly inside them, then the upper
    /// bound: the boundaries of the knot spans (polynomial pieces).
    pub(crate) fn breaks(&self) -> Vec<f64> {
        let (lo, hi) = self.bounds();
        let mut breaks = vec![lo];
        breaks.extend(self.knots().iter().copied().filter(|&k| k > lo && k < hi));
        breaks.push(hi);
        breaks
    }

    /// Whether the direction fails to be C1 at each of `breaks` (from
    /// [`KnotVector::breaks`]): an end of a non-periodic direction, a crease
    /// (a knot of multiplicity `>= degree`), or a periodic seam that is a
    /// crease. A minimum of a smooth function there need not be a critical
    /// point.
    pub(crate) fn crease_breaks(&self, breaks: &[f64]) -> Vec<bool> {
        let last = breaks.len() - 1;
        let creased = |m: u32| m as usize >= self.degree;
        breaks
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i == 0 || i == last {
                    !self.periodic || creased(self.multiplicities()[0])
                } else {
                    self.knots()
                        .iter()
                        .position(|&k| k == x)
                        .is_some_and(|k| creased(self.multiplicities()[k]))
                }
            })
            .collect()
    }

    /// The parameter intervals on which the direction is at least C1, so
    /// that a local refinement bounded to one of them minimises a smooth
    /// function: the range cut at every knot of multiplicity `>= degree` (a
    /// C0 crease). `None` for a periodic direction smooth everywhere, where
    /// refinement may run unbounded. On a periodic direction with a smooth
    /// seam, the interval across the seam runs from the last crease past the
    /// upper bound to the first crease plus the period.
    pub(crate) fn smooth_regions(&self) -> Option<Vec<(f64, f64)>> {
        let (lo, hi) = self.bounds();
        let creases: Vec<f64> = self
            .knots()
            .iter()
            .zip(self.multiplicities())
            .filter(|&(&k, &m)| k > lo && k < hi && m as usize >= self.degree)
            .map(|(&k, _)| k)
            .collect();
        let smooth_seam = self.periodic && self.seam_multiplicity < self.degree;
        if smooth_seam {
            if creases.is_empty() {
                return None;
            }
            let mut regions: Vec<(f64, f64)> = creases.windows(2).map(|w| (w[0], w[1])).collect();
            regions.push((creases[creases.len() - 1], creases[0] + (hi - lo)));
            return Some(regions);
        }
        let mut cuts = vec![lo];
        cuts.extend(creases);
        cuts.push(hi);
        Some(cuts.windows(2).map(|w| (w[0], w[1])).collect())
    }

    /// The [`KnotVector::bezier_matrix`] of every span between consecutive
    /// `breaks` (from [`KnotVector::breaks`]): the first pole index of each
    /// span, and the matrices row-major, `(degree + 1)^2` values per span.
    pub(crate) fn span_bezier_matrices(&self, breaks: &[f64]) -> (Vec<usize>, Vec<f64>) {
        fn fill<const N: usize>(knots: &KnotVector, breaks: &[f64]) -> (Vec<usize>, Vec<f64>) {
            let n = knots.degree + 1;
            let mut firsts = Vec::with_capacity(breaks.len() - 1);
            let mut matrices = Vec::with_capacity((breaks.len() - 1) * n * n);
            let mut matrix = [[0.0f64; N]; N];
            for w in breaks.windows(2) {
                firsts.push(knots.bezier_matrix(0.5 * (w[0] + w[1]), &mut matrix));
                for row in &matrix[..n] {
                    matrices.extend_from_slice(&row[..n]);
                }
            }
            (firsts, matrices)
        }
        if self.degree < SMALL {
            fill::<SMALL>(self, breaks)
        } else {
            fill::<LARGE>(self, breaks)
        }
    }

    /// The matrix taking the `degree + 1` poles of the span containing `u`
    /// (pass a parameter inside the span) to its Bernstein control points
    /// over its own interval: point `k` is `sum_j matrix[k][j] * P[first +
    /// j]`, the pole index wrapping modulo the pole count in a periodic
    /// direction. Returns `first`. Rows are convex weights, so the map is
    /// stable.
    ///
    /// Point `k` is the blossom of the piece at `k` copies of the span end
    /// and `degree - k` of its start, by de Boor's recurrence with the
    /// argument varying by level, carried out on the poles' coefficients.
    #[allow(clippy::needless_range_loop)] // index arithmetic on the triangular recurrence
    pub(crate) fn bezier_matrix<const N: usize>(
        &self,
        u: f64,
        matrix: &mut [[f64; N]; N],
    ) -> usize {
        let p = self.degree;
        let mut left = [0.0f64; N];
        let mut right = [0.0f64; N];
        let first = self.span_knots(u, &mut left, &mut right);
        // The span's `2p` surrounding knots relative to `u`, the span itself
        // being `[knot(p - 1), knot(p)]`.
        let knot = |m: usize| {
            if m < p {
                -left[p - m]
            } else {
                right[m + 1 - p]
            }
        };
        let (a, b) = (knot(p - 1), knot(p));
        // Level `r`, index `j` blends between `knot(j - 1)` and
        // `knot(p + j - r)`.
        let mut inverse = [[0.0f64; N]; N];
        for r in 1..=p {
            for j in r..=p {
                inverse[r][j] = 1.0 / (knot(p + j - r) - knot(j - 1));
            }
        }
        // `d[j][i]`: coefficient of pole `i` in the level's point `j`,
        // nonzero for `i` in `j - r..=j` at level `r`.
        let mut d = [[0.0f64; N]; N];
        for k in 0..=p {
            for (j, dj) in d.iter_mut().enumerate().take(p + 1) {
                dj[..=p].fill(0.0);
                dj[j] = 1.0;
            }
            for r in 1..=p {
                let x = if r <= k { b } else { a };
                for j in (r..=p).rev() {
                    let alpha = (x - knot(j - 1)) * inverse[r][j];
                    for i in j - r..=j {
                        d[j][i] = (1.0 - alpha) * d[j - 1][i] + alpha * d[j][i];
                    }
                }
            }
            matrix[k][..=p].copy_from_slice(&d[p][..=p]);
        }
        first
    }
}

/// The poles of a B-spline, read in place: Cartesian points and, if
/// rational, one weight per point. A surface's poles are row-major, rows
/// along `u`.
#[derive(Clone, Copy)]
pub(crate) struct Poles<'a> {
    pub(crate) points: &'a [Point3D],
    pub(crate) weights: Option<&'a [f64]>,
}

impl Poles<'_> {
    /// Homogeneous coordinates `(x w, y w, z w, w)` of pole `i` (`w = 1` if
    /// nonrational).
    #[inline]
    pub(crate) fn homogeneous(&self, i: usize) -> [f64; 4] {
        let p = self.points[i];
        match self.weights {
            Some(weights) => {
                let w = weights[i];
                [p.x * w, p.y * w, p.z * w, w]
            }
            None => [p.x, p.y, p.z, 1.0],
        }
    }
}

/// Highest derivative order the evaluator computes.
pub(crate) const MAX_DERIVATIVE: usize = 2;

/// Basis-table width for degrees below 8: covers every curve and surface
/// seen in practice, and zeroing an 8-wide table is far cheaper than a
/// `MAX_DEGREE + 1`-wide one.
pub(crate) const SMALL: usize = 8;

/// Basis-table width for every supported degree.
pub(crate) const LARGE: usize = MAX_DEGREE + 1;

/// The `degree + 1` basis functions that are non-zero at a parameter, with
/// their derivatives, and the poles they weight.
pub(crate) struct Window<const N: usize> {
    /// `ders[k][j]`: `k`-th derivative of the `j`-th non-zero basis function.
    pub(crate) ders: [[f64; N]; MAX_DERIVATIVE + 1],
    /// Index of the pole weighted by the first non-zero basis function.
    first: usize,
    /// Pole count, by which indices wrap in a periodic direction.
    pole_count: usize,
    /// Whether pole indices wrap.
    periodic: bool,
}

impl<const N: usize> Window<N> {
    /// An empty window, filled by [`KnotVector::window`].
    pub(crate) fn new() -> Self {
        Window {
            ders: [[0.0; N]; MAX_DERIVATIVE + 1],
            first: 0,
            pole_count: 0,
            periodic: false,
        }
    }

    /// Index of the pole weighted by the `j`-th non-zero basis function,
    /// wrapped modulo the pole count in a periodic direction.
    #[inline]
    pub(crate) fn pole(&self, j: usize) -> usize {
        wrap(self.first + j, self.pole_count, self.periodic)
    }
}

/// Pole `index` of a direction with `pole_count` poles, wrapped if periodic.
#[inline]
fn wrap(index: usize, pole_count: usize, periodic: bool) -> usize {
    if periodic && index >= pole_count {
        index % pole_count
    } else {
        index
    }
}

/// The non-zero basis functions at `u` on the non-periodic span `span`
/// (`flat[span] <= u <= flat[span + 1]`, a nonzero-width span), into
/// `values[..=degree]`; `left` and `right` are scratch.
pub(crate) fn basis_at<const N: usize>(
    u: f64,
    degree: usize,
    span: usize,
    flat: &[f64],
    left: &mut [f64; N],
    right: &mut [f64; N],
    values: &mut [f64; N],
) {
    for j in 1..=degree {
        left[j] = u - flat[span + 1 - j];
        right[j] = flat[span + j] - u;
    }
    basis_values(degree, left, right, values);
}

/// Non-zero basis functions alone (The NURBS Book, A2.2): the value-only
/// fast path of `basis_derivatives` (the tests' reference), without its
/// triangular table.
fn basis_values<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    out: &mut [f64; N],
) {
    out[0] = 1.0;
    for j in 1..=degree {
        let mut saved = 0.0;
        for r in 0..j {
            // The reciprocal does not depend on the recurrence, so its
            // division runs ahead of it.
            let temp = out[r] * (1.0 / (right[r + 1] + left[j - r]));
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
}

/// Non-zero basis functions and their first derivatives: A2.2, whose last
/// step divides each degree-`(p - 1)` function by exactly the knot span its
/// derivative needs, `N'_j = p * (temp_{j-1} - temp_j)`.
fn basis_first<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    ders: &mut [[f64; N]; MAX_DERIVATIVE + 1],
) {
    let p = degree;
    let [values, firsts, _] = ders;
    values[0] = 1.0;
    for j in 1..p {
        let mut saved = 0.0;
        for r in 0..j {
            let temp = values[r] * (1.0 / (right[r + 1] + left[j - r]));
            values[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        values[j] = saved;
    }
    let mut saved = 0.0;
    let mut previous = 0.0;
    for r in 0..p {
        let temp = values[r] * (1.0 / (right[r + 1] + left[p - r]));
        values[r] = saved + right[r + 1] * temp;
        saved = left[p - r] * temp;
        firsts[r] = p as f64 * (previous - temp);
        previous = temp;
    }
    values[p] = saved;
    firsts[p] = p as f64 * previous;
}

/// Non-zero basis functions with their first and second derivatives: the
/// recurrence of [`basis_first`] carried one level further. The level
/// building degree `p - 1` gives that degree's first derivatives, and the
/// level building degree `p` divides them by the same knot spans as the
/// values, `N''_j = p * (D_{j-1} / s_{j-1} - D_j / s_j)` with `D` the
/// first derivatives of degree `p - 1` and `s` the level's spans.
fn basis_second<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    ders: &mut [[f64; N]; MAX_DERIVATIVE + 1],
) {
    let p = degree;
    let [values, firsts, seconds] = ders;
    // First derivatives of degree p - 1 (zero for degree 0).
    let mut lower = [0.0f64; N];
    values[0] = 1.0;
    for j in 1..p.saturating_sub(1) {
        let mut saved = 0.0;
        for r in 0..j {
            let temp = values[r] * (1.0 / (right[r + 1] + left[j - r]));
            values[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        values[j] = saved;
    }
    if p >= 2 {
        let q = p - 1;
        let mut saved = 0.0;
        let mut previous = 0.0;
        for r in 0..q {
            let temp = values[r] * (1.0 / (right[r + 1] + left[q - r]));
            values[r] = saved + right[r + 1] * temp;
            saved = left[q - r] * temp;
            lower[r] = q as f64 * (previous - temp);
            previous = temp;
        }
        values[q] = saved;
        lower[q] = q as f64 * previous;
    }
    let mut saved = 0.0;
    let mut previous = 0.0;
    let mut previous_lower = 0.0;
    for r in 0..p {
        let inverse = 1.0 / (right[r + 1] + left[p - r]);
        let temp = values[r] * inverse;
        values[r] = saved + right[r + 1] * temp;
        saved = left[p - r] * temp;
        firsts[r] = p as f64 * (previous - temp);
        previous = temp;
        let lower_temp = lower[r] * inverse;
        seconds[r] = p as f64 * (previous_lower - lower_temp);
        previous_lower = lower_temp;
    }
    values[p] = saved;
    firsts[p] = p as f64 * previous;
    seconds[p] = p as f64 * previous_lower;
}

/// Non-zero basis functions and their derivatives up to `order` (The NURBS
/// Book, A2.3), from `left[j] = u - U[span + 1 - j]` and
/// `right[j] = U[span + j] - u` for `j` in `1..=degree`.
///
/// Every divisor is a difference of two knots on either side of a
/// nonzero-width span, so none is zero. Derivatives above the degree vanish
/// and are left at zero.
#[cfg(test)]
fn basis_derivatives<const N: usize>(
    degree: usize,
    left: &[f64; N],
    right: &[f64; N],
    order: usize,
    ders: &mut [[f64; N]; MAX_DERIVATIVE + 1],
) {
    let p = degree;
    // ndu: basis functions in the upper triangle, knot differences in the
    // lower one.
    let mut ndu = [[0.0f64; N]; N];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }
    let order = order.min(p).min(MAX_DERIVATIVE);
    if order == 0 {
        return;
    }
    let mut a = [[0.0f64; MAX_DERIVATIVE + 1]; 2];
    for r in 0..=p {
        let (mut s1, mut s2) = (0, 1);
        a[0][0] = 1.0;
        for k in 1..=order {
            let mut d = 0.0;
            let pk = p - k;
            if r >= k {
                let rk = r - k;
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            let j1 = if r + 1 >= k { 1 } else { k - r };
            let j2 = if r <= pk + 1 { k - 1 } else { p - r };
            for j in j1..=j2 {
                // r + j >= k holds for every j in j1..=j2.
                let rkj = r + j - k;
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rkj];
                d += a[s2][j] * ndu[rkj][pk];
            }
            if r <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }
    let mut factor = p as f64;
    for (k, row) in ders.iter_mut().enumerate().take(order + 1).skip(1) {
        for value in row.iter_mut().take(p + 1) {
            *value *= factor;
        }
        factor *= (p - k) as f64;
    }
}

/// The point at `u`: the value-only path of [`derivatives`].
#[inline]
pub(crate) fn point(knots: &KnotVector, poles: Poles, u: f64) -> Point3D {
    if !knots.periodic {
        match knots.degree {
            1 => return point_of_degree::<1>(knots, poles, u),
            2 => return point_of_degree::<2>(knots, poles, u),
            3 => return point_of_degree::<3>(knots, poles, u),
            _ => {}
        }
    }
    if knots.degree < SMALL {
        point_sized::<SMALL>(knots, poles, u)
    } else {
        point_sized::<LARGE>(knots, poles, u)
    }
}

/// [`point`] of a non-periodic direction of degree `P` (at most 3): the
/// span's poles sliced once and every loop over constant bounds, so
/// unrolled.
#[inline(always)]
fn point_of_degree<const P: usize>(knots: &KnotVector, poles: Poles, u: f64) -> Point3D {
    let (first, values) = knots.basis_of_degree::<P>(u);
    let points = &poles.points[first..=first + P];
    // Each sum starts from its first term, sparing additions of zero.
    match poles.weights {
        None => {
            let mut sum = [
                values[0] * points[0].x,
                values[0] * points[0].y,
                values[0] * points[0].z,
            ];
            for j in 1..=P {
                sum[0] += values[j] * points[j].x;
                sum[1] += values[j] * points[j].y;
                sum[2] += values[j] * points[j].z;
            }
            Point3D::new(sum[0], sum[1], sum[2])
        }
        Some(weights) => {
            let weights = &weights[first..=first + P];
            let b = values[0] * weights[0];
            let mut sum = [b * points[0].x, b * points[0].y, b * points[0].z, b];
            for j in 1..=P {
                let b = values[j] * weights[j];
                sum[0] += b * points[j].x;
                sum[1] += b * points[j].y;
                sum[2] += b * points[j].z;
                sum[3] += b;
            }
            let inverse = 1.0 / sum[3];
            Point3D::new(sum[0] * inverse, sum[1] * inverse, sum[2] * inverse)
        }
    }
}

/// The recurrence of [`basis_values`] for the constant degree `P` (at
/// most 3).
#[inline(always)]
fn basis_values_of<const P: usize>(left: &[f64; 4], right: &[f64; 4]) -> [f64; 4] {
    let mut out = [1.0, 0.0, 0.0, 0.0];
    for j in 1..=P {
        let mut saved = 0.0;
        // Bounded by `P` rather than `j`, so that the loop's bounds are
        // constant and it unrolls.
        for r in 0..P {
            if r < j {
                let temp = out[r] * (1.0 / (right[r + 1] + left[j - r]));
                out[r] = if r == 0 {
                    right[1] * temp
                } else {
                    saved + right[r + 1] * temp
                };
                saved = left[j - r] * temp;
            }
        }
        out[j] = saved;
    }
    out
}

#[inline]
fn point_sized<const N: usize>(knots: &KnotVector, poles: Poles, u: f64) -> Point3D {
    let mut values = [0.0f64; N];
    let first = knots.basis(u, &mut values);
    let degree = knots.degree;
    let mut sum = [0.0f64; 4];
    match poles.weights {
        None => {
            for (j, &b) in values.iter().enumerate().take(degree + 1) {
                let p = poles.points[knots.pole_index(first, j)];
                sum[0] += b * p.x;
                sum[1] += b * p.y;
                sum[2] += b * p.z;
            }
            Point3D::new(sum[0], sum[1], sum[2])
        }
        Some(weights) => {
            for (j, &b) in values.iter().enumerate().take(degree + 1) {
                let i = knots.pole_index(first, j);
                let (p, b) = (poles.points[i], b * weights[i]);
                sum[0] += b * p.x;
                sum[1] += b * p.y;
                sum[2] += b * p.z;
                sum[3] += b;
            }
            let inverse = 1.0 / sum[3];
            Point3D::new(sum[0] * inverse, sum[1] * inverse, sum[2] * inverse)
        }
    }
}

/// The point (as a vector from the origin) and its derivatives up to
/// `ORDER` (`<=` [`MAX_DERIVATIVE`]) at `u`, `[C, C', C'']`, those beyond
/// `ORDER` zero.
///
/// A rational curve sums the homogeneous point `H = (x w, y w, z w, w)` and
/// its derivatives, then applies the quotient rule: `C = H / w`,
/// `C' = (H' - C w') / w`, `C'' = (H'' - 2 C' w' - C w'') / w`.
#[inline]
pub(crate) fn derivatives<const ORDER: usize>(
    knots: &KnotVector,
    poles: Poles,
    u: f64,
) -> [Vector3D; 3] {
    if knots.degree < SMALL {
        derivatives_sized::<SMALL, ORDER>(knots, poles, u)
    } else {
        derivatives_sized::<LARGE, ORDER>(knots, poles, u)
    }
}

#[inline]
fn derivatives_sized<const N: usize, const ORDER: usize>(
    knots: &KnotVector,
    poles: Poles,
    u: f64,
) -> [Vector3D; 3] {
    debug_assert!(
        ORDER <= MAX_DERIVATIVE,
        "derivative order {ORDER} not supported"
    );
    let mut w = Window::<N>::new();
    knots.window(u, ORDER, &mut w);
    let mut sums = [[0.0f64; 4]; MAX_DERIVATIVE + 1];
    for j in 0..=knots.degree {
        let i = w.pole(j);
        let p = poles.points[i];
        let weight = poles.weights.map_or(1.0, |weights| weights[i]);
        for (k, sum) in sums.iter_mut().enumerate().take(ORDER + 1) {
            let b = w.ders[k][j] * weight;
            sum[0] += b * p.x;
            sum[1] += b * p.y;
            sum[2] += b * p.z;
            sum[3] += b;
        }
    }
    let vector = |s: [f64; 4]| Vector3D::new(s[0], s[1], s[2]);
    let mut out = sums.map(vector);
    if poles.weights.is_some() {
        let [w0, w1, w2] = sums.map(|s| s[3]);
        let inverse = 1.0 / w0;
        out[0] = out[0] * inverse;
        if ORDER >= 1 {
            out[1] = (out[1] - w1 * out[0]) * inverse;
        }
        if ORDER >= 2 {
            out[2] = (out[2] - 2.0 * w1 * out[1] - w2 * out[0]) * inverse;
        }
    }
    out
}

/// The bounds for refining from `t` (within `[lo, lo + period]`): each
/// region of [`KnotVector::smooth_regions`] containing `t` (two at a
/// crease), with `t` shifted by the period into those crossing a periodic
/// seam; a single unbounded entry when `regions` is `None`.
pub(crate) fn regions_at(
    regions: &Option<Vec<(f64, f64)>>,
    t: f64,
    period: Option<f64>,
) -> Vec<(Option<(f64, f64)>, f64)> {
    let Some(regions) = regions else {
        return vec![(None, t)];
    };
    let mut out = Vec::new();
    for &(a, b) in regions {
        let shifted = std::iter::once(t).chain(period.map(|p| t + p));
        if let Some(x) = shifted.into_iter().find(|&x| a <= x && x <= b) {
            out.push((Some((a, b)), x));
        }
    }
    out
}

/// The parameter at which to evaluate derivatives for a refinement
/// bounded to `bounds`: `x` itself, except exactly at the upper bound,
/// where the next float down selects the span on the left (evaluation
/// takes the span on the right of a knot, which beyond a crease is not
/// the region being refined).
pub(crate) fn inward(x: f64, bounds: Option<(f64, f64)>) -> f64 {
    match bounds {
        Some((_, hi)) if x >= hi => x.next_down(),
        _ => x,
    }
}

/// Bounding box `[min.., max..]` of the `degree + 1` poles of every span
/// between consecutive `breaks` (each span lies in it, by the convex hull
/// property; weights are positive).
pub(crate) fn span_boxes(knots: &KnotVector, poles: Poles, breaks: &[f64]) -> Vec<[f64; 6]> {
    fn boxes<const N: usize>(knots: &KnotVector, poles: Poles, breaks: &[f64]) -> Vec<[f64; 6]> {
        let (mut left, mut right) = ([0.0f64; N], [0.0f64; N]);
        breaks
            .windows(2)
            .map(|span| {
                // The midpoint's span is exactly this one.
                let first = knots.span_knots(0.5 * (span[0] + span[1]), &mut left, &mut right);
                let mut b = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for j in 0..=knots.degree {
                    let p = poles.points[wrap(first + j, knots.pole_count, knots.periodic)];
                    for (c, x) in [p.x, p.y, p.z].into_iter().enumerate() {
                        b[c] = b[c].min(x);
                        b[c + 3] = b[c + 3].max(x);
                    }
                }
                b
            })
            .collect()
    }
    if knots.degree < SMALL {
        boxes::<SMALL>(knots, poles, breaks)
    } else {
        boxes::<LARGE>(knots, poles, breaks)
    }
}

/// Homogeneous Bernstein control points `(x w, y w, z w, w)` (`w = 1` if
/// nonrational), over its own parameter interval, of the polynomial piece
/// on the span containing `u` (pass a parameter inside the span), into
/// `out[..=degree]`: the span's poles mapped by
/// [`KnotVector::bezier_matrix`].
pub(crate) fn span_bezier(knots: &KnotVector, poles: Poles, u: f64, out: &mut [[f64; 4]]) {
    fn bezier<const N: usize>(knots: &KnotVector, poles: Poles, u: f64, out: &mut [[f64; 4]]) {
        let degree = knots.degree;
        let mut matrix = [[0.0f64; N]; N];
        let first = knots.bezier_matrix(u, &mut matrix);
        let mut local = [[0.0f64; 4]; N];
        for (j, pole) in local.iter_mut().enumerate().take(degree + 1) {
            *pole = poles.homogeneous(wrap(first + j, knots.pole_count, knots.periodic));
        }
        for (point, row) in out.iter_mut().zip(&matrix).take(degree + 1) {
            *point = [0.0; 4];
            for (&m, pole) in row.iter().zip(&local).take(degree + 1) {
                for c in 0..4 {
                    point[c] += m * pole[c];
                }
            }
        }
    }
    if knots.degree < SMALL {
        bezier::<SMALL>(knots, poles, u, out);
    } else {
        bezier::<LARGE>(knots, poles, u, out);
    }
}

/// Validates a B-spline direction (shared by curves and surface directions).
///
/// Checks degree bounds, distinct knot count, strict monotonicity,
/// multiplicity bounds, periodic end-multiplicity match, and the pole-count
/// relation. Does *not* validate weights (see [`BSplineConstructionError`]).
pub(crate) fn validate_direction(
    degree: usize,
    n_poles: usize,
    knots: &[f64],
    mults: &[u32],
    periodic: bool,
) -> Result<(), BSplineConstructionError> {
    if !(1..=MAX_DEGREE).contains(&degree) {
        return Err(BSplineConstructionError::InvalidDegree);
    }
    if knots.len() < 2 || mults.len() < 2 || knots.len() != mults.len() {
        return Err(BSplineConstructionError::TooFewKnots);
    }
    for pair in knots.windows(2) {
        if pair[1] <= pair[0] {
            return Err(BSplineConstructionError::KnotsNotIncreasing);
        }
    }

    let last = mults.len() - 1;
    let deg = degree as u32;
    if periodic {
        if mults[0] != mults[last] {
            return Err(BSplineConstructionError::PeriodicEndMultiplicityMismatch);
        }
        if mults.iter().any(|&m| m > deg) {
            return Err(BSplineConstructionError::MultiplicityTooLarge);
        }
    } else {
        if mults[0] > deg + 1 || mults[last] > deg + 1 {
            return Err(BSplineConstructionError::MultiplicityTooLarge);
        }
        if mults[1..last].iter().any(|&m| m > deg) {
            return Err(BSplineConstructionError::MultiplicityTooLarge);
        }
    }

    let sum_mults: u32 = mults.iter().sum();
    let expected = if periodic {
        sum_mults as i64 - mults[last] as i64
    } else {
        sum_mults as i64 - degree as i64 - 1
    };
    if expected < 0 || n_poles as i64 != expected {
        return Err(BSplineConstructionError::PoleCountMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_second_matches_general_derivatives() {
        // Uneven knots, every degree, parameters across a span.
        for p in 1..=9usize {
            let flat: Vec<f64> = (0..2 * p + 2)
                .map(|i| (i as f64).powf(1.3) + 0.1 * (i % 3) as f64)
                .collect();
            let span = p;
            for step in 0..=10 {
                let u = flat[span] + (flat[span + 1] - flat[span]) * step as f64 / 10.0;
                let (mut left, mut right) = ([0.0; LARGE], [0.0; LARGE]);
                for j in 1..=p {
                    left[j] = u - flat[span + 1 - j];
                    right[j] = flat[span + j] - u;
                }
                let mut fast = [[0.0; LARGE]; MAX_DERIVATIVE + 1];
                let mut general = [[0.0; LARGE]; MAX_DERIVATIVE + 1];
                basis_second(p, &left, &right, &mut fast);
                basis_derivatives(p, &left, &right, 2, &mut general);
                for k in 0..=2 {
                    for j in 0..=p {
                        let scale = general[k][..=p].iter().fold(1.0f64, |m, x| m.max(x.abs()));
                        assert!(
                            (fast[k][j] - general[k][j]).abs() <= 1e-13 * scale,
                            "degree {p}, order {k}, function {j}: {} vs {}",
                            fast[k][j],
                            general[k][j]
                        );
                    }
                }
            }
        }
    }

    // ---- flat_knots ----

    #[test]
    fn test_flat_knots_repeats() {
        assert_eq!(
            flat_knots(&[0.0, 1.0, 2.0], &[3, 1, 3]),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]
        );
    }

    #[test]
    fn test_flat_knots_all_single() {
        assert_eq!(
            flat_knots(&[0.0, 1.0, 2.0], &[1, 1, 1]),
            vec![0.0, 1.0, 2.0]
        );
    }

    // ---- helpers over the new API ----

    fn points(coordinates: &[[f64; 3]]) -> Vec<Point3D> {
        coordinates
            .iter()
            .map(|c| Point3D::new(c[0], c[1], c[2]))
            .collect()
    }

    fn knot_vector(
        degree: usize,
        knots: &[f64],
        mults: &[u32],
        periodic: bool,
        count: usize,
    ) -> KnotVector {
        KnotVector::new(degree, knots.to_vec(), mults.to_vec(), periodic, count).unwrap()
    }

    fn components(v: Vector3D) -> [f64; 3] {
        [v.x, v.y, v.z]
    }

    // ---- degree-1 line ----

    #[test]
    fn test_degree1_line_midpoint_and_derivative() {
        // Two poles (0,0,0) and (2,0,0); knots [0,1] mults [2,2] => a straight
        // segment. At u = 0.5 the point is the midpoint (1,0,0); the first
        // derivative is the constant tangent (2,0,0).
        let poles = points(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        let knots = knot_vector(1, &[0.0, 1.0], &[2, 2], false, 2);
        let view = Poles {
            points: &poles,
            weights: None,
        };
        let [c, d1, _] = derivatives::<1>(&knots, view, 0.5);
        assert_close(&components(c), &[1.0, 0.0, 0.0]);
        assert_close(&components(d1), &[2.0, 0.0, 0.0]);
    }

    // ---- degree-2 de Casteljau by hand ----

    #[test]
    fn test_degree2_bezier_by_hand() {
        // Poles (0,0,0),(1,2,0),(2,0,0); knots [0,1] mults [3,3] => a Bezier
        // parabola. de Casteljau at t = 0.5:
        //   lerp(P0,P1)=(0.5,1,0), lerp(P1,P2)=(1.5,1,0),
        //   lerp of those = (1,1,0).
        // Derivative curve poles: 2*(P1-P0)=(2,4,0), 2*(P2-P1)=(2,-4,0);
        //   at t=0.5 the tangent is their midpoint (2,0,0).
        let poles = points(&[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]]);
        let knots = knot_vector(2, &[0.0, 1.0], &[3, 3], false, 3);
        let view = Poles {
            points: &poles,
            weights: None,
        };
        let [c, d1, _] = derivatives::<2>(&knots, view, 0.5);
        assert_close(&components(c), &[1.0, 1.0, 0.0]);
        assert_close(&components(d1), &[2.0, 0.0, 0.0]);
        let p = point(&knots, view, 0.5);
        assert_close(&[p.x, p.y, p.z], &[1.0, 1.0, 0.0]);
    }

    // ---- fixed-degree point path ----

    #[test]
    fn test_point_of_low_degree_matches_general_path() {
        // Uneven knots with an interior double knot; parameters across every
        // span, on the knots, at the ends and outside the range.
        for degree in 1..=3usize {
            let double = degree.min(2) as u32;
            let knots = knot_vector(
                degree,
                &[0.0, 0.3, 0.45, 1.2, 2.0],
                &[degree as u32 + 1, 1, double, 1, degree as u32 + 1],
                false,
                degree + 3 + double as usize,
            );
            let poles: Vec<Point3D> = (0..knots.pole_count)
                .map(|i| {
                    let t = i as f64;
                    Point3D::new(t, (1.7 * t).sin(), 0.3 * t * t)
                })
                .collect();
            let weights: Vec<f64> = (0..knots.pole_count)
                .map(|i| 0.5 + 0.3 * (i % 3) as f64)
                .collect();
            for weights in [None, Some(weights.as_slice())] {
                let view = Poles {
                    points: &poles,
                    weights,
                };
                for step in -2..=42 {
                    let u = step as f64 / 20.0;
                    let p = point(&knots, view, u);
                    let [c, _, _] = derivatives::<0>(&knots, view, u);
                    assert_close(&[p.x, p.y, p.z], &components(c));
                }
            }
        }
    }

    // ---- rational quarter circle ----

    #[test]
    fn test_rational_quarter_circle_on_unit_circle() {
        // Standard rational quadratic quarter circle: poles (1,0),(1,1),(0,1),
        // weights [1, sqrt(2)/2, 1], degree 2, knots [0,1] mults [3,3].
        let w = std::f64::consts::FRAC_1_SQRT_2; // sqrt(2)/2
        let poles = points(&[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        let weights = [1.0, w, 1.0];
        let knots = knot_vector(2, &[0.0, 1.0], &[3, 3], false, 3);
        let view = Poles {
            points: &poles,
            weights: Some(&weights),
        };
        for t in [0.0, 0.25, 0.5, 1.0] {
            let p = point(&knots, view, t);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!((r - 1.0).abs() < 1e-12, "t={t} radius={r}");
            // The tangent of a circle is orthogonal to its radius.
            let [c, d1, _] = derivatives::<1>(&knots, view, t);
            assert_close(&components(c), &[p.x, p.y, p.z]);
            assert!((c.x * d1.x + c.y * d1.y).abs() < 1e-12, "t={t}");
        }
        // Endpoint t = 1 lands exactly on (0,1,0).
        let p = point(&knots, view, 1.0);
        assert_close(&[p.x, p.y, p.z], &[0.0, 1.0, 0.0]);
    }

    // ---- periodic wrap smoke ----

    #[test]
    fn test_periodic_value_repeats_over_period() {
        // Fixture case (c) poles (hexagon ring), degree 3, knots [0..6]
        // mults all 1, period 6. Evaluating at u and u + period must give the
        // same point (and derivatives).
        let poles = periodic_ring_poles();
        let knots = knot_vector(3, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1; 7], true, 6);
        let view = Poles {
            points: &poles,
            weights: None,
        };
        for u in [0.3, 1.7, 4.9] {
            let a = derivatives::<2>(&knots, view, u);
            let b = derivatives::<2>(&knots, view, u + 6.0);
            for k in 0..3 {
                assert_close(&components(a[k]), &components(b[k]));
            }
        }
    }

    // ---- span_bezier ----

    /// The Bernstein form of every span matches the evaluator inside it.
    fn assert_bezier_matches(
        degree: usize,
        periodic: bool,
        knots: &[f64],
        mults: &[u32],
        poles: &[Point3D],
        weights: Option<&[f64]>,
    ) {
        let vector = knot_vector(degree, knots, mults, periodic, poles.len());
        let view = Poles {
            points: poles,
            weights,
        };
        let binomial =
            |n: usize, k: usize| (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64);
        for span in knots.windows(2) {
            let (a, b) = (span[0], span[1]);
            let mut bezier = vec![[0.0; 4]; degree + 1];
            span_bezier(&vector, view, 0.5 * (a + b), &mut bezier);
            for s in [0.0, 0.2, 0.5, 0.9, 1.0] {
                let expected = point(&vector, view, a + s * (b - a));
                let mut sum = [0.0; 4];
                for (k, x) in bezier.iter().enumerate() {
                    let basis = binomial(degree, k)
                        * s.powi(k as i32)
                        * (1.0 - s).powi((degree - k) as i32);
                    for c in 0..4 {
                        sum[c] += basis * x[c];
                    }
                }
                let got = [sum[0] / sum[3], sum[1] / sum[3], sum[2] / sum[3]];
                for (g, e) in got.into_iter().zip([expected.x, expected.y, expected.z]) {
                    assert!((g - e).abs() < 1e-12, "span [{a}, {b}] s {s}: {g} vs {e}");
                }
            }
        }
    }

    #[test]
    fn test_span_bezier_nonuniform_with_interior_multiplicity() {
        let poles: Vec<Point3D> = (0..8)
            .map(|i| Point3D::new(i as f64, (i * i % 5) as f64, (i % 3) as f64 - 1.0))
            .collect();
        assert_bezier_matches(
            3,
            false,
            &[0.0, 0.7, 1.0, 2.5, 4.0],
            &[4, 1, 2, 1, 4],
            &poles,
            None,
        );
    }

    #[test]
    fn test_span_bezier_rational_homogeneous() {
        let weights = [1.0, 0.4, 2.5, 1.2, 0.8, 3.0];
        let poles: Vec<Point3D> = (0..6)
            .map(|i| Point3D::new(i as f64, ((i * 7) % 4) as f64, -(i as f64)))
            .collect();
        assert_bezier_matches(
            2,
            false,
            &[-1.0, 0.5, 1.5, 3.0, 3.5],
            &[3, 1, 1, 1, 3],
            &poles,
            Some(&weights),
        );
    }

    #[test]
    fn test_span_bezier_periodic_with_seam_multiplicity() {
        let poles = periodic_ring_poles();
        assert_bezier_matches(
            3,
            true,
            &[0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
            &[1, 1, 1, 1, 1, 1, 1],
            &poles,
            None,
        );
        assert_bezier_matches(
            3,
            true,
            &[0.0, 1.5, 3.0, 4.0, 5.0, 6.0],
            &[2, 1, 1, 1, 1, 2],
            &poles,
            None,
        );
    }

    fn periodic_ring_poles() -> Vec<Point3D> {
        points(&[
            [2.0, 0.0, 0.3],
            [1.0000000000000002, 1.7320508075688772, -0.3],
            [-0.9999999999999996, 1.7320508075688774, 0.3],
            [-2.0, 2.4492935982947064e-16, -0.3],
            [-1.0000000000000009, -1.7320508075688767, 0.3],
            [1.0000000000000002, -1.7320508075688772, -0.3],
        ])
    }

    // ---- validation: every error variant ----

    #[test]
    fn test_validation_ok_nonperiodic() {
        // clamped cubic, 6 poles, knots [0,1,2,3] mults [4,1,1,4].
        assert_eq!(
            validate_direction(3, 6, &[0.0, 1.0, 2.0, 3.0], &[4, 1, 1, 4], false),
            Ok(())
        );
    }

    #[test]
    fn test_validation_ok_periodic() {
        assert_eq!(
            validate_direction(3, 6, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1; 7], true),
            Ok(())
        );
    }

    #[test]
    fn test_validation_invalid_degree_low() {
        assert_eq!(
            validate_direction(0, 1, &[0.0, 1.0], &[1, 1], false),
            Err(BSplineConstructionError::InvalidDegree)
        );
    }

    #[test]
    fn test_validation_invalid_degree_high() {
        assert_eq!(
            validate_direction(26, 1, &[0.0, 1.0], &[1, 1], false),
            Err(BSplineConstructionError::InvalidDegree)
        );
    }

    #[test]
    fn test_validation_too_few_knots() {
        assert_eq!(
            validate_direction(1, 1, &[0.0], &[2], false),
            Err(BSplineConstructionError::TooFewKnots)
        );
    }

    #[test]
    fn test_validation_knots_not_increasing() {
        assert_eq!(
            validate_direction(1, 2, &[1.0, 1.0], &[2, 2], false),
            Err(BSplineConstructionError::KnotsNotIncreasing)
        );
    }

    #[test]
    fn test_validation_multiplicity_too_large_interior() {
        // Interior mult 3 > degree 2.
        assert_eq!(
            validate_direction(2, 3, &[0.0, 1.0, 2.0], &[3, 3, 3], false),
            Err(BSplineConstructionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_validation_multiplicity_too_large_end() {
        // End mult 5 > degree + 1 = 4 for degree 3.
        assert_eq!(
            validate_direction(3, 3, &[0.0, 1.0], &[5, 3], false),
            Err(BSplineConstructionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_validation_periodic_mult_too_large() {
        // Periodic: any mult > degree is invalid.
        assert_eq!(
            validate_direction(2, 4, &[0.0, 1.0, 2.0, 3.0], &[3, 1, 1, 3], true),
            Err(BSplineConstructionError::MultiplicityTooLarge)
        );
    }

    #[test]
    fn test_validation_periodic_end_mismatch() {
        assert_eq!(
            validate_direction(3, 6, &[0.0, 1.0, 2.0, 3.0], &[2, 1, 1, 3], true),
            Err(BSplineConstructionError::PeriodicEndMultiplicityMismatch)
        );
    }

    #[test]
    fn test_validation_pole_count_mismatch_nonperiodic() {
        // sum(mults) - degree - 1 = 10 - 3 - 1 = 6; give 5.
        assert_eq!(
            validate_direction(3, 5, &[0.0, 1.0, 2.0, 3.0], &[4, 1, 1, 4], false),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
    }

    #[test]
    fn test_validation_pole_count_mismatch_periodic() {
        // sum(mults) - mults[last] = 7 - 1 = 6; give 7.
        assert_eq!(
            validate_direction(3, 7, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1; 7], true),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
    }

    // The weight variants are not raised by validate_direction; assert they
    // exist and format, documenting their intended use by the constructing
    // type.
    #[test]
    fn test_weight_error_variants_display() {
        assert!(
            !BSplineConstructionError::WeightCountMismatch
                .to_string()
                .is_empty()
        );
        assert!(
            !BSplineConstructionError::NonPositiveWeight
                .to_string()
                .is_empty()
        );
    }

    // ---- fixture replay: the correctness gate ----

    #[test]
    fn test_fixture_replay_all_cases() {
        replay_fixture();
    }

    // ---- helpers ----

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-9 * f64::max(1.0, e.abs());
            assert!((a - e).abs() <= tol, "index {i}: {a} vs {e}");
        }
    }

    // --- fixture parsing (minimal, no serde derive to keep it explicit) ---

    use serde_json::Value;

    fn replay_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/curves_bspline.json"
        );
        let text = std::fs::read_to_string(path).expect("read fixture");
        let json: Value = serde_json::from_str(&text).expect("parse fixture");
        let cases = json["cases"].as_array().expect("cases array");
        for case in cases {
            replay_case(case);
        }
    }

    fn arr_f64(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn replay_case(case: &Value) {
        let name = case["name"].as_str().unwrap();
        let degree = case["degree"].as_u64().unwrap() as usize;
        let periodic = case["periodic"].as_bool().unwrap();
        let knots = arr_f64(&case["knots"]);
        let mults: Vec<u32> = case["mults"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let pole_rows: Vec<Vec<f64>> = case["poles"]
            .as_array()
            .unwrap()
            .iter()
            .map(arr_f64)
            .collect();
        let weights: Option<Vec<f64>> = case
            .get("weights")
            .and_then(|w| if w.is_null() { None } else { Some(arr_f64(w)) });

        let poles: Vec<Point3D> = pole_rows
            .iter()
            .map(|p| Point3D::new(p[0], p[1], p[2]))
            .collect();
        let vector = KnotVector::new(degree, knots, mults, periodic, poles.len()).unwrap();
        let view = Poles {
            points: &poles,
            weights: weights.as_deref(),
        };

        for sample in case["samples"].as_array().unwrap() {
            let u = sample["u"].as_f64().unwrap();
            let all = derivatives::<2>(&vector, view, u);
            let p = point(&vector, view, u);
            let euc: Vec<f64> = all.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
            for (c, (&a, b)) in euc[..3].iter().zip([p.x, p.y, p.z]).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-12 * f64::max(1.0, b.abs()),
                    "{name} u={u} point[{c}]"
                );
            }
            for (key, idx) in [("point", 0usize), ("d1", 1), ("d2", 2)] {
                let expected = arr_f64(&sample[key]);
                let got = &euc[idx * 3..idx * 3 + 3];
                for (c, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
                    let tol = 1e-7 * f64::max(1.0, e.abs());
                    assert!(
                        (g - e).abs() <= tol,
                        "{name} u={u} {key}[{c}]: got {g}, expected {e} (tol {tol})"
                    );
                }
            }
        }
    }
}
