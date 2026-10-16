//! Internal B-spline surface evaluation: tensor-product NURBS on top of the
//! curve engine ([`crate::curve_math::bspline`]).
//!
//! A B-spline surface is a tensor product of two directional B-spline bases,
//! one [`KnotVector`] per direction. Poles are read through a [`Poles`]
//! view, row-major `[i_u][i_v]`: `i_u` indexes the `u` direction (rows) and
//! `i_v` the `v` direction (columns), with a separate weight per pole if
//! rational.
//!
//! Evaluation is a direct tensor product: [`KnotVector::window`] gives the
//! non-zero basis functions (and derivatives) in each direction with the
//! pole indices they weight, already wrapped for periodic directions, and
//! only the `(u_degree + 1) * (v_degree + 1)` poles they select are summed.
//! The cost depends on the degrees only, never on the pole counts, and
//! nothing is allocated.
//!
//! A rational surface folds each weight into the product of its basis
//! functions, summing the homogeneous point `(x w, y w, z w, w)` and its
//! derivatives, and applies the quotient rule at the very end.
//!
//! Bernstein nets of patches are row-major `[k_u][k_v][h]`, with `h = 4`
//! homogeneous coordinates per point if rational, else `h = 3`.

use super::super::curve_math::bspline as curve;
use crate::curve_math::bspline::{KnotVector, Poles};
use crate::curves::Side;
use crate::{Point3D, Vector3D};

/// The surface point `S(u, v)`: the value-only path of
/// [`surface_derivatives`].
#[inline]
pub(crate) fn surface_point(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u: f64,
    v: f64,
) -> Point3D {
    if let (Some((first_u, u_values)), Some((first_v, v_values))) =
        (u_knots.low_degree_basis(u), v_knots.low_degree_basis(v))
    {
        return point_from_basis(
            u_knots,
            v_knots,
            poles,
            (first_u, &u_values),
            (first_v, &v_values),
        );
    }
    if u_knots.degree().max(v_knots.degree()) < curve::SMALL {
        point_sized::<{ curve::SMALL }>(u_knots, v_knots, poles, u, v)
    } else {
        point_sized::<{ curve::LARGE }>(u_knots, v_knots, poles, u, v)
    }
}

#[inline]
fn point_sized<const N: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u: f64,
    v: f64,
) -> Point3D {
    let mut u_values = [0.0f64; N];
    let mut v_values = [0.0f64; N];
    let first_u = u_knots.basis(u, &mut u_values);
    let first_v = v_knots.basis(v, &mut v_values);
    point_from_basis(
        u_knots,
        v_knots,
        poles,
        (first_u, &u_values),
        (first_v, &v_values),
    )
}

/// The point from each direction's first pole and non-zero basis
/// functions.
#[inline]
fn point_from_basis<const N: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    (first_u, u_values): (usize, &[f64; N]),
    (first_v, v_values): (usize, &[f64; N]),
) -> Point3D {
    let s = match poles.weights {
        Some(weights) => point_sum::<N, true>(
            u_knots,
            v_knots,
            poles.points,
            weights,
            u_values,
            v_values,
            first_u,
            first_v,
        ),
        None => point_sum::<N, false>(
            u_knots,
            v_knots,
            poles.points,
            &[],
            u_values,
            v_values,
            first_u,
            first_v,
        ),
    };
    if poles.weights.is_some() {
        let inverse = 1.0 / s[3];
        Point3D::new(s[0] * inverse, s[1] * inverse, s[2] * inverse)
    } else {
        Point3D::new(s[0], s[1], s[2])
    }
}

/// `Σ_i Σ_j Nu_i Nv_j w_ij (P_ij, 1)` over the local poles (`w_ij = 1`
/// unless `RATIONAL`), from the non-zero basis functions of each direction
/// and the first pole they weight.
#[allow(clippy::too_many_arguments)] // both directions' bases and poles
#[inline]
fn point_sum<const N: usize, const RATIONAL: bool>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    points: &[Point3D],
    weights: &[f64],
    u_values: &[f64; N],
    v_values: &[f64; N],
    first_u: usize,
    first_v: usize,
) -> [f64; 4] {
    let (p, q) = (u_knots.degree(), v_knots.degree());
    let n_v = v_knots.pole_count();
    let wraps = v_knots.wraps(first_v);
    let mut sum = [0.0f64; 4];
    for (i, &nu) in u_values.iter().enumerate().take(p + 1) {
        let row = u_knots.pole_index(first_u, i) * n_v;
        let mut along = [0.0f64; 4];
        for (j, &nv) in v_values.iter().enumerate().take(q + 1) {
            // Only a periodic direction's last spans wrap.
            let index = row
                + if wraps {
                    v_knots.pole_index(first_v, j)
                } else {
                    first_v + j
                };
            let point = points[index];
            let b = if RATIONAL { nv * weights[index] } else { nv };
            along[0] += b * point.x;
            along[1] += b * point.y;
            along[2] += b * point.z;
            along[3] += b;
        }
        for c in 0..4 {
            sum[c] += nu * along[c];
        }
    }
    sum
}

/// The surface point (as a vector from the origin) and its partial
/// derivatives up to total order `ORDER` (`<= 2`), `[S, Su, Sv, Suu, Suv,
/// Svv]`, those beyond `ORDER` zero.
///
/// A rational surface applies the quotient rule to the homogeneous
/// results: `S = H / w`, `Su = (Hu - wu S) / w`,
/// `Suu = (Huu - 2 wu Su - wuu S) / w`,
/// `Suv = (Huv - wu Sv - wv Su - wuv S) / w`, and alike along `v`.
pub(crate) fn surface_derivatives<const ORDER: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u: f64,
    v: f64,
) -> [Vector3D; 6] {
    surface_derivatives_from::<ORDER>(u_knots, v_knots, poles, u, v, None, None)
}

/// [`surface_derivatives`] from one side per direction: [`Side::Below`]
/// evaluates the span ending at the parameter in that direction,
/// [`Side::Above`] the span starting there (across the seam of a
/// periodic direction too).
pub(crate) fn surface_derivatives_sided<const ORDER: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u: f64,
    v: f64,
    u_side: Side,
    v_side: Side,
) -> [Vector3D; 6] {
    surface_derivatives_from::<ORDER>(u_knots, v_knots, poles, u, v, Some(u_side), Some(v_side))
}

/// [`surface_derivatives`] with an optional side per direction, `None`
/// being plain evaluation.
#[inline]
fn surface_derivatives_from<const ORDER: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u: f64,
    v: f64,
    u_side: Option<Side>,
    v_side: Option<Side>,
) -> [Vector3D; 6] {
    let sums = tensor::<ORDER>(u_knots, v_knots, poles, u, v, u_side, v_side);
    let [s, su, sv, suu, suv, svv] = sums.map(|a| Vector3D::new(a[0], a[1], a[2]));
    if poles.weights.is_none() {
        return [s, su, sv, suu, suv, svv];
    }
    let [w, wu, wv, wuu, wuv, wvv] = sums.map(|a| a[3]);
    let inverse = 1.0 / w;
    let s = s * inverse;
    let su = (su - wu * s) * inverse;
    let sv = (sv - wv * s) * inverse;
    let suu = (suu - 2.0 * wu * su - wuu * s) * inverse;
    let suv = (suv - wu * sv - wv * su - wuv * s) * inverse;
    let svv = (svv - 2.0 * wv * sv - wvv * s) * inverse;
    [s, su, sv, suu, suv, svv]
}

/// Axis-aligned bounding boxes `[min_x, min_y, min_z, max_x, max_y,
/// max_z]` of the control poles of every knot-span patch, row-major over
/// the spans between consecutive `u_breaks` (outer) and `v_breaks` (inner).
///
/// By the convex hull property (weights are positive) each patch lies in
/// its box, so the distance to the box bounds the distance to the patch
/// from below.
pub(crate) fn patch_boxes(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u_breaks: &[f64],
    v_breaks: &[f64],
) -> Vec<[f64; 6]> {
    if u_knots.degree().max(v_knots.degree()) < curve::SMALL {
        boxes::<{ curve::SMALL }>(u_knots, v_knots, poles, u_breaks, v_breaks)
    } else {
        boxes::<{ curve::LARGE }>(u_knots, v_knots, poles, u_breaks, v_breaks)
    }
}

fn boxes<const N: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u_breaks: &[f64],
    v_breaks: &[f64],
) -> Vec<[f64; 6]> {
    // The window at a span's midpoint selects exactly the poles of that
    // span's patch.
    let span_window = |w: &[f64], knots: &KnotVector| {
        let mut out = curve::Window::<N>::new();
        knots.window(0.5 * (w[0] + w[1]), 0, &mut out);
        out
    };
    let v_windows: Vec<curve::Window<N>> = v_breaks
        .windows(2)
        .map(|w| span_window(w, v_knots))
        .collect();
    let n_v = v_knots.pole_count();
    let mut out = Vec::with_capacity((u_breaks.len() - 1) * v_windows.len());
    for w in u_breaks.windows(2) {
        let wu = span_window(w, u_knots);
        for wv in &v_windows {
            let mut b = [
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for i in 0..=u_knots.degree() {
                let row = wu.pole(i) * n_v;
                for j in 0..=v_knots.degree() {
                    let p = poles.points[row + wv.pole(j)];
                    for (c, x) in [p.x, p.y, p.z].into_iter().enumerate() {
                        b[c] = b[c].min(x);
                        b[c + 3] = b[c + 3].max(x);
                    }
                }
            }
            out.push(b);
        }
    }
    out
}

/// One direction of a knot-span patch: its span's
/// [`KnotVector::bezier_matrix`] (row-major, `(degree + 1)^2` values), first
/// pole index, degree and pole count.
pub(crate) type SpanMatrix<'a> = (&'a [f64], usize, usize, usize);

/// Bernstein control net of a knot-span patch over its own parameter
/// rectangle, written row-major `[k_u][k_v][h]` to `out`
/// (`(u_degree + 1) * (v_degree + 1) * h` values, homogeneous when
/// rational); `scratch` holds as many.
///
/// The tensor product of each direction's span matrix applied to the
/// patch's poles, the pole indices wrapping modulo the pole count (a
/// periodic direction), so seams come out exactly as the evaluator's.
pub(crate) fn patch_bezier(
    u: SpanMatrix,
    v: SpanMatrix,
    poles: Poles,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    // Both degrees 2 or 3 take a path fixed to them, whose loops unroll.
    let fixed = if u.2 == v.2 { u.2 + 1 } else { 0 };
    match (poles.weights.is_some(), fixed, u.2.max(v.2) < curve::SMALL) {
        (true, 3, _) => bezier_net::<4, 3, true>(u, v, poles, scratch, out),
        (true, 4, _) => bezier_net::<4, 4, true>(u, v, poles, scratch, out),
        (true, _, true) => bezier_net::<4, { curve::SMALL }, false>(u, v, poles, scratch, out),
        (true, _, false) => bezier_net::<4, { curve::LARGE }, false>(u, v, poles, scratch, out),
        (false, 3, _) => bezier_net::<3, 3, true>(u, v, poles, scratch, out),
        (false, 4, _) => bezier_net::<3, 4, true>(u, v, poles, scratch, out),
        (false, _, true) => bezier_net::<3, { curve::SMALL }, false>(u, v, poles, scratch, out),
        (false, _, false) => bezier_net::<3, { curve::LARGE }, false>(u, v, poles, scratch, out),
    }
}

/// [`patch_bezier`] with `H` values per point, for degrees below `N`, or
/// both `N - 1` if `FIXED`.
fn bezier_net<const H: usize, const N: usize, const FIXED: bool>(
    (mu, first_u, p, n_u): SpanMatrix,
    (mv, first_v, q, n_v): SpanMatrix,
    poles: Poles,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    let (m, n) = if FIXED { (N, N) } else { (p + 1, q + 1) };
    let point = |net: &[f64], k: usize| -> [f64; H] { std::array::from_fn(|c| net[k * H + c]) };
    // Along v first: `scratch[i][l]` is Bernstein point `l` of pole row `i`.
    for i in 0..m {
        let start = ((first_u + i) % n_u) * n_v;
        let mut row = [[0.0f64; H]; N];
        for j in 0..n {
            let homogeneous = poles.homogeneous(start + (first_v + j) % n_v);
            for (x, weights) in row.iter_mut().zip(mv.chunks_exact(n)) {
                for c in 0..H {
                    x[c] += weights[j] * homogeneous[c];
                }
            }
        }
        scratch[i * n * H..(i + 1) * n * H].copy_from_slice(row[..n].as_flattened());
    }
    for (k, weights) in mu.chunks_exact(m).enumerate() {
        let mut row = [[0.0f64; H]; N];
        for (i, &weight) in weights.iter().enumerate() {
            for (l, x) in row.iter_mut().enumerate().take(n) {
                let y = point(scratch, i * n + l);
                for c in 0..H {
                    x[c] += weight * y[c];
                }
            }
        }
        out[k * n * H..(k + 1) * n * H].copy_from_slice(row[..n].as_flattened());
    }
}

/// De Casteljau subdivision at the midpoint of a Bernstein net, row-major
/// `[rows][cols][h]`: along the rows (the `u` direction) if `along_rows`,
/// else along the columns. `left` and `right` receive the nets over the
/// lower and upper halves, each over its own unit square.
pub(crate) fn halve_net(
    net: &[f64],
    rows: usize,
    cols: usize,
    h: usize,
    along_rows: bool,
    left: &mut [f64],
    right: &mut [f64],
) {
    if along_rows {
        halve_blocks(net, rows, cols * h, left, right);
    } else {
        let width = cols * h;
        for r in 0..rows {
            let span = r * width..(r + 1) * width;
            halve_blocks(
                &net[span.clone()],
                cols,
                h,
                &mut left[span.clone()],
                &mut right[span],
            );
        }
    }
}

/// [`halve_net`] on a sequence of `n` control points of `block` values.
fn halve_blocks(net: &[f64], n: usize, block: usize, left: &mut [f64], right: &mut [f64]) {
    right.copy_from_slice(net);
    for r in 0..n {
        left[r * block..(r + 1) * block].copy_from_slice(&right[..block]);
        for x in 0..(n - 1 - r) * block {
            right[x] = 0.5 * (right[x] + right[x + block]);
        }
    }
}

/// The Euclidean points of a Bernstein net with `h` values per point.
pub(crate) fn net_points(net: &[f64], h: usize, out: &mut Vec<[f64; 3]>) {
    out.clear();
    out.extend(net.chunks_exact(h).map(|c| {
        if h == 4 {
            [c[0] / c[3], c[1] / c[3], c[2] / c[3]]
        } else {
            [c[0], c[1], c[2]]
        }
    }));
}

/// Lower bound on the distance from `point` to a Bernstein patch with the
/// Euclidean control points `points` (`[rows][cols]`): the larger of the
/// distances to their axis-aligned box and to their box in a frame fitted
/// to the patch corners. Weights are positive, so the patch lies in the
/// points' convex hull, hence in both boxes; on a small patch the fitted
/// box is flat, its bound exact to second order in the patch size.
pub(crate) fn net_distance_bound(
    points: &[[f64; 3]],
    rows: usize,
    cols: usize,
    point: [f64; 3],
) -> f64 {
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let add = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let unit = |a: [f64; 3]| {
        let n = dot(a, a).sqrt();
        (n > 0.0).then(|| [a[0] / n, a[1] / n, a[2] / n])
    };
    let box_distance = |axes: &[[f64; 3]; 3]| {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &x in points {
            let d = sub(x, point);
            for a in 0..3 {
                // Comparisons, cheaper than `f64::min`.
                let y = dot(d, axes[a]);
                if y < lo[a] {
                    lo[a] = y;
                }
                if y > hi[a] {
                    hi[a] = y;
                }
            }
        }
        (0..3)
            .map(|a| lo[a].max(-hi[a]).max(0.0).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let aligned = box_distance(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let corner = |i: usize, j: usize| points[i * cols + j];
    let (c00, c01) = (corner(0, 0), corner(0, cols - 1));
    let (c10, c11) = (corner(rows - 1, 0), corner(rows - 1, cols - 1));
    let along_s = add(sub(c10, c00), sub(c11, c01));
    let along_t = add(sub(c01, c00), sub(c11, c10));
    // A net collapsed to an edge curve has no extent along `s`: then the
    // frame follows the curve's chord, with any normal to it.
    let fitted = unit(along_s).or_else(|| unit(along_t)).map_or(0.0, |t1| {
        let n = unit(cross(along_s, along_t)).unwrap_or_else(|| {
            let i = (0..3)
                .min_by(|&a, &b| t1[a].abs().total_cmp(&t1[b].abs()))
                .unwrap_or(0);
            let mut axis = [0.0; 3];
            axis[i] = 1.0;
            unit(cross(t1, axis)).unwrap_or(axis)
        });
        box_distance(&[t1, cross(n, t1), n])
    });
    aligned.max(fitted)
}

/// Replaces a Bernstein net, row-major `[rows][cols][h]`, by the net of its
/// edge curve `side` (`0` or `1`) across the rows (the `u` direction) if
/// `along_rows`, else across the columns: every row (column) becomes a
/// copy of that edge one, so the net no longer varies in that direction.
pub(crate) fn collapse_net(
    net: &mut [f64],
    rows: usize,
    cols: usize,
    h: usize,
    along_rows: bool,
    side: usize,
) {
    let width = cols * h;
    if along_rows {
        let edge = side * (rows - 1) * width;
        for r in 0..rows {
            net.copy_within(edge..edge + width, r * width);
        }
    } else {
        for r in 0..rows {
            let edge = r * width + side * (cols - 1) * h;
            for c in 0..cols {
                net.copy_within(edge..edge + h, r * width + c * h);
            }
        }
    }
}

/// A closed interval `[lo, hi]`, enclosing a function over a patch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Interval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl Interval {
    pub(crate) const ZERO: Interval = Interval { lo: 0.0, hi: 0.0 };
    const EMPTY: Interval = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };

    fn hull(self, x: f64) -> Interval {
        Interval {
            lo: self.lo.min(x),
            hi: self.hi.max(x),
        }
    }

    fn scaled(self, k: f64) -> Interval {
        Interval {
            lo: self.lo * k,
            hi: self.hi * k,
        }
    }

    fn shifted(self, x: f64) -> Interval {
        Interval {
            lo: self.lo + x,
            hi: self.hi + x,
        }
    }

    pub(crate) fn add(self, o: Interval) -> Interval {
        Interval {
            lo: self.lo + o.lo,
            hi: self.hi + o.hi,
        }
    }

    pub(crate) fn sub(self, o: Interval) -> Interval {
        Interval {
            lo: self.lo - o.hi,
            hi: self.hi - o.lo,
        }
    }

    pub(crate) fn mul(self, o: Interval) -> Interval {
        let (a, b) = (self.lo * o.lo, self.lo * o.hi);
        let (c, d) = (self.hi * o.lo, self.hi * o.hi);
        Interval {
            lo: a.min(b).min(c.min(d)),
            hi: a.max(b).max(c.max(d)),
        }
    }

    pub(crate) fn square(self) -> Interval {
        let (a, b) = (self.lo * self.lo, self.hi * self.hi);
        Interval {
            lo: if self.lo <= 0.0 && self.hi >= 0.0 {
                0.0
            } else {
                a.min(b)
            },
            hi: a.max(b),
        }
    }

    /// The largest magnitude in the interval.
    pub(crate) fn magnitude(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    /// The midpoint.
    pub(crate) fn mid(self) -> f64 {
        0.5 * (self.lo + self.hi)
    }
}

/// Enclosures, over a Bernstein patch in its own parameters `(s, t)` in
/// `[0, 1]^2`, of the derivatives of `f = |S - point|^2 / 2`.
pub(crate) struct DistanceBounds {
    /// `[f_s, f_t]`.
    pub(crate) gradient: [Interval; 2],
    /// `[f_ss, f_st, f_tt]`.
    pub(crate) hessian: [Interval; 3],
    /// Upper bounds on `|S_s|^2` and `|S_t|^2`: how far the patch stretches
    /// in each direction.
    pub(crate) stretch: [f64; 2],
}

/// [`DistanceBounds`] of the patch with the Bernstein net `net`, row-major
/// `[rows][cols][h]` (homogeneous when `h == 4`), whose Euclidean control
/// points are `points`.
///
/// With the enclosures of [`derivative_bounds`], `f_s = S_s . r`, `f_ss =
/// S_s . S_s + S_ss . r` and so on, with `r = S - point` in the hull of
/// `points`.
///
/// Unless `second`, only the gradient is bounded, the rest left zero.
pub(crate) fn distance_bounds(
    net: &[f64],
    points: &[[f64; 3]],
    rows: usize,
    cols: usize,
    h: usize,
    point: [f64; 3],
    second: bool,
) -> DistanceBounds {
    let (value, d) = derivative_bounds(net, points, rows, cols, h, second);
    let mut gradient = [Interval::ZERO; 2];
    let mut hessian = [Interval::ZERO; 3];
    let mut stretch = [0.0; 2];
    for c in 0..3 {
        let r = value[c].shifted(-point[c]);
        gradient[0] = gradient[0].add(d[0][c].mul(r));
        gradient[1] = gradient[1].add(d[1][c].mul(r));
        hessian[0] = hessian[0].add(d[0][c].square()).add(d[2][c].mul(r));
        hessian[1] = hessian[1].add(d[0][c].mul(d[1][c])).add(d[3][c].mul(r));
        hessian[2] = hessian[2].add(d[1][c].square()).add(d[4][c].mul(r));
        stretch[0] += d[0][c].magnitude().powi(2);
        stretch[1] += d[1][c].magnitude().powi(2);
    }
    DistanceBounds {
        gradient,
        hessian,
        stretch,
    }
}

/// Enclosures, over the patch with the Bernstein net `net` (as in
/// [`distance_bounds`]) in its own parameters `(s, t)` in `[0, 1]^2`, of
/// each coordinate of the point (the hull of `points`) and of the
/// derivatives `[S_s, S_t, S_ss, S_st, S_tt]`.
///
/// The derivatives of the (homogeneous) patch lie in the hulls of their
/// own Bernstein nets, scaled forward differences of `net`; rational ones
/// follow by the quotient rule in interval arithmetic, about the centre
/// of the patch's box so that the intervals shrink with the patch.
///
/// Unless `second`, only the first derivatives are bounded, the rest left
/// zero.
pub(crate) fn derivative_bounds(
    net: &[f64],
    points: &[[f64; 3]],
    rows: usize,
    cols: usize,
    h: usize,
    second: bool,
) -> ([Interval; 3], [[Interval; 3]; 5]) {
    let (p, q) = (rows - 1, cols - 1);
    let mut value = [Interval::EMPTY; 3];
    for x in points {
        for c in 0..3 {
            if x[c] < value[c].lo {
                value[c].lo = x[c];
            }
            if x[c] > value[c].hi {
                value[c].hi = x[c];
            }
        }
    }
    // A rational patch is taken about its centre: `S - centre` has the
    // homogeneous coordinates `H - centre * w`.
    let centre = if h == 4 {
        value.map(Interval::mid)
    } else {
        [0.0; 3]
    };
    let (mut hulls, weight) = if h == 4 {
        difference_hulls::<4>(net, rows, cols, centre, second)
    } else {
        difference_hulls::<3>(net, rows, cols, centre, second)
    };
    let (pf, qf) = (p as f64, q as f64);
    let factors = [pf, qf, pf * (pf - 1.0), pf * qf, qf * (qf - 1.0)];
    for (hull, k) in hulls.iter_mut().zip(factors) {
        for x in hull.iter_mut() {
            // An empty hull is a derivative beyond the degree: zero.
            *x = if x.lo > x.hi {
                Interval::ZERO
            } else {
                x.scaled(k)
            };
        }
    }
    // Euclidean `[S_s, S_t, S_ss, S_st, S_tt]`, per coordinate.
    let mut d = [[Interval::ZERO; 3]; 5];
    if h == 4 {
        let inverse = Interval {
            lo: 1.0 / weight.hi,
            hi: 1.0 / weight.lo,
        };
        let [ws, wt, wss, wst, wtt] = [0, 1, 2, 3, 4].map(|k| hulls[k][3]);
        for c in 0..3 {
            let s = value[c].shifted(-centre[c]);
            let ss = hulls[0][c].sub(s.mul(ws)).mul(inverse);
            let st = hulls[1][c].sub(s.mul(wt)).mul(inverse);
            d[0][c] = ss;
            d[1][c] = st;
            d[2][c] = hulls[2][c]
                .sub(ws.mul(ss).scaled(2.0))
                .sub(wss.mul(s))
                .mul(inverse);
            d[3][c] = hulls[3][c]
                .sub(ws.mul(st))
                .sub(wt.mul(ss))
                .sub(wst.mul(s))
                .mul(inverse);
            d[4][c] = hulls[4][c]
                .sub(wt.mul(st).scaled(2.0))
                .sub(wtt.mul(s))
                .mul(inverse);
        }
    } else {
        for (dk, hull) in d.iter_mut().zip(&hulls) {
            dk.copy_from_slice(&hull[..3]);
        }
    }
    (value, d)
}

/// Hulls of each homogeneous coordinate's difference nets for
/// [`derivative_bounds`], in the order `[s, t, ss, st, tt]` (the second
/// ones only if `second`), before the degree factors, and the hull of the
/// weights (empty unless `H == 4`). A rational net, of `H == 4` values a
/// point, is taken about `centre`.
fn difference_hulls<const H: usize>(
    net: &[f64],
    rows: usize,
    cols: usize,
    centre: [f64; 3],
    second: bool,
) -> ([[Interval; 4]; 5], Interval) {
    let width = cols * H;
    let row = |i: usize| &net[i * width..(i + 1) * width];
    // A point of the net, about `centre` if rational.
    let shifted = |x: &[f64; H]| -> [f64; H] {
        std::array::from_fn(|c| {
            if c < 3 && H == 4 {
                x[c] - centre[c] * x[3]
            } else {
                x[c]
            }
        })
    };
    let points = |i: usize| row(i).as_chunks::<H>().0.iter().map(shifted);
    let mut hulls = [[Interval::EMPTY; 4]; 5];
    // Each difference net over exactly its own indices, the hull widened
    // by comparisons (cheaper than `f64::min`, and as blind to NaN).
    let mut widen = |k: usize, d: [f64; H]| {
        for (hull, x) in hulls[k].iter_mut().zip(d) {
            if x < hull.lo {
                hull.lo = x;
            }
            if x > hull.hi {
                hull.hi = x;
            }
        }
    };
    let difference =
        |a: [f64; H], b: [f64; H]| -> [f64; H] { std::array::from_fn(|c| b[c] - a[c]) };
    for i in 0..rows {
        // Along `t`, within the row.
        for (a, b) in points(i).zip(points(i).skip(1)) {
            widen(1, difference(a, b));
        }
        if second {
            for ((a, b), c) in points(i).zip(points(i).skip(1)).zip(points(i).skip(2)) {
                widen(4, difference(difference(a, b), difference(b, c)));
            }
        }
        if i + 1 == rows {
            break;
        }
        // Along `s`, to the next row.
        for (a, b) in points(i).zip(points(i + 1)) {
            widen(0, difference(a, b));
        }
        if second {
            for ((a, b), (c, d)) in points(i)
                .zip(points(i + 1))
                .zip(points(i).skip(1).zip(points(i + 1).skip(1)))
            {
                widen(3, difference(difference(a, b), difference(c, d)));
            }
            if i + 2 < rows {
                for ((a, b), c) in points(i).zip(points(i + 1)).zip(points(i + 2)) {
                    widen(2, difference(difference(a, b), difference(b, c)));
                }
            }
        }
    }
    let mut weight = Interval::EMPTY;
    if H == 4 {
        for x in net.as_chunks::<H>().0 {
            weight = weight.hull(x[3]);
        }
    }
    (hulls, weight)
}

/// Homogeneous `[H, Hu, Hv, Huu, Huv, Hvv]` at `(u, v)` (`w = 1` if
/// nonrational), the derivatives beyond total order `ORDER` zero.
fn tensor<const ORDER: usize>(
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    poles: Poles,
    u: f64,
    v: f64,
    u_side: Option<Side>,
    v_side: Option<Side>,
) -> [[f64; 4]; 6] {
    fn sized<const N: usize, const ORDER: usize>(
        u_knots: &KnotVector,
        v_knots: &KnotVector,
        poles: Poles,
        u: f64,
        v: f64,
        u_side: Option<Side>,
        v_side: Option<Side>,
    ) -> [[f64; 4]; 6] {
        let mut wu = curve::Window::<N>::new();
        let mut wv = curve::Window::<N>::new();
        u_knots.window_sided(u, ORDER, u_side, &mut wu);
        v_knots.window_sided(v, ORDER, v_side, &mut wv);
        match poles.weights {
            Some(weights) => {
                sum::<N, ORDER, true>(&wu, &wv, u_knots, v_knots, poles.points, weights)
            }
            None => sum::<N, ORDER, false>(&wu, &wv, u_knots, v_knots, poles.points, &[]),
        }
    }
    if u_knots.degree().max(v_knots.degree()) < curve::SMALL {
        sized::<{ curve::SMALL }, ORDER>(u_knots, v_knots, poles, u, v, u_side, v_side)
    } else {
        sized::<{ curve::LARGE }, ORDER>(u_knots, v_knots, poles, u, v, u_side, v_side)
    }
}

/// `Σ_i Σ_j Nu_i Nv_j w_ij (P_ij, 1)` over the local poles (`w_ij = 1`
/// unless `RATIONAL`), with the mixed basis derivatives up to total order
/// `ORDER` in the slots of [`tensor`].
fn sum<const N: usize, const ORDER: usize, const RATIONAL: bool>(
    wu: &curve::Window<N>,
    wv: &curve::Window<N>,
    u_knots: &KnotVector,
    v_knots: &KnotVector,
    points: &[Point3D],
    weights: &[f64],
) -> [[f64; 4]; 6] {
    let n_v = v_knots.pole_count();
    let mut out = [[0.0f64; 4]; 6];
    for i in 0..=u_knots.degree() {
        let row = wu.pole(i) * n_v;
        // Along v: the value row and its v-derivatives up to order ORDER.
        let mut along = [[0.0f64; 4]; 3];
        for j in 0..=v_knots.degree() {
            let index = row + wv.pole(j);
            let p = points[index];
            let weight = if RATIONAL { weights[index] } else { 1.0 };
            for (k, acc) in along.iter_mut().enumerate().take(ORDER + 1) {
                let b = wv.ders[k][j] * weight;
                acc[0] += b * p.x;
                acc[1] += b * p.y;
                acc[2] += b * p.z;
                acc[3] += b;
            }
        }
        let nu = wu.ders[0][i];
        for c in 0..4 {
            out[0][c] += nu * along[0][c];
        }
        if ORDER >= 1 {
            let du = wu.ders[1][i];
            for c in 0..4 {
                out[1][c] += du * along[0][c];
                out[2][c] += nu * along[1][c];
            }
        }
        if ORDER >= 2 {
            let (du, duu) = (wu.ders[1][i], wu.ders[2][i]);
            for c in 0..4 {
                out[3][c] += duu * along[0][c];
                out[4][c] += du * along[1][c];
                out[5][c] += nu * along[2][c];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- fixed-degree point path ----

    #[test]
    fn test_point_of_low_degrees_matches_general_path() {
        // Every pair of degrees 1 to 3 and a degree-4 direction, which takes
        // the general path, over uneven knots, polynomial and rational.
        for (p, q) in [(1, 3), (2, 2), (3, 1), (3, 3), (2, 4)] {
            let direction = |degree: usize| {
                KnotVector::new(
                    degree,
                    vec![0.0, 0.4, 0.55, 1.0],
                    vec![degree as u32 + 1, 1, 1, degree as u32 + 1],
                    false,
                    degree + 3,
                )
                .unwrap()
            };
            let (u_knots, v_knots) = (direction(p), direction(q));
            let count = (p + 3) * (q + 3);
            let poles: Vec<Point3D> = (0..count)
                .map(|k| {
                    let (i, j) = ((k / (q + 3)) as f64, (k % (q + 3)) as f64);
                    Point3D::new(i, j, (0.7 * i + 1.3 * j).sin())
                })
                .collect();
            let weights: Vec<f64> = (0..count).map(|k| 0.6 + 0.2 * (k % 4) as f64).collect();
            for weights in [None, Some(weights.as_slice())] {
                let view = Poles {
                    points: &poles,
                    weights,
                };
                for a in -1..=11 {
                    for b in -1..=11 {
                        let (u, v) = (a as f64 / 10.0, b as f64 / 10.0);
                        let point = surface_point(&u_knots, &v_knots, view, u, v);
                        let [s, ..] = surface_derivatives::<0>(&u_knots, &v_knots, view, u, v);
                        let close = (point.x - s.x)
                            .abs()
                            .max((point.y - s.y).abs())
                            .max((point.z - s.z).abs());
                        assert!(
                            close < 1e-12,
                            "degrees {p} {q} at ({u}, {v}): {point:?} vs {s:?}"
                        );
                    }
                }
            }
        }
    }

    // ---- bilinear patch by hand: the smallest tensor product ----

    // Degree-1 x degree-1 patch over the unit square with corner poles
    //   P00=(0,0,0) P01=(0,1,0)
    //   P10=(1,0,0) P11=(1,1,1)
    // (row-major [i_u][i_v]). Bilinear interpolation gives
    //   S(u,v) = (u, v, u*v), Su = (1, 0, v), Sv = (0, 1, u).
    fn bilinear_poles() -> Vec<Point3D> {
        vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(1.0, 1.0, 1.0),
        ]
    }

    fn unit_linear() -> KnotVector {
        KnotVector::new(1, vec![0.0, 1.0], vec![2, 2], false, 2).unwrap()
    }

    #[test]
    fn test_bilinear_patch_point_by_hand() {
        let poles = bilinear_poles();
        let knots = unit_linear();
        let view = Poles {
            points: &poles,
            weights: None,
        };
        let s = surface_point(&knots, &knots, view, 0.5, 0.5);
        assert_close3(&[s.x, s.y, s.z], &[0.5, 0.5, 0.25]);
    }

    #[test]
    fn test_bilinear_patch_derivatives_by_hand() {
        let poles = bilinear_poles();
        let knots = unit_linear();
        let view = Poles {
            points: &poles,
            weights: None,
        };
        // At (u, v) = (0.3, 0.7): S=(0.3,0.7,0.21), Su=(1,0,0.7), Sv=(0,1,0.3).
        let [s, su, sv, ..] = surface_derivatives::<1>(&knots, &knots, view, 0.3, 0.7);
        assert_close3(&[s.x, s.y, s.z], &[0.3, 0.7, 0.21]);
        assert_close3(&[su.x, su.y, su.z], &[1.0, 0.0, 0.7]);
        assert_close3(&[sv.x, sv.y, sv.z], &[0.0, 1.0, 0.3]);
    }

    // ---- fixture replay: the correctness gate ----

    #[test]
    fn test_fixture_replay_all_cases() {
        replay_fixture();
    }

    // ---- helpers ----

    fn assert_close3(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-9 * f64::max(1.0, e.abs());
            assert!((a - e).abs() <= tol, "index {i}: {a} vs {e}");
        }
    }

    use serde_json::Value;

    fn replay_fixture() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/surfaces_bspline.json"
        );
        let text = std::fs::read_to_string(path).expect("read fixture");
        let json: Value = serde_json::from_str(&text).expect("parse fixture");
        for case in json["cases"].as_array().expect("cases array") {
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

    fn arr_u32(v: &Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn replay_case(case: &Value) {
        let name = case["name"].as_str().unwrap();
        let u_degree = case["u_degree"].as_u64().unwrap() as usize;
        let v_degree = case["v_degree"].as_u64().unwrap() as usize;
        let u_periodic = case["u_periodic"].as_bool().unwrap();
        let v_periodic = case["v_periodic"].as_bool().unwrap();
        let u_knots = arr_f64(&case["u_knots"]);
        let v_knots = arr_f64(&case["v_knots"]);
        let u_mults = arr_u32(&case["u_mults"]);
        let v_mults = arr_u32(&case["v_mults"]);

        // poles[i_u][i_v] = [x, y, z]
        let pole_grid: Vec<Vec<Vec<f64>>> = case["poles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(arr_f64).collect())
            .collect();
        let n_u = pole_grid.len();
        let n_v = pole_grid[0].len();

        let weights: Option<Vec<Vec<f64>>> = case.get("weights").and_then(|w| {
            if w.is_null() {
                None
            } else {
                Some(w.as_array().unwrap().iter().map(arr_f64).collect())
            }
        });
        let poles: Vec<Point3D> = pole_grid
            .iter()
            .flatten()
            .map(|p| Point3D::new(p[0], p[1], p[2]))
            .collect();
        let weights: Option<Vec<f64>> = weights.map(|w| w.concat());
        let view = Poles {
            points: &poles,
            weights: weights.as_deref(),
        };
        let u_vector = KnotVector::new(u_degree, u_knots, u_mults, u_periodic, n_u).unwrap();
        let v_vector = KnotVector::new(v_degree, v_knots, v_mults, v_periodic, n_v).unwrap();

        for sample in case["samples"].as_array().unwrap() {
            let u = sample["u"].as_f64().unwrap();
            let v = sample["v"].as_f64().unwrap();

            let s0 = surface_point(&u_vector, &v_vector, view, u, v);
            let [s, su, sv, ..] = surface_derivatives::<1>(&u_vector, &v_vector, view, u, v);
            let [_, _, _, suu, suv, svv] =
                surface_derivatives::<2>(&u_vector, &v_vector, view, u, v);
            for (key, d) in [("d2u", suu), ("d2uv", suv), ("d2v", svv)] {
                check(name, u, v, key, &[d.x, d.y, d.z], &arr_f64(&sample[key]));
            }
            // d0 and d1 agree on the point.
            check(name, u, v, "d0", &[s0.x, s0.y, s0.z], &[s.x, s.y, s.z]);
            check(
                name,
                u,
                v,
                "point",
                &[s.x, s.y, s.z],
                &arr_f64(&sample["point"]),
            );
            check(
                name,
                u,
                v,
                "d1u",
                &[su.x, su.y, su.z],
                &arr_f64(&sample["d1u"]),
            );
            check(
                name,
                u,
                v,
                "d1v",
                &[sv.x, sv.y, sv.z],
                &arr_f64(&sample["d1v"]),
            );
        }
    }

    fn check(name: &str, u: f64, v: f64, key: &str, got: &[f64], expected: &[f64]) {
        for (c, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-7 * f64::max(1.0, e.abs());
            assert!(
                (g - e).abs() <= tol,
                "{name} (u={u}, v={v}) {key}[{c}]: got {g}, expected {e} (tol {tol})"
            );
        }
    }
}
