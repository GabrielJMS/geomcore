//! Internal B-spline surface evaluation: tensor-product NURBS on top of the
//! curve engine ([`crate::curve_math::bspline`]).
//!
//! A B-spline surface is a tensor product of two directional B-spline bases.
//! Poles are stored row-major as `[i_u][i_v][h]`, where `i_u` indexes the `u`
//! direction (`n_u` rows), `i_v` indexes the `v` direction (`n_v` columns),
//! and `h` is the per-pole coordinate stride (`h = 3` for polynomial surfaces,
//! `h = 4` for the homogeneous coordinates `(x*w, y*w, z*w, w)` of a rational
//! surface).
//!
//! Evaluation is a direct tensor product: [`curve::window`] gives the
//! non-zero basis functions (and derivatives) in each direction with the
//! pole indices they weight, already wrapped for periodic directions, and
//! only the `(u_degree + 1) * (v_degree + 1)` poles they select are summed.
//! The cost depends on the degrees only, never on the pole counts, and
//! nothing is allocated.
//!
//! For rational surfaces the whole computation is carried out in homogeneous
//! coordinates (`h = 4`), and the quotient rule is applied per output at the
//! very end: `S = H/w`, `Su = (Hu_xyz - S*Hu_w)/w`, `Sv = (Hv_xyz - S*Hv_w)/w`.

use super::super::curve_math::bspline as curve;
use crate::{Point3D, Vector3D};

/// Evaluates the surface point `S(u, v)`.
///
/// `poles_homog` is the flat, row-major `[i_u][i_v][h]` pole buffer with `h`
/// coordinates per pole (`h = 3` polynomial, `h = 4` homogeneous). `u_flat` /
/// `v_flat` are the plain [`curve::flat_knots`] of each direction.
#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
pub(crate) fn surface_d0(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    u_flat: &[f64],
    v_flat: &[f64],
    poles_homog: &[f64],
    n_u: usize,
    n_v: usize,
    h: usize,
) -> Point3D {
    debug_assert_eq!(poles_homog.len(), n_u * n_v * h);
    let [hs, ..] = if u_degree.max(v_degree) < curve::SMALL {
        tensor::<{ curve::SMALL }>(
            u,
            v,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
            0,
        )
    } else {
        tensor::<{ curve::LARGE }>(
            u,
            v,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
            0,
        )
    };
    if h == 4 {
        let w = hs[3];
        Point3D::new(hs[0] / w, hs[1] / w, hs[2] / w)
    } else {
        Point3D::new(hs[0], hs[1], hs[2])
    }
}

/// Evaluates the surface value and both first partial derivatives,
/// returning `(S, Su, Sv)`.
///
/// Layout of `poles_homog` and the knot flats is as for [`surface_d0`]. For
/// rational surfaces the first-order quotient rule is applied to the
/// homogeneous results.
#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
pub(crate) fn surface_d1(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    u_flat: &[f64],
    v_flat: &[f64],
    poles_homog: &[f64],
    n_u: usize,
    n_v: usize,
    h: usize,
) -> (Point3D, Vector3D, Vector3D) {
    debug_assert_eq!(poles_homog.len(), n_u * n_v * h);
    let [h_s, h_su, h_sv, ..] = if u_degree.max(v_degree) < curve::SMALL {
        tensor::<{ curve::SMALL }>(
            u,
            v,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
            1,
        )
    } else {
        tensor::<{ curve::LARGE }>(
            u,
            v,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
            1,
        )
    };

    if h == 4 {
        // Rational quotient rule, first order:
        //   S  = H / w
        //   Su = (Hu_xyz - S * Hu_w) / w
        //   Sv = (Hv_xyz - S * Hv_w) / w
        let w = h_s[3];
        let s = Point3D::new(h_s[0] / w, h_s[1] / w, h_s[2] / w);
        let su = Vector3D::new(
            (h_su[0] - s.x * h_su[3]) / w,
            (h_su[1] - s.y * h_su[3]) / w,
            (h_su[2] - s.z * h_su[3]) / w,
        );
        let sv = Vector3D::new(
            (h_sv[0] - s.x * h_sv[3]) / w,
            (h_sv[1] - s.y * h_sv[3]) / w,
            (h_sv[2] - s.z * h_sv[3]) / w,
        );
        (s, su, sv)
    } else {
        let s = Point3D::new(h_s[0], h_s[1], h_s[2]);
        let su = Vector3D::new(h_su[0], h_su[1], h_su[2]);
        let sv = Vector3D::new(h_sv[0], h_sv[1], h_sv[2]);
        (s, su, sv)
    }
}

/// Evaluates the surface value with its first and second partial
/// derivatives, returning `[S, Su, Sv, Suu, Suv, Svv]` (the point as a
/// vector from the origin).
///
/// Layout of `poles_homog` and the knot flats is as for [`surface_d0`]. For
/// rational surfaces the second-order quotient rule is applied to the
/// homogeneous results:
/// `Suu = (Huu - 2 w_u Su - w_uu S)/w`,
/// `Suv = (Huv - w_u Sv - w_v Su - w_uv S)/w`,
/// `Svv = (Hvv - 2 w_v Sv - w_vv S)/w`.
#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
pub(crate) fn surface_d2(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    u_flat: &[f64],
    v_flat: &[f64],
    poles_homog: &[f64],
    n_u: usize,
    n_v: usize,
    h: usize,
) -> [Vector3D; 6] {
    debug_assert_eq!(poles_homog.len(), n_u * n_v * h);
    let hs = if u_degree.max(v_degree) < curve::SMALL {
        tensor::<{ curve::SMALL }>(
            u,
            v,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
            2,
        )
    } else {
        tensor::<{ curve::LARGE }>(
            u,
            v,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
            2,
        )
    };
    let xyz = |a: [f64; 4]| Vector3D::new(a[0], a[1], a[2]);
    let [s, su, sv, suu, suv, svv] = hs.map(xyz);
    if h != 4 {
        return [s, su, sv, suu, suv, svv];
    }
    let [w, wu, wv, wuu, wuv, wvv] = hs.map(|a| a[3]);
    let inv = 1.0 / w;
    let s = s * inv;
    let su = (su - wu * s) * inv;
    let sv = (sv - wv * s) * inv;
    let suu = (suu - 2.0 * wu * su - wuu * s) * inv;
    let suv = (suv - wu * sv - wv * su - wuv * s) * inv;
    let svv = (svv - 2.0 * wv * sv - wvv * s) * inv;
    [s, su, sv, suu, suv, svv]
}

/// Axis-aligned bounding boxes `[min_x, min_y, min_z, max_x, max_y,
/// max_z]` of the Euclidean control poles of every knot-span patch,
/// row-major over the spans between consecutive `u_breaks` (outer) and
/// `v_breaks` (inner).
///
/// By the convex hull property (weights are positive) each patch lies in
/// its box, so the distance to the box bounds the distance to the patch
/// from below.
#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
pub(crate) fn patch_boxes(
    u_breaks: &[f64],
    v_breaks: &[f64],
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    u_flat: &[f64],
    v_flat: &[f64],
    poles_homog: &[f64],
    n_u: usize,
    n_v: usize,
    h: usize,
) -> Vec<[f64; 6]> {
    if u_degree.max(v_degree) < curve::SMALL {
        boxes::<{ curve::SMALL }>(
            u_breaks,
            v_breaks,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
        )
    } else {
        boxes::<{ curve::LARGE }>(
            u_breaks,
            v_breaks,
            u_degree,
            v_degree,
            u_periodic,
            v_periodic,
            u_flat,
            v_flat,
            poles_homog,
            n_u,
            n_v,
            h,
        )
    }
}

#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
fn boxes<const N: usize>(
    u_breaks: &[f64],
    v_breaks: &[f64],
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    u_flat: &[f64],
    v_flat: &[f64],
    poles_homog: &[f64],
    n_u: usize,
    n_v: usize,
    h: usize,
) -> Vec<[f64; 6]> {
    // The window at a span's midpoint selects exactly the poles of that
    // span's patch.
    let span_window = |w: &[f64], degree, periodic, flat, n| {
        let mut out = curve::Window::<N>::new();
        curve::window(0.5 * (w[0] + w[1]), degree, periodic, flat, n, 0, &mut out);
        out
    };
    let v_windows: Vec<curve::Window<N>> = v_breaks
        .windows(2)
        .map(|w| span_window(w, v_degree, v_periodic, v_flat, n_v))
        .collect();
    let mut out = Vec::with_capacity((u_breaks.len() - 1) * v_windows.len());
    for w in u_breaks.windows(2) {
        let wu = span_window(w, u_degree, u_periodic, u_flat, n_u);
        for wv in &v_windows {
            let mut b = [
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for i in 0..=u_degree {
                let row = wu.pole(i) * n_v;
                for j in 0..=v_degree {
                    let base = (row + wv.pole(j)) * h;
                    let inv = if h == 4 {
                        1.0 / poles_homog[base + 3]
                    } else {
                        1.0
                    };
                    for c in 0..3 {
                        let x = poles_homog[base + c] * inv;
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
/// [`curve::bezier_matrix`] (row-major, `(degree + 1)^2` values), first
/// pole index, degree and pole count.
pub(crate) type SpanMatrix<'a> = (&'a [f64], usize, usize, usize);

/// Bernstein control net of a knot-span patch over its own parameter
/// rectangle, written row-major `[k_u][k_v][h]` to `out`
/// (`(u_degree + 1) * (v_degree + 1) * h` values, homogeneous when
/// `h == 4`); `scratch` holds as many.
///
/// The tensor product of each direction's span matrix applied to the
/// patch's poles, the pole indices wrapping modulo the pole count (a
/// periodic direction), so seams come out exactly as the evaluator's.
pub(crate) fn patch_bezier(
    u: SpanMatrix,
    v: SpanMatrix,
    poles_homog: &[f64],
    h: usize,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    match (h, u.2.max(v.2) < curve::SMALL) {
        (4, true) => bezier_net::<4, { curve::SMALL }>(u, v, poles_homog, scratch, out),
        (4, false) => bezier_net::<4, { curve::LARGE }>(u, v, poles_homog, scratch, out),
        (_, true) => bezier_net::<3, { curve::SMALL }>(u, v, poles_homog, scratch, out),
        (_, false) => bezier_net::<3, { curve::LARGE }>(u, v, poles_homog, scratch, out),
    }
}

/// [`patch_bezier`] with `H` values per point, for degrees below `N`.
fn bezier_net<const H: usize, const N: usize>(
    (mu, first_u, p, n_u): SpanMatrix,
    (mv, first_v, q, n_v): SpanMatrix,
    poles_homog: &[f64],
    scratch: &mut [f64],
    out: &mut [f64],
) {
    let (m, n) = (p + 1, q + 1);
    let point = |net: &[f64], k: usize| -> [f64; H] { std::array::from_fn(|c| net[k * H + c]) };
    // Along v first: `scratch[i][l]` is Bernstein point `l` of pole row `i`.
    for i in 0..m {
        let start = ((first_u + i) % n_u) * n_v;
        let mut row = [[0.0f64; H]; N];
        for j in 0..n {
            let pole = point(poles_homog, start + (first_v + j) % n_v);
            for (x, weights) in row.iter_mut().zip(mv.chunks_exact(n)) {
                for c in 0..H {
                    x[c] += weights[j] * pole[c];
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
                let y = dot(d, axes[a]);
                lo[a] = lo[a].min(y);
                hi[a] = hi[a].max(y);
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
            value[c] = value[c].hull(x[c]);
        }
    }
    // A rational patch is taken about its centre: `S - centre` has the
    // homogeneous coordinates `H - centre * w`.
    let centre = if h == 4 {
        value.map(Interval::mid)
    } else {
        [0.0; 3]
    };
    let at = |i: usize, j: usize, c: usize| {
        let k = (i * cols + j) * h;
        if c < 3 && h == 4 {
            net[k + c] - centre[c] * net[k + 3]
        } else {
            net[k + c]
        }
    };
    // Hulls of each homogeneous coordinate's difference nets, in the
    // order `[s, t, ss, st, tt]`, before the degree factors.
    let mut hulls = [[Interval::EMPTY; 4]; 5];
    let mut weight = Interval::EMPTY;
    for i in 0..=p {
        for j in 0..=q {
            #[allow(clippy::needless_range_loop)] // `c` indexes the net too
            for c in 0..h {
                let x = at(i, j, c);
                if i < p {
                    let ds = at(i + 1, j, c) - x;
                    hulls[0][c] = hulls[0][c].hull(ds);
                    if second && i + 1 < p {
                        hulls[2][c] = hulls[2][c].hull(at(i + 2, j, c) - at(i + 1, j, c) - ds);
                    }
                    if second && j < q {
                        hulls[3][c] = hulls[3][c].hull(at(i + 1, j + 1, c) - at(i, j + 1, c) - ds);
                    }
                }
                if j < q {
                    let dt = at(i, j + 1, c) - x;
                    hulls[1][c] = hulls[1][c].hull(dt);
                    if second && j + 1 < q {
                        hulls[4][c] = hulls[4][c].hull(at(i, j + 2, c) - at(i, j + 1, c) - dt);
                    }
                }
            }
            if h == 4 {
                weight = weight.hull(at(i, j, 3));
            }
        }
    }
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

/// Homogeneous `[H, Hu, Hv, Huu, Huv, Hvv]` at `(u, v)`, each in the first
/// `h` slots; the derivatives beyond `order` are zero.
#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
fn tensor<const N: usize>(
    u: f64,
    v: f64,
    u_degree: usize,
    v_degree: usize,
    u_periodic: bool,
    v_periodic: bool,
    u_flat: &[f64],
    v_flat: &[f64],
    poles_homog: &[f64],
    n_u: usize,
    n_v: usize,
    h: usize,
    order: usize,
) -> [[f64; 4]; 6] {
    let mut wu = curve::Window::<N>::new();
    let mut wv = curve::Window::<N>::new();
    curve::window(u, u_degree, u_periodic, u_flat, n_u, order, &mut wu);
    curve::window(v, v_degree, v_periodic, v_flat, n_v, order, &mut wv);
    // Fixed strides and orders let the loops unroll.
    match (h, order) {
        (3, 0) => sum::<N, 3, 0>(&wu, &wv, u_degree, v_degree, poles_homog, n_v),
        (3, 1) => sum::<N, 3, 1>(&wu, &wv, u_degree, v_degree, poles_homog, n_v),
        (3, _) => sum::<N, 3, 2>(&wu, &wv, u_degree, v_degree, poles_homog, n_v),
        (_, 0) => sum::<N, 4, 0>(&wu, &wv, u_degree, v_degree, poles_homog, n_v),
        (_, 1) => sum::<N, 4, 1>(&wu, &wv, u_degree, v_degree, poles_homog, n_v),
        _ => sum::<N, 4, 2>(&wu, &wv, u_degree, v_degree, poles_homog, n_v),
    }
}

/// `Σ_i Σ_j Nu_i Nv_j P_ij` over the local poles, with the mixed basis
/// derivatives up to total order `D` in the slots of [`tensor`].
fn sum<const N: usize, const H: usize, const D: usize>(
    wu: &curve::Window<N>,
    wv: &curve::Window<N>,
    u_degree: usize,
    v_degree: usize,
    poles: &[f64],
    n_v: usize,
) -> [[f64; 4]; 6] {
    let mut out = [[0.0f64; 4]; 6];
    for i in 0..=u_degree {
        let row = wu.pole(i) * n_v;
        // Along v: the value row and its v-derivatives up to order D.
        let mut along = [[0.0f64; H]; 3];
        for j in 0..=v_degree {
            let base = (row + wv.pole(j)) * H;
            let pole: &[f64; H] = poles[base..base + H].try_into().unwrap();
            for (k, acc) in along.iter_mut().enumerate().take(D + 1) {
                let n = wv.ders[k][j];
                for c in 0..H {
                    acc[c] += n * pole[c];
                }
            }
        }
        let nu = wu.ders[0][i];
        for c in 0..H {
            out[0][c] += nu * along[0][c];
        }
        if D >= 1 {
            let du = wu.ders[1][i];
            for c in 0..H {
                out[1][c] += du * along[0][c];
                out[2][c] += nu * along[1][c];
            }
        }
        if D >= 2 {
            let (du, duu) = (wu.ders[1][i], wu.ders[2][i]);
            for c in 0..H {
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

    // ---- bilinear patch by hand: the smallest tensor product ----

    // Degree-1 x degree-1 patch over the unit square with corner poles
    //   P00=(0,0,0) P01=(0,1,0)
    //   P10=(1,0,0) P11=(1,1,1)
    // (row-major [i_u][i_v]). Bilinear interpolation gives
    //   S(u,v) = (u, v, u*v), Su = (1, 0, v), Sv = (0, 1, u).
    fn bilinear_poles() -> Vec<f64> {
        vec![
            0.0, 0.0, 0.0, // P00
            0.0, 1.0, 0.0, // P01
            1.0, 0.0, 0.0, // P10
            1.0, 1.0, 1.0, // P11
        ]
    }

    #[test]
    fn test_bilinear_patch_point_by_hand() {
        let poles = bilinear_poles();
        let flat = curve::flat_knots(&[0.0, 1.0], &[2, 2]);
        let s = surface_d0(0.5, 0.5, 1, 1, false, false, &flat, &flat, &poles, 2, 2, 3);
        assert_close3(&[s.x, s.y, s.z], &[0.5, 0.5, 0.25]);
    }

    #[test]
    fn test_bilinear_patch_derivatives_by_hand() {
        let poles = bilinear_poles();
        let flat = curve::flat_knots(&[0.0, 1.0], &[2, 2]);
        // At (u, v) = (0.3, 0.7): S=(0.3,0.7,0.21), Su=(1,0,0.7), Sv=(0,1,0.3).
        let (s, su, sv) = surface_d1(0.3, 0.7, 1, 1, false, false, &flat, &flat, &poles, 2, 2, 3);
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
        let h = if weights.is_some() { 4 } else { 3 };

        // Pack row-major [i_u][i_v][h].
        let mut poles = Vec::with_capacity(n_u * n_v * h);
        for iu in 0..n_u {
            for iv in 0..n_v {
                let p = &pole_grid[iu][iv];
                match &weights {
                    Some(ws) => {
                        let w = ws[iu][iv];
                        poles.extend_from_slice(&[p[0] * w, p[1] * w, p[2] * w, w]);
                    }
                    None => poles.extend_from_slice(&p[0..3]),
                }
            }
        }

        let u_flat = curve::flat_knots(&u_knots, &u_mults);
        let v_flat = curve::flat_knots(&v_knots, &v_mults);

        for sample in case["samples"].as_array().unwrap() {
            let u = sample["u"].as_f64().unwrap();
            let v = sample["v"].as_f64().unwrap();

            let s0 = surface_d0(
                u, v, u_degree, v_degree, u_periodic, v_periodic, &u_flat, &v_flat, &poles, n_u,
                n_v, h,
            );
            let (s, su, sv) = surface_d1(
                u, v, u_degree, v_degree, u_periodic, v_periodic, &u_flat, &v_flat, &poles, n_u,
                n_v, h,
            );

            let [_, _, _, suu, suv, svv] = surface_d2(
                u, v, u_degree, v_degree, u_periodic, v_periodic, &u_flat, &v_flat, &poles, n_u,
                n_v, h,
            );
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
