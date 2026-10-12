//! PROTOTYPE — throwaway, lives only on branch `prototype/bezier-deflection-bound`.
//!
//! Question: can B-spline discretization carry a *proven* chord deviation,
//! and what does that cost in points and time compared with the
//! estimate-based subdivision CAD kernels usually use? Also: which layout
//! makes it vectorize and parallelize?
//!
//! Methods (per knot span, on its Bernstein control points):
//!   A  a priori   : k = ceil(sqrt(n(n-1) M / 8 eps)) uniform pieces, M the
//!                   largest second difference of the control points
//!                   (rational: a homogeneous variant, see `apriori`). Proven.
//!   B  hull       : de Casteljau halving until every control point is within
//!                   eps of the chord segment (convex hull). Proven, adaptive.
//!   C  midpoint   : halve until the curve midpoint is within eps of the
//!                   chord midpoint (estimate, like quasi-uniform deflection).
//!   C2 chord line : halve until the curve midpoint is within eps of the
//!                   chord line (estimate, like a curvature-deflection test).
//!   D  oracle     : halve until the TRUE deviation is within eps (reference
//!                   for the fewest points bisection can reach; slow).
//!
//! "true dev" is measured independently with `BSplineCurve3D::eval_point`:
//! dense samples plus golden-section refinement of the distance to each chord.
//!
//! Run: cargo run --release --example prototype_flatten

use geomcore::Point3D;
use geomcore::curves::BSplineCurve3D;
use std::hint::black_box;
use std::time::Instant;

type H = [f64; 4];
type V = [f64; 3];
const MAXN: usize = 26;

// ---------------------------------------------------------------- vectors

fn proj(h: &H) -> V {
    [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
}
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn dist_to_segment(p: V, a: V, b: V) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    norm(sub(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]))
}
fn dist_to_line(p: V, a: V, b: V) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    if l2 == 0.0 {
        return norm(sub(p, a));
    }
    let t = dot(sub(p, a), ab) / l2;
    norm(sub(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]))
}
fn p3(p: Point3D) -> V {
    [p.x, p.y, p.z]
}

// ---------------------------------------------------------------- pieces

#[derive(Clone)]
struct Piece {
    a: f64,
    b: f64,
    n: usize,
    rational: bool,
    cp: [H; MAXN],
}

fn pieces(curve: &BSplineCurve3D) -> Vec<Piece> {
    curve
        .prototype_bezier_spans()
        .into_iter()
        .map(|(a, b, pts)| {
            let mut cp = [[0.0; 4]; MAXN];
            cp[..pts.len()].copy_from_slice(&pts);
            Piece { a, b, n: pts.len() - 1, rational: curve.is_rational(), cp }
        })
        .collect()
}

fn bez_eval(cp: &[H], n: usize, s: f64) -> V {
    let mut t = [[0.0; 4]; MAXN];
    t[..=n].copy_from_slice(&cp[..=n]);
    for r in 1..=n {
        for i in 0..=n - r {
            for c in 0..4 {
                t[i][c] = (1.0 - s) * t[i][c] + s * t[i + 1][c];
            }
        }
    }
    proj(&t[0])
}

fn split_half(cp: &[H], n: usize) -> ([H; MAXN], [H; MAXN]) {
    let mut t = [[0.0; 4]; MAXN];
    t[..=n].copy_from_slice(&cp[..=n]);
    let (mut l, mut r) = ([[0.0; 4]; MAXN], [[0.0; 4]; MAXN]);
    l[0] = t[0];
    r[n] = t[n];
    for k in 1..=n {
        for i in 0..=n - k {
            for c in 0..4 {
                t[i][c] = 0.5 * (t[i][c] + t[i + 1][c]);
            }
        }
        l[k] = t[0];
        r[n - k] = t[n - k];
    }
    (l, r)
}

// ---------------------------------------------------------------- output

/// Structure-of-arrays polyline; `dev[i]` is the claimed deviation of the
/// segment `i -> i + 1` (NaN when the method only estimates).
#[derive(Default)]
struct Polyline {
    t: Vec<f64>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    dev: Vec<f64>,
}
impl Polyline {
    fn push(&mut self, t: f64, p: V, dev: f64) {
        self.t.push(t);
        self.x.push(p[0]);
        self.y.push(p[1]);
        self.z.push(p[2]);
        self.dev.push(dev);
    }
    fn len(&self) -> usize {
        self.t.len()
    }
}

// ---------------------------------------------------------------- A: a priori

/// Pieces `k` and the proven deviation bound `K / (8 k^2)`.
///
/// Nonrational: `C(s) - L(s)` is bounded by `h^2 / 8 max|C''|`, and
/// `|C''| <= n (n - 1) max|D2 P_i|` over the Bernstein form.
/// Rational: with `x = w (C - c)` and `L` the 4D chord (whose projection
/// lies on the 3D chord), `|C - proj L| <= (|x - Lx| + R |w - Lw|) / w_min`,
/// R bounding the distance from `c` to the hull, so
/// `K = n (n - 1) (max|D2 x_i| + R max|D2 w_i|) / min w_i`.
fn apriori(p: &Piece, eps: f64) -> (usize, f64) {
    let n = p.n;
    if n < 2 {
        return (1, 0.0);
    }
    let nn = (n * (n - 1)) as f64;
    let cp = &p.cp[..=n];
    let k_const = if !p.rational {
        (0..n - 1)
            .map(|i| {
                let d: V = std::array::from_fn(|c| cp[i + 2][c] - 2.0 * cp[i + 1][c] + cp[i][c]);
                norm(d)
            })
            .fold(0.0, f64::max)
            * nn
    } else {
        let pts: Vec<V> = cp.iter().map(proj).collect();
        let c: V = std::array::from_fn(|j| pts.iter().map(|q| q[j]).sum::<f64>() / pts.len() as f64);
        let r = pts.iter().map(|q| norm(sub(*q, c))).fold(0.0, f64::max);
        let x: Vec<V> = cp.iter().map(|h| [h[0] - c[0] * h[3], h[1] - c[1] * h[3], h[2] - c[2] * h[3]]).collect();
        let mx = (0..n - 1)
            .map(|i| norm(std::array::from_fn(|j| x[i + 2][j] - 2.0 * x[i + 1][j] + x[i][j])))
            .fold(0.0, f64::max);
        let mw = (0..n - 1).map(|i| (cp[i + 2][3] - 2.0 * cp[i + 1][3] + cp[i][3]).abs()).fold(0.0, f64::max);
        let wmin = cp.iter().map(|h| h[3]).fold(f64::INFINITY, f64::min);
        nn * (mx + r * mw) / wmin
    };
    let k = ((k_const / (8.0 * eps)).sqrt().ceil() as usize).clamp(1, 1 << 20);
    (k, k_const / (8.0 * (k * k) as f64))
}

/// Monomial coefficients `coef[c][j] = binom(n, j) D^j P_0` (homogeneous).
fn power_basis(p: &Piece) -> [[f64; MAXN]; 4] {
    let n = p.n;
    let mut coef = [[0.0; MAXN]; 4];
    for c in 0..4 {
        let mut d = [0.0; MAXN];
        for i in 0..=n {
            d[i] = p.cp[i][c];
        }
        let mut binom = 1.0;
        for j in 0..=n {
            coef[c][j] = binom * d[0];
            for i in 0..n - j {
                d[i] = d[i + 1] - d[i];
            }
            binom = binom * (n - j) as f64 / (j + 1) as f64;
        }
    }
    coef
}

const LANES: usize = 8;

/// Writes the `k` points `s = j / k, j < k` of one piece into the slices,
/// Horner over coefficient-outer, point-inner loops (vectorizes).
fn fill_piece(p: &Piece, k: usize, dev: f64, t: &mut [f64], x: &mut [f64], y: &mut [f64], z: &mut [f64], d: &mut [f64]) {
    let coef = power_basis(p);
    let n = p.n;
    let inv = 1.0 / k as f64;
    let mut start = 0;
    while start < k {
        let m = (k - start).min(LANES);
        let mut s = [0.0; LANES];
        for (j, sj) in s.iter_mut().enumerate() {
            *sj = (start + j) as f64 * inv;
        }
        let mut acc = [[0.0; LANES]; 4];
        let dims = if p.rational { 4 } else { 3 };
        for c in 0..dims {
            let a = &mut acc[c];
            a.fill(coef[c][n]);
            for mm in (0..n).rev() {
                let cm = coef[c][mm];
                for j in 0..LANES {
                    a[j] = a[j] * s[j] + cm;
                }
            }
        }
        if !p.rational {
            acc[3].fill(1.0);
        }
        for j in 0..m {
            let iw = 1.0 / acc[3][j];
            t[start + j] = p.a + (p.b - p.a) * s[j];
            x[start + j] = acc[0][j] * iw;
            y[start + j] = acc[1][j] * iw;
            z[start + j] = acc[2][j] * iw;
            d[start + j] = dev;
        }
        start += m;
    }
}

fn apriori_counts(ps: &[Piece], eps: f64, ks: &mut Vec<(usize, f64)>) -> usize {
    ks.clear();
    ks.extend(ps.iter().map(|p| apriori(p, eps)));
    ks.iter().map(|k| k.0).sum::<usize>() + 1
}

/// Fills exactly `apriori_counts` points into preallocated slices.
fn apriori_fill(ps: &[Piece], ks: &[(usize, f64)], t: &mut [f64], x: &mut [f64], y: &mut [f64], z: &mut [f64], d: &mut [f64]) {
    let mut off = 0;
    for (p, &(k, dev)) in ps.iter().zip(ks) {
        let e = off + k;
        fill_piece(p, k, dev, &mut t[off..e], &mut x[off..e], &mut y[off..e], &mut z[off..e], &mut d[off..e]);
        off = e;
    }
    let last = ps.last().unwrap();
    let q = proj(&last.cp[last.n]);
    (t[off], x[off], y[off], z[off], d[off]) = (last.b, q[0], q[1], q[2], f64::NAN);
}

fn method_a(ps: &[Piece], eps: f64) -> Polyline {
    let mut ks = Vec::new();
    let total = apriori_counts(ps, eps, &mut ks);
    let mut out = Polyline {
        t: vec![0.0; total],
        x: vec![0.0; total],
        y: vec![0.0; total],
        z: vec![0.0; total],
        dev: vec![0.0; total],
    };
    apriori_fill(ps, &ks, &mut out.t, &mut out.x, &mut out.y, &mut out.z, &mut out.dev);
    out
}

/// Same parameters as A, points by de Casteljau one at a time (layout baseline).
fn method_a_casteljau(ps: &[Piece], eps: f64) -> Polyline {
    let mut out = Polyline::default();
    for p in ps {
        let (k, dev) = apriori(p, eps);
        for j in 0..k {
            let s = j as f64 / k as f64;
            out.push(p.a + (p.b - p.a) * s, bez_eval(&p.cp, p.n, s), dev);
        }
    }
    let last = ps.last().unwrap();
    out.push(last.b, proj(&last.cp[last.n]), f64::NAN);
    out
}

// ---------------------------------------------------------------- E: split, then a priori

/// Halves a piece while its halves' a priori counts sum to clearly less
/// than its own; leaves keep their uniform count. Still proven, still counted
/// before any point is evaluated.
fn split_leaves(p: &Piece, eps: f64, depth: u32, leaves: &mut Vec<(Piece, usize, f64)>) {
    let (k, dev) = apriori(p, eps);
    if k <= 2 || depth >= 30 {
        leaves.push((p.clone(), k, dev));
        return;
    }
    let (l, r) = split_half(&p.cp, p.n);
    let tm = 0.5 * (p.a + p.b);
    let lp = Piece { a: p.a, b: tm, cp: l, ..p.clone() };
    let rp = Piece { a: tm, b: p.b, cp: r, ..p.clone() };
    let (kl, kr) = (apriori(&lp, eps).0, apriori(&rp, eps).0);
    if (kl + kr) as f64 > 0.85 * k as f64 {
        leaves.push((p.clone(), k, dev));
        return;
    }
    split_leaves(&lp, eps, depth + 1, leaves);
    split_leaves(&rp, eps, depth + 1, leaves);
}

fn method_e(ps: &[Piece], eps: f64) -> Polyline {
    let mut leaves = Vec::new();
    for p in ps {
        split_leaves(p, eps, 0, &mut leaves);
    }
    let (lp, ks): (Vec<Piece>, Vec<(usize, f64)>) = leaves.into_iter().map(|(p, k, d)| (p, (k, d))).unzip();
    let total = ks.iter().map(|k| k.0).sum::<usize>() + 1;
    let mut out = Polyline {
        t: vec![0.0; total],
        x: vec![0.0; total],
        y: vec![0.0; total],
        z: vec![0.0; total],
        dev: vec![0.0; total],
    };
    apriori_fill(&lp, &ks, &mut out.t, &mut out.x, &mut out.y, &mut out.z, &mut out.dev);
    out
}

// ---------------------------------------------------------------- B: hull

fn hull_rec(cp: &[H; MAXN], n: usize, t0: f64, t1: f64, eps: f64, depth: u32, out: &mut Polyline) {
    let a = proj(&cp[0]);
    let b = proj(&cp[n]);
    let flat = (1..n).map(|i| dist_to_segment(proj(&cp[i]), a, b)).fold(0.0, f64::max);
    if flat <= eps || depth >= 50 {
        out.push(t0, a, flat);
        return;
    }
    let (l, r) = split_half(cp, n);
    let tm = 0.5 * (t0 + t1);
    hull_rec(&l, n, t0, tm, eps, depth + 1, out);
    hull_rec(&r, n, tm, t1, eps, depth + 1, out);
}

fn method_b_into(ps: &[Piece], eps: f64, out: &mut Polyline) {
    for p in ps {
        hull_rec(&p.cp, p.n, p.a, p.b, eps, 0, out);
    }
    let last = ps.last().unwrap();
    out.push(last.b, proj(&last.cp[last.n]), f64::NAN);
}

fn method_b(ps: &[Piece], eps: f64) -> Polyline {
    let mut out = Polyline::default();
    method_b_into(ps, eps, &mut out);
    out
}

// ---------------------------------------------------------------- C, C2: estimates

#[allow(clippy::too_many_arguments)]
fn mid_rec(p: &Piece, s0: f64, s1: f64, a: V, b: V, eps: f64, line: bool, depth: u32, out: &mut Polyline) {
    let sm = 0.5 * (s0 + s1);
    let m = bez_eval(&p.cp, p.n, sm);
    let d = if line {
        dist_to_line(m, a, b)
    } else {
        norm(sub(m, [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5]))
    };
    // Every span split at least once, as the real algorithms seed a few points.
    if (d <= eps && depth >= 1) || depth >= 50 {
        out.push(p.a + (p.b - p.a) * s0, a, f64::NAN);
        return;
    }
    mid_rec(p, s0, sm, a, m, eps, line, depth + 1, out);
    mid_rec(p, sm, s1, m, b, eps, line, depth + 1, out);
}

fn method_c(ps: &[Piece], eps: f64, line: bool) -> Polyline {
    let mut out = Polyline::default();
    for p in ps {
        mid_rec(p, 0.0, 1.0, proj(&p.cp[0]), proj(&p.cp[p.n]), eps, line, 0, &mut out);
    }
    let last = ps.last().unwrap();
    out.push(last.b, proj(&last.cp[last.n]), f64::NAN);
    out
}

// ---------------------------------------------------------------- truth

/// The largest distance from the curve over `[t0, t1]` to its chord.
fn true_dev(c: &BSplineCurve3D, t0: f64, t1: f64) -> f64 {
    let (a, b) = (p3(c.eval_point(t0)), p3(c.eval_point(t1)));
    let f = |t: f64| dist_to_segment(p3(c.eval_point(t)), a, b);
    const N: usize = 48;
    let ts: Vec<f64> = (0..=N).map(|i| t0 + (t1 - t0) * i as f64 / N as f64).collect();
    let (mut best, mut bi) = (0.0, 0);
    for (i, &t) in ts.iter().enumerate() {
        let v = f(t);
        if v > best {
            (best, bi) = (v, i);
        }
    }
    let (mut lo, mut hi) = (ts[bi.saturating_sub(1)], ts[(bi + 1).min(N)]);
    let g = 0.5 * (5f64.sqrt() - 1.0);
    for _ in 0..60 {
        let (m1, m2) = (hi - g * (hi - lo), lo + g * (hi - lo));
        if f(m1) > f(m2) { hi = m2 } else { lo = m1 }
    }
    best.max(f(0.5 * (lo + hi)))
}

fn method_d(c: &BSplineCurve3D, ps: &[Piece], eps: f64) -> Polyline {
    fn rec(c: &BSplineCurve3D, t0: f64, t1: f64, eps: f64, depth: u32, out: &mut Polyline) {
        let d = true_dev(c, t0, t1);
        if d <= eps || depth >= 40 {
            out.push(t0, p3(c.eval_point(t0)), d);
            return;
        }
        let tm = 0.5 * (t0 + t1);
        rec(c, t0, tm, eps, depth + 1, out);
        rec(c, tm, t1, eps, depth + 1, out);
    }
    let mut out = Polyline::default();
    for p in ps {
        rec(c, p.a, p.b, eps, 0, &mut out);
    }
    let last = ps.last().unwrap();
    out.push(last.b, p3(c.eval_point(last.b)), f64::NAN);
    out
}

struct Check {
    worst: f64,
    violations: usize,
    bound_broken: usize,
    point_err: f64,
}

fn check(c: &BSplineCurve3D, pl: &Polyline, eps: f64) -> Check {
    let mut ch = Check { worst: 0.0, violations: 0, bound_broken: 0, point_err: 0.0 };
    for i in 0..pl.len() {
        let q = p3(c.eval_point(pl.t[i]));
        ch.point_err = ch.point_err.max(norm(sub(q, [pl.x[i], pl.y[i], pl.z[i]])));
    }
    for i in 0..pl.len() - 1 {
        let d = true_dev(c, pl.t[i], pl.t[i + 1]);
        ch.worst = ch.worst.max(d / eps);
        if d > eps * (1.0 + 1e-9) {
            ch.violations += 1;
        }
        if !pl.dev[i].is_nan() && d > pl.dev[i] * (1.0 + 1e-9) + 1e-14 {
            ch.bound_broken += 1;
        }
    }
    ch
}

fn time_ns<F: FnMut() -> Polyline>(mut f: F) -> f64 {
    let mut reps = 1usize;
    loop {
        let start = Instant::now();
        for _ in 0..reps {
            black_box(f());
        }
        let el = start.elapsed().as_secs_f64();
        if el > 0.05 {
            return el * 1e9 / reps as f64;
        }
        reps *= 2;
    }
}

// ---------------------------------------------------------------- curves

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn clamped(degree: usize, n_poles: usize) -> (Vec<f64>, Vec<u32>) {
    let inner = n_poles - degree - 1;
    let knots: Vec<f64> = (0..=inner + 1).map(|i| i as f64 / (inner + 1) as f64).collect();
    let mut mults = vec![1u32; inner + 2];
    mults[0] = degree as u32 + 1;
    *mults.last_mut().unwrap() = degree as u32 + 1;
    (knots, mults)
}

fn wiggly(seed: u64, n_poles: usize, degree: usize) -> BSplineCurve3D {
    let mut r = Lcg(seed);
    let poles = (0..n_poles)
        .map(|i| Point3D::new(i as f64 * 0.25, r.next() - 0.5, 0.5 * (r.next() - 0.5)))
        .collect();
    let (k, m) = clamped(degree, n_poles);
    BSplineCurve3D::new(degree, poles, k, m, false).unwrap()
}

fn curves() -> Vec<(&'static str, BSplineCurve3D)> {
    let pt = |x, y, z| Point3D::new(x, y, z);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let circle_poles = vec![
        pt(1.0, 0.0, 0.0), pt(1.0, 1.0, 0.0), pt(0.0, 1.0, 0.0), pt(-1.0, 1.0, 0.0), pt(-1.0, 0.0, 0.0),
        pt(-1.0, -1.0, 0.0), pt(0.0, -1.0, 0.0), pt(1.0, -1.0, 0.0), pt(1.0, 0.0, 0.0),
    ];
    let wild_poles: Vec<Point3D> = (0..8).map(|i| pt(i as f64, ((i * 7) % 5) as f64 - 2.0, (i % 3) as f64)).collect();
    let (wk, wm) = clamped(3, 8);
    let (k7, m7) = clamped(7, 30);
    let mut r = Lcg(7);
    let p7 = (0..30).map(|i| pt(i as f64 * 0.3, r.next() - 0.5, r.next() - 0.5)).collect();
    vec![
        ("cubic, 6 poles", BSplineCurve3D::new(3,
            vec![pt(0.0, 0.0, 0.0), pt(1.0, 2.0, 0.0), pt(2.0, 2.0, 1.0), pt(3.0, 0.0, 1.0), pt(4.0, 1.0, 0.0), pt(5.0, 0.0, 0.0)],
            vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4], false).unwrap()),
        ("S-curve, midpoint on chord", BSplineCurve3D::new(3,
            vec![pt(0.0, 0.0, 0.0), pt(1.0, 1.0, 0.0), pt(2.0, -1.0, 0.0), pt(3.0, 0.0, 0.0)],
            vec![0.0, 1.0], vec![4, 4], false).unwrap()),
        ("wiggly cubic, 40 poles", wiggly(42, 40, 3)),
        ("quintic, clustered knots", BSplineCurve3D::new(5,
            (0..10).map(|i| pt(i as f64, ((i * 3) % 4) as f64 * 0.5, (i % 2) as f64 * 0.3)).collect(),
            vec![0.0, 0.01, 0.02, 0.5, 0.9, 1.0], vec![6, 1, 1, 1, 1, 6], false).unwrap()),
        ("degree 7, 30 poles", BSplineCurve3D::new(7, p7, k7, m7, false).unwrap()),
        ("uneven speed cubic", BSplineCurve3D::new(3,
            vec![pt(0.0, 0.0, 0.0), pt(0.01, 0.0, 0.0), pt(0.02, 0.01, 0.0), pt(5.0, 3.0, 0.0), pt(10.0, 0.0, 0.0), pt(10.01, 0.01, 0.0)],
            vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4], false).unwrap()),
        ("rational circle r=1, 9 poles", BSplineCurve3D::new_rational(2, circle_poles,
            vec![1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0], vec![0.0, 0.25, 0.5, 0.75, 1.0], vec![3, 2, 2, 2, 3], false).unwrap()),
        ("rational cubic, weights 0.2..8", BSplineCurve3D::new_rational(3, wild_poles,
            vec![1.0, 8.0, 0.2, 1.0, 5.0, 0.3, 2.0, 1.0], wk, wm, false).unwrap()),
    ]
}

fn size(c: &BSplineCurve3D) -> f64 {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in c.poles() {
        for (j, v) in p3(*p).into_iter().enumerate() {
            lo[j] = lo[j].min(v);
            hi[j] = hi[j].max(v);
        }
    }
    norm(sub(hi, lo))
}

// ---------------------------------------------------------------- batch

fn batch(threads: usize) {
    const COUNT: usize = 20_000;
    let curves: Vec<BSplineCurve3D> = (0..COUNT).map(|i| wiggly(i as u64 + 1, 40, 3)).collect();
    let all: Vec<Vec<Piece>> = curves.iter().map(pieces).collect(); // a cache, as the span tree is
    let eps = 1e-4;
    println!("\n== Batch: {COUNT} wiggly cubics (37 spans each), eps = {eps:e}, {threads} threads ==");

    // A, two passes: counts -> prefix sums -> one allocation -> fill disjoint slices.
    let run_a = |par: bool| -> (usize, f64) {
        let start = Instant::now();
        let mut ks: Vec<Vec<(usize, f64)>> = vec![Vec::new(); COUNT];
        let mut counts = vec![0usize; COUNT];
        if par {
            let chunk = COUNT.div_ceil(threads);
            std::thread::scope(|sc| {
                for ((ps, kk), cc) in all.chunks(chunk).zip(ks.chunks_mut(chunk)).zip(counts.chunks_mut(chunk)) {
                    sc.spawn(move || {
                        for ((p, k), c) in ps.iter().zip(kk).zip(cc) {
                            *c = apriori_counts(p, eps, k);
                        }
                    });
                }
            });
        } else {
            for ((p, k), c) in all.iter().zip(&mut ks).zip(&mut counts) {
                *c = apriori_counts(p, eps, k);
            }
        }
        let total: usize = counts.iter().sum();
        let (mut t, mut x, mut y, mut z, mut d) =
            (vec![0.0; total], vec![0.0; total], vec![0.0; total], vec![0.0; total], vec![0.0; total]);
        // Carve per-curve slices, then hand contiguous runs of curves to threads.
        let mut jobs = Vec::with_capacity(COUNT);
        {
            let (mut rt, mut rx, mut ry, mut rz, mut rd) = (&mut t[..], &mut x[..], &mut y[..], &mut z[..], &mut d[..]);
            for i in 0..COUNT {
                let n = counts[i];
                let (a, b) = rt.split_at_mut(n);
                rt = b;
                let (a2, b2) = rx.split_at_mut(n);
                rx = b2;
                let (a3, b3) = ry.split_at_mut(n);
                ry = b3;
                let (a4, b4) = rz.split_at_mut(n);
                rz = b4;
                let (a5, b5) = rd.split_at_mut(n);
                rd = b5;
                jobs.push((i, a, a2, a3, a4, a5));
            }
        }
        if par {
            let chunk = COUNT.div_ceil(threads);
            let ks = &ks;
            let all = &all;
            std::thread::scope(|sc| {
                for js in jobs.chunks_mut(chunk) {
                    sc.spawn(move || {
                        for (i, a, b, c, dd, e) in js.iter_mut() {
                            apriori_fill(&all[*i], &ks[*i], a, b, c, dd, e);
                        }
                    });
                }
            });
        } else {
            for (i, a, b, c, dd, e) in jobs.iter_mut() {
                apriori_fill(&all[*i], &ks[*i], a, b, c, dd, e);
            }
        }
        black_box((&t, &x, &y, &z, &d));
        (total, start.elapsed().as_secs_f64())
    };

    // B: counts unknown up front -> per-thread growth, then concatenation.
    let run_b = |par: bool| -> (usize, f64) {
        let start = Instant::now();
        let out = if par {
            let chunk = COUNT.div_ceil(threads);
            let parts: Vec<Polyline> = std::thread::scope(|sc| {
                let hs: Vec<_> = all
                    .chunks(chunk)
                    .map(|ps| {
                        sc.spawn(move || {
                            let mut o = Polyline::default();
                            for p in ps {
                                method_b_into(p, eps, &mut o);
                            }
                            o
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut o = Polyline::default();
            for p in parts {
                o.t.extend(p.t);
                o.x.extend(p.x);
                o.y.extend(p.y);
                o.z.extend(p.z);
                o.dev.extend(p.dev);
            }
            o
        } else {
            let mut o = Polyline::default();
            for p in &all {
                method_b_into(p, eps, &mut o);
            }
            o
        };
        (out.len(), start.elapsed().as_secs_f64())
    };

    for _ in 0..2 {
        black_box(run_a(false));
    } // warm up
    let best = |f: &dyn Fn() -> (usize, f64)| (0..5).map(|_| f()).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
    let ((na, ta), (nap, tap)) = (best(&|| run_a(false)), best(&|| run_a(true)));
    let ((nb, tb), (nbp, tbp)) = (best(&|| run_b(false)), best(&|| run_b(true)));
    println!("  {:<28}{:>10}{:>11}{:>13}{:>9}", "method", "points", "ms", "Mpoints/s", "speedup");
    println!("  {:<28}{:>10}{:>11.2}{:>13.1}{:>9}", "A a priori, serial", na, ta * 1e3, na as f64 / ta / 1e6, "1.0");
    println!("  {:<28}{:>10}{:>11.2}{:>13.1}{:>9.1}", "A a priori, parallel", nap, tap * 1e3, nap as f64 / tap / 1e6, ta / tap);
    println!("  {:<28}{:>10}{:>11.2}{:>13.1}{:>9}", "B hull, serial", nb, tb * 1e3, nb as f64 / tb / 1e6, "1.0");
    println!("  {:<28}{:>10}{:>11.2}{:>13.1}{:>9.1}", "B hull, parallel", nbp, tbp * 1e3, nbp as f64 / tbp / 1e6, tb / tbp);
}

// ---------------------------------------------------------------- main

fn main() {
    for rel in [1e-3, 1e-5] {
        println!("\n######## eps = {rel:e} x size (bounding-box diagonal of the poles) ########");
        for (name, c) in curves() {
            let eps = rel * size(&c);
            let ps = pieces(&c);
            println!(
                "\n== {name}  (degree {}, {} spans, eps = {eps:.3e}) ==",
                c.degree(),
                ps.len()
            );
            println!(
                "  {:<26}{:>8}{:>14}{:>12}{:>9}{:>12}{:>11}",
                "method", "points", "true max/eps", "violations", "bound ok", "ns / call", "ns / point"
            );
            let row = |label: &str, pl: Polyline, ns: Option<f64>, proven: bool| {
                let ch = check(&c, &pl, eps);
                assert!(ch.point_err < 1e-9 * size(&c), "{label}: point error {}", ch.point_err);
                let ns_s = ns.map_or("-".into(), |v| format!("{v:.0}"));
                let np_s = ns.map_or("-".into(), |v| format!("{:.1}", v / pl.len() as f64));
                let ok = if proven { if ch.bound_broken == 0 { "yes" } else { "NO" } } else { "n/a" };
                println!(
                    "  {:<26}{:>8}{:>14.3}{:>12}{:>9}{:>12}{:>11}",
                    label, pl.len(), ch.worst, ch.violations, ok, ns_s, np_s
                );
            };
            row("A a priori (Horner SoA)", method_a(&ps, eps), Some(time_ns(|| method_a(&ps, eps))), true);
            row("A a priori (de Casteljau)", method_a_casteljau(&ps, eps), Some(time_ns(|| method_a_casteljau(&ps, eps))), true);
            row("A params, eval_points", method_a(&ps, eps), Some(time_ns(|| {
                let a = method_a(&ps, eps);
                let pts = c.eval_points(&a.t);
                black_box(pts);
                a
            })), true);
            row("E split + a priori", method_e(&ps, eps), Some(time_ns(|| method_e(&ps, eps))), true);
            row("B hull adaptive", method_b(&ps, eps), Some(time_ns(|| method_b(&ps, eps))), true);
            row("C midpoint (estimate)", method_c(&ps, eps, false), Some(time_ns(|| method_c(&ps, eps, false))), false);
            row("C2 chord line (estimate)", method_c(&ps, eps, true), Some(time_ns(|| method_c(&ps, eps, true))), false);
            row("D oracle bisection", method_d(&c, &ps, eps), None, false);
            if name.starts_with("rational circle") {
                let n = (std::f64::consts::TAU / (2.0 * (1.0 - eps).acos())).ceil() as usize + 1;
                println!("  {:<26}{:>8}   (exact circle formula, uniform angle)", "reference", n);
            }
        }
    }
    let t0 = Instant::now();
    let curve = wiggly(1, 40, 3);
    let n = 10_000;
    for _ in 0..n {
        black_box(pieces(black_box(&curve)));
    }
    println!(
        "\nBernstein extraction (37 spans, allocating accessor): {:.0} ns per curve",
        t0.elapsed().as_secs_f64() * 1e9 / n as f64
    );
    batch(std::thread::available_parallelism().map_or(4, |n| n.get()));
}
