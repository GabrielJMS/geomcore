//! Real roots of low-degree polynomials with multiplicity clustering.
//!
//! The input is first scaled *exactly* (powers of two on the variable and
//! on the coefficients) so the roots satisfy `|y| <= 2` and the leading
//! coefficient dominates: coefficients spanning hundreds of orders of
//! magnitude, or roots near `1e±150`, cost no accuracy. Only exact zeros
//! are trimmed (leading zeros lower the degree, trailing zeros are exact
//! roots at 0); a small nonzero leading coefficient is information (a
//! large root), never noise to discard.
//!
//! The scaled monic polynomial is split into real linear and quadratic
//! factors (Cardano/trigonometric cubic, Ferrari quartic) and the
//! factorization is refined by Newton on its coefficients, so complex
//! pairs hugging the real axis are resolved as well as real roots. A
//! quadratic factor whose two roots, real or complex, sit within the
//! clustering band reports one double root at the polynomial's critical
//! point; simple roots are Newton-polished on the exactly scaled input.
//! Near-multiple roots survive as multiplicity counts instead of
//! collapsing or vanishing — tangencies downstream depend on it.

use super::solve::solve_square;
use crate::InlineList;

/// Safety factor on the rounding-noise radius of a multiple root.
const NOISE_FACTOR: f64 = 4.0;

/// A real root with its multiplicity (clustered near-coincidences add up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RealRoot {
    /// Root location.
    pub(crate) value: f64,
    /// How many roots coincided here (1 = simple).
    pub(crate) multiplicity: u32,
}

/// Real roots of the ascending-coefficient polynomial `coeffs`, ordered,
/// with multiplicity.
///
/// Roots whose half-separation is within the band `tol * (1 + |x|)`
/// coalesce into one root (multiplicities add), and a complex pair whose
/// imaginary part is within the band reports as a real double root at its
/// real part: a tangency. The band is widened by the rounding-noise radius
/// of a multiple root (`(c*eps*sum|a_i||x|^i / |p^(k)(x)/k!|)^(1/k)`, `k >= 2`),
/// so exact multiple roots split by rounding still report as one, at any
/// scale. The band is in the units of `x`: callers whose variable is an
/// offset parameter (a line) should centre it near the geometry first.
///
/// Non-finite coefficients, the zero polynomial, and constants yield no
/// roots; so do degrees above 4 (callers needing them do not exist yet).
pub(crate) fn real_roots(coeffs: &[f64], tol: f64) -> Vec<RealRoot> {
    let Some((poly, cands, _)) = Scaled::solve(coeffs, tol) else {
        return Vec::new();
    };
    let roots = poly.finish(cands, tol.max(0.0));
    roots.value[..roots.len]
        .iter()
        .zip(&roots.mult)
        .filter_map(|(&y, &m)| poly.unscaled(y, m))
        .collect()
}

/// [`real_roots`] with geometric tangency: `touches(x)` says whether the
/// curve point at parameter `x` is within tolerance of the other shape.
///
/// A grazing contact sits at a critical point of the polynomial, and
/// input rounding may split its double root into two real roots or a
/// complex pair, either one wider than the band. The factorization
/// already locates both: the critical point between two neighbouring
/// roots, and the one near a complex pair's real part, are each one
/// bounded Newton search on the derivative. Where `touches` holds there,
/// a complex pair is a double root and neighbouring roots merge (a whole
/// chain of them when several neighbours touch, at its midpoint), with
/// their multiplicities summed.
pub(crate) fn real_roots_touching(
    coeffs: &[f64],
    tol: f64,
    touches: impl Fn(f64) -> bool,
) -> InlineList<RealRoot, 4> {
    let mut out = InlineList::new();
    let Some((poly, cands, pairs)) = Scaled::solve(coeffs, tol) else {
        return out;
    };
    let tol = tol.max(0.0);
    let touches_at = |y: f64| poly.unscaled(y, 1).is_some_and(|r| touches(r.value));
    let roots = poly.finish(cands, tol);
    // Roots, and complex pairs as either a touching double root at their
    // critical point or a separator (`mult == 0`), in increasing order.
    let mut items = [(0.0, 0u32); 4];
    let mut n = 0;
    for k in 0..roots.len {
        items[n] = (roots.value[k], roots.mult[k]);
        n += 1;
    }
    // A pair's critical point is searched for within half the distance to
    // every other item, so it is its own and not that of a neighbouring
    // contact. The narrower pair goes first: two pairs about the same real
    // part share its critical point, which belongs to the one nearer the
    // real axis.
    let mut order = [0, 1];
    if pairs.len == 2 && pairs.half[1] < pairs.half[0] {
        order = [1, 0];
    }
    for &k in &order[..pairs.len] {
        let m = pairs.centre[k];
        let reach = items[..n]
            .iter()
            .fold(pairs.half[k], |r, it| r.min(0.5 * (it.0 - m).abs()));
        let Some(y) = poly.extremum(m, m - reach, m + reach, tol) else {
            continue;
        };
        if items[..n]
            .iter()
            .all(|it| (it.0 - y).abs() > tol * (poly.unit + y.abs()))
        {
            items[n] = (y, if touches_at(y) { 2 } else { 0 });
            n += 1;
        }
    }
    let items = &mut items[..n];
    if pairs.len > 0 {
        items.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    }
    // Chains of items linked by a touching critical point between them.
    let mut k = 0;
    while k < n {
        let (first, mut mult) = items[k];
        if mult == 0 {
            k += 1;
            continue;
        }
        // Touching points: the links, and the double roots (their own
        // critical point). One alone is the contact; several, a chain
        // reported at its midpoint.
        let (mut touching, mut at) = (u32::from(mult >= 2), first);
        while k + 1 < n && items[k + 1].1 > 0 {
            let (lo, hi) = (items[k].0, items[k + 1].0);
            let link = poly
                .extremum(0.5 * (lo + hi), lo, hi, tol)
                .filter(|&y| lo < y && y < hi && touches_at(y));
            let Some(y) = link else {
                break;
            };
            k += 1;
            mult += items[k].1;
            touching += 1 + u32::from(items[k].1 >= 2);
            at = y;
        }
        let y = match touching {
            0 => first,
            1 => at,
            _ => 0.5 * (first + items[k].0),
        };
        if let Some(r) = poly.unscaled(y, mult) {
            out.push(r);
        }
        k += 1;
    }
    out
}

/// Up to four root candidates in the scaled variable.
#[derive(Default)]
struct Candidates {
    value: [f64; 4],
    mult: [u32; 4],
    len: usize,
}

impl Candidates {
    fn push(&mut self, value: f64, mult: u32) {
        if value.is_finite() && self.len < 4 {
            self.value[self.len] = value;
            self.mult[self.len] = mult;
            self.len += 1;
        }
    }
}

/// Complex pairs `centre ± i*half` beyond the band, in the scaled variable.
#[derive(Default)]
struct Pairs {
    centre: [f64; 2],
    half: [f64; 2],
    len: usize,
}

/// The input polynomial `a(x)` (exact zero roots removed, degree `d`) in
/// the scaled variable `y = x / 2^e`: `f` is `a` scaled exactly by powers
/// of two (`|f_i| <= |f_d|`, `|f_d|` in `[1, 2)`), and `g = f / f_d` is
/// monic.
struct Scaled {
    f: [f64; 5],
    g: [f64; 5],
    /// `log2|f_i|` (`-inf` for zero coefficients).
    lg: [f64; 5],
    d: usize,
    e: i32,
    /// `2^-e`: one unit of `x` measured in `y`.
    unit: f64,
}

impl Scaled {
    /// The scaled polynomial of `coeffs`, its root candidates and its
    /// complex pairs; `None` when there are no roots to find (non-finite
    /// coefficients, the zero polynomial, a constant, degree above 4).
    fn solve(coeffs: &[f64], tol: f64) -> Option<(Scaled, Candidates, Pairs)> {
        if coeffs.iter().any(|c| !c.is_finite()) {
            return None;
        }
        let mut n = coeffs.len();
        while n > 0 && coeffs[n - 1] == 0.0 {
            n -= 1;
        }
        if n <= 1 || n > 5 {
            return None;
        }
        let tol = tol.max(0.0);
        let zeros = coeffs.iter().take_while(|&&c| c == 0.0).count();
        let poly = Scaled::new(&coeffs[zeros..n]);
        let mut cands = Candidates::default();
        let mut pairs = Pairs::default();
        if zeros > 0 {
            cands.push(0.0, zeros as u32);
        }
        let factors = factorize(&poly, 0);
        for &r in &factors.lin[..factors.nlin] {
            cands.push(r, 1);
        }
        for &(al, be) in &factors.quad[..factors.nquad] {
            poly.push_quadratic(al, be, tol, &mut cands, &mut pairs);
        }
        Some((poly, cands, pairs))
    }

    /// The root `y` mapped back to `x`, if finite there.
    fn unscaled(&self, y: f64, multiplicity: u32) -> Option<RealRoot> {
        let value = ldexp(y, self.e);
        value.is_finite().then_some(RealRoot {
            value,
            multiplicity,
        })
    }

    fn new(a: &[f64]) -> Scaled {
        let d = a.len() - 1;
        let mut lg = [f64::NEG_INFINITY; 5];
        for (l, &c) in lg.iter_mut().zip(a) {
            *l = c.abs().log2();
        }
        let lead = lg[d];
        // Smallest power of two bounding every |a_i / a_d|^(1/(d-i)): the
        // Fujiwara root-bound scale, so all roots satisfy |y| <= 2.
        let mut ef = f64::NEG_INFINITY;
        for (i, &l) in lg[..d].iter().enumerate() {
            ef = ef.max((l - lead) / (d - i) as f64);
        }
        let e = if ef.is_finite() {
            ef.ceil().clamp(-4000.0, 4000.0) as i32
        } else {
            0
        };
        let top = lead.floor() as i32 + e * d as i32;
        let mut f = [0.0; 5];
        for (i, (fi, &ai)) in f.iter_mut().zip(a).enumerate() {
            let shift = e * i as i32 - top;
            *fi = ldexp(ai, shift);
            lg[i] += shift as f64;
        }
        let mut g = [0.0; 5];
        for i in 0..d {
            g[i] = f[i] / f[d];
        }
        g[d] = 1.0;
        Scaled {
            f,
            g,
            lg,
            d,
            e,
            unit: ldexp(1.0, -e),
        }
    }

    /// Clustering band at `y`: `tol * (1 + |x|)` in `y` units, plus the
    /// rounding-noise radius of a multiple root there.
    fn band(&self, y: f64, tol: f64) -> f64 {
        tol * (self.unit + y.abs()) + self.noise(y)
    }

    /// Whether a half-separation `half` about `y` is within the band,
    /// skipping the noise radius when even its crudest bound
    /// (`(budget / |f_d|)^(1/d)`, the `k = d` term) cannot close the gap.
    fn within_band(&self, half: f64, y: f64, tol: f64) -> bool {
        let base = tol * (self.unit + y.abs());
        if half <= base {
            return true;
        }
        let ay = y.abs();
        let mag = self.f[..=self.d]
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * ay + c.abs());
        let budget = NOISE_FACTOR * f64::EPSILON * mag;
        if (half - base).powi(self.d as i32) * self.f[self.d].abs() > budget {
            return false;
        }
        half <= base + self.noise(y)
    }

    /// Radius within which rounding (relative `NOISE_FACTOR * eps` on the
    /// coefficients) can move a multiple root at `y`: the smallest
    /// `(budget / |t_k|)^(1/k)` over the Taylor coefficients `t_k`, `k >= 2`.
    fn noise(&self, y: f64) -> f64 {
        let d = self.d;
        let t = taylor(&self.f, d, y);
        let ay = y.abs();
        let mag = self.f[..=d]
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * ay + c.abs());
        let budget = NOISE_FACTOR * f64::EPSILON * mag;
        let mut r = f64::INFINITY;
        for (k, &tk) in t.iter().enumerate().take(d + 1).skip(2) {
            if tk != 0.0 {
                let q = budget / tk.abs();
                r = r.min(match k {
                    2 => q.sqrt(),
                    3 => q.cbrt(),
                    _ => q.sqrt().sqrt(),
                });
            }
        }
        if r.is_finite() { r } else { 0.0 }
    }

    /// Classify the factor `y^2 + al*y + be`: a double root when its roots
    /// (real or complex) sit within the band of their midpoint, two simple
    /// real roots, or a complex pair.
    fn push_quadratic(&self, al: f64, be: f64, tol: f64, out: &mut Candidates, pairs: &mut Pairs) {
        let disc = discriminant(al, be);
        let mid = -0.5 * al;
        let half = 0.5 * disc.abs().sqrt();
        if self.within_band(half, mid, tol) {
            let band = self.band(mid, tol);
            out.push(self.critical_point(mid, mid - band, mid + band), 2);
        } else if disc > 0.0 {
            // Cancellation-free: q = -(al + sign(al)*sqrt(disc))/2.
            let q = -0.5 * (al + disc.sqrt().copysign(al));
            out.push(q, 1);
            out.push(be / q, 1);
        } else if mid.is_finite() && pairs.len < 2 {
            pairs.centre[pairs.len] = mid;
            pairs.half[pairs.len] = half;
            pairs.len += 1;
        }
    }

    /// Newton on `f'` from `y` (a double root's midpoint), staying within
    /// `[lo, hi]`: the tangency sits at the polynomial's extremum.
    fn critical_point(&self, mut y: f64, lo: f64, hi: f64) -> f64 {
        let (mut d1, mut d2) = derivs12(&self.f, self.d, y);
        for _ in 0..4 {
            if d1 == 0.0 || d2 == 0.0 {
                break;
            }
            let next = y - d1 / d2;
            if !(lo <= next && next <= hi) {
                break;
            }
            let (n1, n2) = derivs12(&self.f, self.d, next);
            if n1.abs() >= d1.abs() || n1.is_nan() {
                break;
            }
            (y, d1, d2) = (next, n1, n2);
        }
        y
    }

    /// [`Scaled::critical_point`] when Newton converges there, its last
    /// step within the band; `None` when `[lo, hi]` holds no extremum.
    fn extremum(&self, y: f64, lo: f64, hi: f64, tol: f64) -> Option<f64> {
        let y = self.critical_point(y, lo, hi);
        let (d1, d2) = derivs12(&self.f, self.d, y);
        (d1.abs() <= tol * (self.unit + y.abs()) * d2.abs()).then_some(y)
    }

    /// Newton-polish a simple root on `f`, never stepping beyond a quarter
    /// of the gap to its nearest neighbour nor accepting a larger residual.
    fn polish(&self, mut y: f64, gap: f64) -> f64 {
        let (mut fy, mut dy) = value_slope_compensated(&self.f, self.d, y);
        for _ in 0..4 {
            if fy == 0.0 || dy == 0.0 {
                break;
            }
            let step = fy / dy;
            if step.abs() > 0.25 * gap || step.is_nan() {
                break;
            }
            let next = y - step;
            if next == y {
                break;
            }
            let (nf, nd) = value_slope_compensated(&self.f, self.d, next);
            if nf.abs() >= fy.abs() || nf.is_nan() {
                break;
            }
            (y, fy, dy) = (next, nf, nd);
            if step.abs() <= f64::EPSILON * y.abs() {
                break;
            }
        }
        y
    }

    /// Polish simple roots and cluster within the band: the roots in
    /// increasing order, still in the scaled variable.
    fn finish(&self, c: Candidates, tol: f64) -> Candidates {
        let n = c.len;
        let mut order = [0usize, 1, 2, 3];
        order[..n].sort_unstable_by(|&i, &j| c.value[i].total_cmp(&c.value[j]));
        let vals: [f64; 4] = std::array::from_fn(|k| c.value[order[k]]);
        let mut value = vals;
        let mult: [u32; 4] = std::array::from_fn(|k| c.mult[order[k]]);
        for k in 0..n {
            if mult[k] == 1 {
                let below = if k > 0 {
                    vals[k] - vals[k - 1]
                } else {
                    f64::INFINITY
                };
                let above = if k + 1 < n {
                    vals[k + 1] - vals[k]
                } else {
                    f64::INFINITY
                };
                value[k] = self.polish(vals[k], below.min(above));
            }
        }
        // Polishing never crosses a neighbour, so the order holds; merge
        // runs whose half-gap is within the band of their midpoint.
        let mut out = Candidates::default();
        let mut k = 0;
        while k < n {
            let (mut y, mut m) = (value[k], mult[k]);
            while k + 1 < n
                && self.within_band(0.5 * (value[k + 1] - y), 0.5 * (value[k + 1] + y), tol)
            {
                k += 1;
                let total = m + mult[k];
                y = (y * m as f64 + value[k] * mult[k] as f64) / total as f64;
                m = total;
            }
            out.push(y, m);
            k += 1;
        }
        out
    }
}

/// Real factors of a monic polynomial: linear `(y - r)` and quadratic
/// `(y^2 + al*y + be)`.
#[derive(Default, Clone, Copy)]
struct Factors {
    lin: [f64; 4],
    nlin: usize,
    quad: [(f64, f64); 2],
    nquad: usize,
}

impl Factors {
    fn push_lin(&mut self, r: f64) {
        if self.nlin < 4 {
            self.lin[self.nlin] = r;
            self.nlin += 1;
        }
    }

    fn push_quad(&mut self, al: f64, be: f64) {
        if self.nquad < 2 {
            self.quad[self.nquad] = (al, be);
            self.nquad += 1;
        }
    }

    fn degree(&self) -> usize {
        self.nlin + 2 * self.nquad
    }
}

/// Root-magnitude gap (in bits) beyond which the roots are solved in two
/// groups: a single scaling resolves the large group to full accuracy but
/// leaves the small one at absolute (not relative) precision.
const SPLIT_BITS: f64 = 8.0;

/// Real factorization of `sc.g` in its scaled variable. When the Newton
/// polygon of the coefficients shows root magnitudes separated by more
/// than [`SPLIT_BITS`], the large-root factors (accurate at this scale)
/// are kept and the small roots are re-solved, at their own scale, from
/// the quotient left by dividing the large factors out (from the constant
/// term up, which is stable for large roots).
fn factorize(sc: &Scaled, depth: u32) -> Factors {
    let g = sc.g;
    let mut out = Factors::default();
    match sc.d {
        0 => {}
        1 => out.push_lin(-g[0]),
        2 => out.push_quad(g[1], g[0]),
        3 => {
            let (r, e, f) = cubic_factor(g[2], g[1], g[0]);
            out.push_lin(r);
            out.push_quad(e, f);
        }
        _ => {
            let quads = quartic_factor(g[3], g[2], g[1], g[0]);
            match isolated_root(&quads) {
                // A tight cluster of three roots cannot sit in one
                // quadratic: peel off the far root, solve the cluster as
                // a cubic.
                Some(r) => {
                    let (r, h2, h1, h0) = peel_root(g[3], g[2], g[1], g[0], r);
                    let (r2, e, f) = cubic_factor(h2, h1, h0);
                    out.push_lin(r);
                    out.push_lin(r2);
                    out.push_quad(e, f);
                }
                None => {
                    for (al, be) in quads {
                        out.push_quad(al, be);
                    }
                }
            }
        }
    }
    // A quadratic needs no split: its cancellation-free roots `q` and
    // `be / q` are each relative-accurate already.
    if depth >= 3 || sc.d <= 2 {
        return out;
    }
    let Some((small, threshold)) = magnitude_split(&sc.lg, sc.d) else {
        return out;
    };
    // The large group is the `d - small` largest roots: accurate at this
    // scale, while the small ones may be noise of any magnitude below.
    // Real quadratics are split, since they may straddle the gap.
    let mut units = [(0.0f64, 0.0f64, 0.0f64); 4]; // (magnitude, al, be); be NaN = linear root al
    let mut nu = 0;
    for &r in &out.lin[..out.nlin] {
        units[nu] = (r.abs(), r, f64::NAN);
        nu += 1;
    }
    for &(al, be) in &out.quad[..out.nquad] {
        let disc = discriminant(al, be);
        if disc > 0.0 {
            let q = -0.5 * (al + disc.sqrt().copysign(al));
            for r in [q, be / q] {
                units[nu] = (r.abs(), r, f64::NAN);
                nu += 1;
            }
        } else {
            units[nu] = (be.abs().sqrt(), al, be);
            nu += 1;
        }
    }
    units[..nu].sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let mut large = Factors::default();
    for &(m, al, be) in &units[..nu] {
        if large.degree() >= sc.d - small || m < threshold * f64::EPSILON || m.is_nan() {
            break;
        }
        if be.is_nan() {
            large.push_lin(al);
        } else {
            large.push_quad(al, be);
        }
    }
    if large.degree() != sc.d - small {
        return out;
    }
    // L = product of the large factors (monic, ascending).
    let mut l = [0.0; 5];
    l[0] = 1.0;
    let mut dl = 0;
    let mut mul = |f: &[f64]| {
        let mut next = [0.0; 5];
        for (i, &li) in l[..=dl].iter().enumerate() {
            for (j, &fj) in f.iter().enumerate() {
                next[i + j] += li * fj;
            }
        }
        l = next;
        dl += f.len() - 1;
    };
    for &r in &large.lin[..large.nlin] {
        mul(&[-r, 1.0]);
    }
    for &(al, be) in &large.quad[..large.nquad] {
        mul(&[be, al, 1.0]);
    }
    // g = L * S: solve for S from the low-order coefficients up.
    let mut s = [0.0; 5];
    for i in 0..=small {
        let mut acc = g[i];
        for j in 1..=i.min(dl) {
            acc -= l[j] * s[i - j];
        }
        s[i] = acc / l[0];
    }
    if !(s[..=small].iter().all(|v| v.is_finite()) && s[0] != 0.0 && s[small] != 0.0) {
        return out;
    }
    let sub = Scaled::new(&s[..=small]);
    let inner = factorize(&sub, depth + 1);
    let mut merged = large;
    for &r in &inner.lin[..inner.nlin] {
        merged.push_lin(ldexp(r, sub.e));
    }
    for &(al, be) in &inner.quad[..inner.nquad] {
        merged.push_quad(ldexp(al, sub.e), ldexp(be, 2 * sub.e));
    }
    merged
}

/// The topmost gap wider than [`SPLIT_BITS`] between consecutive
/// root-magnitude groups read off the Newton polygon (upper hull of
/// `(i, log2|f_i|)`): the number of roots below it and the magnitude
/// threshold (geometric middle of the gap).
fn magnitude_split(lg: &[f64; 5], d: usize) -> Option<(usize, f64)> {
    let mut hull = [(0usize, 0.0f64); 5];
    let mut h = 0;
    for (i, &l) in lg.iter().enumerate().take(d + 1) {
        if !l.is_finite() {
            continue;
        }
        let p = (i, l);
        while h >= 2 {
            let (a, b) = (hull[h - 2], hull[h - 1]);
            let cross = (b.0 - a.0) as f64 * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0) as f64;
            if cross >= 0.0 {
                h -= 1;
            } else {
                break;
            }
        }
        hull[h] = p;
        h += 1;
    }
    // Edge k spans hull[k]..hull[k+1]; its roots have log2-magnitude
    // (l_i - l_j) / (j - i), increasing along the hull.
    let mag = |k: usize| (hull[k].1 - hull[k + 1].1) / (hull[k + 1].0 - hull[k].0) as f64;
    // The topmost gap: the group above it is the one this scale resolves.
    (0..h.saturating_sub(2)).rev().find_map(|k| {
        let (lo, hi) = (mag(k), mag(k + 1));
        (hi - lo > SPLIT_BITS).then(|| (hull[k + 1].0, (0.5 * (lo + hi)).exp2()))
    })
}

/// `x * 2^k`, exact unless the result over/underflows.
#[inline]
fn ldexp(mut x: f64, mut k: i32) -> f64 {
    const STEP: i32 = 1000;
    if (-STEP..=STEP).contains(&k) {
        return x * pow2(k);
    }
    while k > STEP && x.is_finite() && x != 0.0 {
        x *= pow2(STEP);
        k -= STEP;
    }
    while k < -STEP && x.is_finite() && x != 0.0 {
        x *= pow2(-STEP);
        k += STEP;
    }
    x * pow2(k.clamp(-STEP, STEP))
}

/// `2^k` for `k` in the normal exponent range.
fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Value and first derivative of the degree-`d` ascending polynomial.
fn value_slope(p: &[f64; 5], d: usize, y: f64) -> (f64, f64) {
    let (mut v, mut s) = (p[d], 0.0);
    for &c in p[..d].iter().rev() {
        s = s * y + v;
        v = v * y + c;
    }
    (v, s)
}

/// Value (compensated Horner: as if evaluated in twice the working
/// precision) and first derivative of the degree-`d` ascending
/// polynomial. Newton on it converges past plain Horner's rounding noise.
fn value_slope_compensated(p: &[f64; 5], d: usize, y: f64) -> (f64, f64) {
    let (mut v, mut err, mut s) = (p[d], 0.0, 0.0);
    for &c in p[..d].iter().rev() {
        s = s * y + v;
        let (prod, prod_err) = two_prod(v, y);
        let sum = prod + c;
        let back = sum - prod;
        let sum_err = (prod - (sum - back)) + (c - back);
        err = err * y + (prod_err + sum_err);
        v = sum;
    }
    (v + err, s)
}

/// `a * b` and its exact rounding error (fma when the target has it,
/// Dekker's splitting otherwise: no software fma on the hot path).
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if cfg!(target_feature = "fma") {
        return (p, a.mul_add(b, -p));
    }
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let split = |x: f64| {
        let c = SPLIT * x;
        let hi = c - (c - x);
        (hi, x - hi)
    };
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// `al^2 - 4*be` with the square's rounding error recovered.
fn discriminant(al: f64, be: f64) -> f64 {
    let (sq, err) = two_prod(al, al);
    (sq - 4.0 * be) + err
}

/// First and second derivatives of the degree-`d` ascending polynomial.
fn derivs12(p: &[f64; 5], d: usize, y: f64) -> (f64, f64) {
    let (mut d1, mut d2) = (0.0, 0.0);
    for i in (1..=d).rev() {
        let k = i as f64;
        d1 = d1 * y + k * p[i];
        if i >= 2 {
            d2 = d2 * y + k * (k - 1.0) * p[i];
        }
    }
    (d1, d2)
}

/// Taylor coefficients of `p` about `y` (repeated synthetic division).
fn taylor(p: &[f64; 5], d: usize, y: f64) -> [f64; 5] {
    let mut t = *p;
    for k in 0..d {
        for i in (k..d).rev() {
            t[i] += y * t[i + 1];
        }
    }
    t
}

/// Newton on a cubic stored as `[c0, c1, c2, c3, 0]`, keeping only
/// residual-decreasing steps.
fn newton_cubic(p: &[f64; 5], mut y: f64) -> f64 {
    let (mut fy, mut dy) = value_slope(p, 3, y);
    for _ in 0..4 {
        if fy == 0.0 || dy == 0.0 {
            break;
        }
        let next = y - fy / dy;
        let (nf, nd) = value_slope(p, 3, next);
        if nf.abs() >= fy.abs() || nf.is_nan() {
            break;
        }
        (y, fy, dy) = (next, nf, nd);
    }
    y
}

/// Real roots (with repeats) of the monic cubic `y^3 + b*y^2 + c*y + d`
/// by Cardano (one real root) or the trigonometric form (three).
fn cubic_closed_form(b: f64, c: f64, d: f64) -> ([f64; 3], usize) {
    let shift = -b / 3.0;
    let p = c - b * b / 3.0;
    let q = d + b * (2.0 * b * b - 9.0 * c) / 27.0;
    let disc = (0.5 * q) * (0.5 * q) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    if disc > 0.0 || p >= 0.0 {
        // One real root: the larger-magnitude Cardano term first, the
        // other from their product -p/3 (no cancellation).
        let a = -(0.5 * q + disc.max(0.0).sqrt().copysign(q)).cbrt();
        let t = if a != 0.0 { a - p / (3.0 * a) } else { 0.0 };
        ([t + shift, 0.0, 0.0], 1)
    } else {
        let m = 2.0 * (-p / 3.0).sqrt();
        let cos3 = (1.5 * q / p * (-3.0 / p).sqrt()).clamp(-1.0, 1.0);
        let (sin, cos) = (cos3.acos() / 3.0).sin_cos();
        // cos(theta -+ 2pi/3) = -cos/2 +- sin*sqrt(3)/2.
        let (a, b) = (-0.5 * cos, 0.5 * 3f64.sqrt() * sin);
        (
            [m * cos + shift, m * (a + b) + shift, m * (a - b) + shift],
            3,
        )
    }
}

/// Newton on factor coefficients `x`. `res` maps `x` to each equation's
/// residual and the magnitude of its terms; residuals are weighted by
/// those magnitudes (rows equilibrated likewise), so equations carrying
/// tiny coefficients — the roots far smaller than the rest — converge as
/// tightly as the large ones. Only weighted-residual-decreasing steps are
/// kept.
fn refine<const N: usize>(
    mut x: [f64; N],
    res: impl Fn(&[f64; N]) -> ([f64; N], [f64; N]),
    jac: impl Fn(&[f64; N]) -> [[f64; N]; N],
) -> [f64; N] {
    let weighted = |(r, m): ([f64; N], [f64; N])| -> [f64; N] {
        std::array::from_fn(|i| if m[i] > 0.0 { r[i] / m[i] } else { r[i] })
    };
    let norm = |w: &[f64; N]| w.iter().map(|v| v.abs()).sum::<f64>();
    let (r0, m0) = res(&x);
    let mut mag = m0;
    let mut w = weighted((r0, m0));
    for _ in 0..8 {
        let current = norm(&w);
        // At the rounding floor already: a step could only shuffle noise.
        if !(current > 0.5 * f64::EPSILON && current.is_finite()) {
            break;
        }
        let mut j = jac(&x);
        for (row, &m) in j.iter_mut().zip(&mag) {
            if m > 0.0 {
                row.iter_mut().for_each(|v| *v /= m);
            }
        }
        // Only an exactly singular step is refused; a near-singular one is
        // left to the residual check below.
        let step = solve_square(j, w, 0.0).filter(|s| s.iter().all(|v| v.is_finite()));
        let Some(step) = step else {
            break;
        };
        let next: [f64; N] = std::array::from_fn(|i| x[i] - step[i]);
        let (nr, nm) = res(&next);
        let nw = weighted((nr, nm));
        let next_norm = norm(&nw);
        if next_norm >= current || next_norm.is_nan() {
            break;
        }
        (x, w, mag) = (next, nw, nm);
    }
    x
}

/// Factor the monic cubic `y^3 + b*y^2 + c*y + d` as
/// `(y - r) * (y^2 + e*y + f)` with `r` real, refined by Newton.
fn cubic_factor(b: f64, c: f64, d: f64) -> (f64, f64, f64) {
    let (roots, n) = cubic_closed_form(b, c, d);
    // The most isolated real root deflates best.
    let isolation = |i: usize| {
        (0..n)
            .filter(|&j| j != i)
            .map(|j| (roots[i] - roots[j]).abs())
            .fold(f64::INFINITY, f64::min)
    };
    let best = (0..n)
        .max_by(|&i, &j| isolation(i).total_cmp(&isolation(j)))
        .unwrap_or(0);
    let r = newton_cubic(&[d, c, b, 1.0, 0.0], roots[best]);
    // f (product of the other two roots) is relative-accurate as -d/r.
    let f = if r != 0.0 { -d / r } else { c };
    let x = refine(
        [r, b + r, f],
        |x| {
            (
                [x[1] - x[0] - b, x[2] - x[0] * x[1] - c, -x[0] * x[2] - d],
                [
                    x[1].abs() + x[0].abs() + b.abs(),
                    x[2].abs() + (x[0] * x[1]).abs() + c.abs(),
                    (x[0] * x[2]).abs() + d.abs(),
                ],
            )
        },
        |x| [[-1.0, 1.0, 0.0], [-x[1], -x[0], 1.0], [-x[2], 0.0, -x[0]]],
    );
    (x[0], x[1], x[2])
}

/// Factor the monic quartic `y^4 + b*y^3 + c*y^2 + d*y + e` into two real
/// quadratics `(y^2 + al*y + be) * (y^2 + ga*y + de)`: Ferrari for the
/// initial split, then Newton on the four factor coefficients.
fn quartic_factor(b: f64, c: f64, d: f64, e: f64) -> [(f64, f64); 2] {
    // Depress y = t + shift: t^4 + p*t^2 + q*t + r.
    let shift = -0.25 * b;
    let b2 = b * b;
    let p = c - 0.375 * b2;
    let q = d - 0.5 * b * c + 0.125 * b2 * b;
    let r = e - 0.25 * b * d + 0.0625 * b2 * c - 0.01171875 * b2 * b2;
    // Resolvent: the largest real root m of m^3 + 2p*m^2 + (p^2 - 4r)*m - q^2
    // (nonnegative: the cubic is -q^2 <= 0 at m = 0) gives u = sqrt(m).
    let resolvent = [-q * q, p * p - 4.0 * r, 2.0 * p, 1.0, 0.0];
    let (ms, n) = cubic_closed_form(resolvent[2], resolvent[1], resolvent[0]);
    let m_raw = ms[..n].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let m = newton_cubic(&resolvent, m_raw);
    let scale = p.abs() + r.abs().sqrt();
    // t-space factors (t^2 + u*t + v) and (t^2 - u*t + w).
    let (u, v, w) = if m > 1e-8 * scale && m.is_finite() {
        let u = m.sqrt();
        (u, 0.5 * (p + m - q / u), 0.5 * (p + m + q / u))
    } else {
        // Biquadratic (q ~ 0): (t^2 + v)(t^2 + w) with v + w = p, v*w = r.
        let disc = (p * p - 4.0 * r).max(0.0);
        let z = 0.5 * (p + disc.sqrt().copysign(p));
        (0.0, z, if z != 0.0 { r / z } else { 0.0 })
    };
    // Back to y (t = y - shift).
    let s2 = shift * shift;
    let x = refine(
        [
            u - 2.0 * shift,
            s2 - u * shift + v,
            -u - 2.0 * shift,
            s2 + u * shift + w,
        ],
        |x| {
            (
                [
                    x[0] + x[2] - b,
                    x[1] + x[3] + x[0] * x[2] - c,
                    x[0] * x[3] + x[1] * x[2] - d,
                    x[1] * x[3] - e,
                ],
                [
                    x[0].abs() + x[2].abs() + b.abs(),
                    x[1].abs() + x[3].abs() + (x[0] * x[2]).abs() + c.abs(),
                    (x[0] * x[3]).abs() + (x[1] * x[2]).abs() + d.abs(),
                    (x[1] * x[3]).abs() + e.abs(),
                ],
            )
        },
        |x| {
            [
                [1.0, 0.0, 1.0, 0.0],
                [x[2], 1.0, x[0], 1.0],
                [x[3], x[2], x[1], x[0]],
                [0.0, x[3], 0.0, x[1]],
            ]
        },
    );
    [(x[0], x[1]), (x[2], x[3])]
}

/// A real root of the two-quadratic split that is far from the other
/// three while those three are tightly clustered (spread under a quarter
/// of its distance to them): a near-triple root no quadratic pairing
/// resolves.
fn isolated_root(quads: &[(f64, f64); 2]) -> Option<f64> {
    // Roots as (re, im).
    let mut z = [(0.0f64, 0.0f64); 4];
    for (k, &(al, be)) in quads.iter().enumerate() {
        let disc = discriminant(al, be);
        if disc >= 0.0 {
            let q = -0.5 * (al + disc.sqrt().copysign(al));
            let other = if q != 0.0 { be / q } else { 0.0 };
            z[2 * k] = (q, 0.0);
            z[2 * k + 1] = (other, 0.0);
        } else {
            let half = 0.5 * (-disc).sqrt();
            z[2 * k] = (-0.5 * al, half);
            z[2 * k + 1] = (-0.5 * al, -half);
        }
    }
    // Squared distances: only compared, and the scaled roots are O(1).
    let dist = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
    let mut d2 = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in i + 1..4 {
            d2[i][j] = dist(z[i], z[j]);
            d2[j][i] = d2[i][j];
        }
    }
    let mut best: Option<(f64, f64)> = None;
    for i in 0..4 {
        if z[i].1 != 0.0 {
            continue;
        }
        let mut iso = f64::INFINITY;
        let mut spread = 0.0f64;
        for j in (0..4).filter(|&j| j != i) {
            iso = iso.min(d2[i][j]);
            for k in (j + 1..4).filter(|&k| k != i) {
                spread = spread.max(d2[j][k]);
            }
        }
        if 16.0 * spread < iso && best.is_none_or(|(b, _)| iso > b) {
            best = Some((iso, z[i].0));
        }
    }
    best.map(|(_, r)| r)
}

/// Split the monic quartic `y^4 + b*y^3 + c*y^2 + d*y + e` as
/// `(y - r) * (y^3 + h2*y^2 + h1*y + h0)` from an approximate root `r`,
/// refined by Newton on `(r, h2, h1, h0)`.
fn peel_root(b: f64, c: f64, d: f64, e: f64, r: f64) -> (f64, f64, f64, f64) {
    let h2 = b + r;
    let h1 = c + r * h2;
    let h0 = d + r * h1;
    let x = refine(
        [r, h2, h1, h0],
        |x| {
            (
                [
                    x[1] - x[0] - b,
                    x[2] - x[0] * x[1] - c,
                    x[3] - x[0] * x[2] - d,
                    -x[0] * x[3] - e,
                ],
                [
                    x[1].abs() + x[0].abs() + b.abs(),
                    x[2].abs() + (x[0] * x[1]).abs() + c.abs(),
                    x[3].abs() + (x[0] * x[2]).abs() + d.abs(),
                    (x[0] * x[3]).abs() + e.abs(),
                ],
            )
        },
        |x| {
            [
                [-1.0, 1.0, 0.0, 0.0],
                [-x[1], -x[0], 1.0, 0.0],
                [-x[2], 0.0, -x[0], 1.0],
                [-x[3], 0.0, 0.0, -x[0]],
            ]
        },
    );
    (x[0], x[1], x[2], x[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain Horner on ascending coefficients.
    fn eval(coeffs: &[f64], x: f64) -> f64 {
        coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    fn values(roots: &[RealRoot]) -> Vec<f64> {
        roots.iter().map(|r| r.value).collect()
    }

    /// Every expected root matched, in order, to relative accuracy `rel`.
    fn assert_roots(coeffs: &[f64], tol: f64, expected: &[(f64, u32)], rel: f64) {
        let roots = real_roots(coeffs, tol);
        assert_eq!(roots.len(), expected.len(), "{coeffs:?}: {roots:?}");
        for (r, &(v, m)) in roots.iter().zip(expected) {
            assert!(
                (r.value - v).abs() <= rel * v.abs().max(f64::MIN_POSITIVE),
                "{coeffs:?}: got {roots:?}, expected {expected:?}"
            );
            assert_eq!(r.multiplicity, m, "{coeffs:?}: {roots:?}");
        }
    }

    #[test]
    fn test_linear() {
        let roots = real_roots(&[2.0, 4.0], 1e-12);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value + 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_quadratic_two_simple() {
        // x^2 - 5x + 6 = (x-2)(x-3).
        let roots = real_roots(&[6.0, -5.0, 1.0], 1e-12);
        assert_eq!(roots.len(), 2);
        assert!((roots[0].value - 2.0).abs() < 1e-9);
        assert!((roots[1].value - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_quadratic_double_root() {
        // (x - 2)^2.
        let roots = real_roots(&[4.0, -4.0, 1.0], 1e-12);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value - 2.0).abs() < 1e-9);
        assert_eq!(roots[0].multiplicity, 2);
    }

    #[test]
    fn test_quadratic_no_real_roots() {
        let roots = real_roots(&[1.0, 0.0, 1.0], 1e-12);
        assert!(roots.is_empty());
    }

    #[test]
    fn test_cubic_three_real() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6.
        let roots = real_roots(&[-6.0, 11.0, -6.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 3);
        for (r, e) in roots.iter().zip([1.0, 2.0, 3.0]) {
            assert!((r.value - e).abs() < 1e-7, "{r:?}");
        }
    }

    #[test]
    fn test_cubic_one_real() {
        // x^3 + x + 1: single real root near -0.6823.
        let roots = real_roots(&[1.0, 1.0, 0.0, 1.0], 1e-12);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value + 0.682_327_803_828_019_3).abs() < 1e-9);
    }

    #[test]
    fn test_cubic_triple_root() {
        // (x + 1)^3 = x^3 + 3x^2 + 3x + 1.
        let roots = real_roots(&[1.0, 3.0, 3.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value + 1.0).abs() < 1e-6);
        assert_eq!(roots[0].multiplicity, 3, "{roots:?}");
    }

    #[test]
    fn test_cubic_double_plus_simple() {
        // (x - 1)^2 (x + 2) = x^3 - 3x + 2.
        let roots = real_roots(&[2.0, -3.0, 0.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 2);
        let dbl = roots.iter().find(|r| (r.value - 1.0).abs() < 1e-6).unwrap();
        assert_eq!(dbl.multiplicity, 2);
    }

    #[test]
    fn test_quartic_four_real() {
        // (x-1)(x-2)(x-3)(x-4).
        let roots = real_roots(&[24.0, -50.0, 35.0, -10.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 4);
        for (r, e) in roots.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((r.value - e).abs() < 1e-6, "{r:?}");
        }
    }

    #[test]
    fn test_quartic_no_real_roots() {
        // (x^2 + 1)^2.
        let roots = real_roots(&[1.0, 0.0, 2.0, 0.0, 1.0], 1e-12);
        assert!(roots.is_empty());
    }

    #[test]
    fn test_quartic_double_double() {
        // (x^2 - 1)^2 = x^4 - 2x^2 + 1: doubles at +-1, also when scaled.
        for k in [1.0, -4.241766773367914] {
            let coeffs = [k, 0.0, -2.0 * k, 0.0, k];
            assert_roots(&coeffs, 1e-7, &[(-1.0, 2), (1.0, 2)], 1e-7);
        }
    }

    #[test]
    fn test_quartic_quadruple_root() {
        // (x + 2)^4.
        let roots = real_roots(&[16.0, 32.0, 24.0, 8.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value + 2.0).abs() < 1e-6);
        assert_eq!(roots[0].multiplicity, 4, "{roots:?}");
    }

    #[test]
    fn test_quartic_two_real_two_complex() {
        // (x^2 + 1)(x - 1)(x - 5) = x^4 - 6x^3 + 6x^2 - 6x + 5.
        let roots = real_roots(&[5.0, -6.0, 6.0, -6.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 2);
        assert!((roots[0].value - 1.0).abs() < 1e-6);
        assert!((roots[1].value - 5.0).abs() < 1e-6);
    }

    /// Ascending coefficients of the product of two polynomials.
    fn product(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                out[i + j] += x * y;
            }
        }
        out
    }

    fn touching(coeffs: &[f64], touches: impl Fn(f64) -> bool) -> Vec<(f64, u32)> {
        real_roots_touching(coeffs, 1e-9, touches)
            .iter()
            .map(|r| (r.value, r.multiplicity))
            .collect()
    }

    fn assert_near(got: &[(f64, u32)], expected: &[(f64, u32)]) {
        assert_eq!(got.len(), expected.len(), "{got:?}");
        for (&(v, m), &(ev, em)) in got.iter().zip(expected) {
            assert!((v - ev).abs() < 1e-9, "{got:?}, expected {expected:?}");
            assert_eq!(m, em, "{got:?}");
        }
    }

    #[test]
    fn test_touching_merges_split_double_root() {
        // (x - 1 - d)(x - 1 + d)(x - 5)(x + 3): two roots 2e-5 apart, wider
        // than the band. Touching between them, they are one double root
        // at the critical point; otherwise they stay apart.
        let d = 1e-5;
        let p = product(&[-(1.0 + d), 1.0], &[-(1.0 - d), 1.0]);
        let p = product(&p, &product(&[-5.0, 1.0], &[3.0, 1.0]));
        let near_one = |x: f64| (x - 1.0).abs() < 1e-3;
        let got = touching(&p, near_one);
        assert_eq!(got.len(), 3, "{got:?}");
        assert!((got[1].0 - 1.0).abs() < 1e-6 && got[1].1 == 2, "{got:?}");
        assert_eq!(touching(&p, |_| false).len(), 4);
    }

    #[test]
    fn test_touching_reports_near_real_complex_pair() {
        // ((x - 1)^2 + d^2)(x - 5)(x + 3): a complex pair 1e-5 off the
        // real axis, a double root at 1 only when the curve touches there.
        let d = 1e-5;
        let p = product(
            &[1.0 + d * d, -2.0, 1.0],
            &product(&[-5.0, 1.0], &[3.0, 1.0]),
        );
        assert_near(
            &touching(&p, |x| (x - 1.0).abs() < 1e-3),
            &[(-3.0, 1), (1.0, 2), (5.0, 1)],
        );
        assert_near(&touching(&p, |_| false), &[(-3.0, 1), (5.0, 1)]);
    }

    #[test]
    fn test_touching_pairs_do_not_share_a_critical_point() {
        // (x^2 + d^2)(x^2 + 1/4): both pairs about 0; the contact there is
        // one double root, not two.
        let p = product(&[1e-10, 0.0, 1.0], &[0.25, 0.0, 1.0]);
        assert_near(&touching(&p, |_| true), &[(0.0, 2)]);
        // A pair beside a split double root, with no critical point of its
        // own (`h^2 > (c - 1)^2 / 8`), does not take the contact's: it
        // stays double, not a chain of four.
        let (d, c, h) = (1e-5, 1.02, 0.05);
        let p = product(&[-(1.0 + d), 1.0], &[-(1.0 - d), 1.0]);
        let p = product(&p, &[c * c + h * h, -2.0 * c, 1.0]);
        let got = touching(&p, |x| (x - 1.0).abs() < 1e-3);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!((got[0].0 - 1.0).abs() < 1e-6 && got[0].1 == 2, "{got:?}");
    }

    #[test]
    fn test_small_leading_coefficient_keeps_its_root() {
        // 1e-15 x^2 + 2x + 1: the tiny leading coefficient is a real
        // (huge) root, and the moderate root stays exact.
        assert_roots(&[1.0, 2.0, 1e-15], 1e-12, &[(-2e15, 1), (-0.5, 1)], 1e-12);
        // Exact zero leading coefficients lower the degree.
        assert_roots(&[1.0, 2.0, 0.0, 0.0], 1e-12, &[(-0.5, 1)], 1e-15);
    }

    #[test]
    fn test_trailing_zeros_are_exact_roots() {
        // x^2 (x - 3): exact double root at 0.
        assert_roots(&[0.0, 0.0, -3.0, 1.0], 1e-12, &[(0.0, 2), (3.0, 1)], 1e-15);
    }

    #[test]
    fn test_roots_satisfy_equation() {
        // Every reported root zeroes its polynomial to tolerance.
        let cases: Vec<Vec<f64>> = vec![
            vec![6.0, -5.0, 1.0],
            vec![-6.0, 11.0, -6.0, 1.0],
            vec![24.0, -50.0, 35.0, -10.0, 1.0],
            vec![5.0, -6.0, 6.0, -6.0, 1.0],
            vec![-2.0, 0.0, 3.0, 0.0, 1.0],
        ];
        for coeffs in &cases {
            for r in real_roots(coeffs, 1e-9) {
                assert!(eval(coeffs, r.value).abs() < 1e-6, "{coeffs:?} {r:?}");
            }
        }
    }

    #[test]
    fn test_wide_coefficient_range_biquadratic() {
        // (x^2 - 1)(x^2 - 1e8): the unit leading coefficient is tiny next
        // to the others but must not be dropped.
        let coeffs = [1e8, 0.0, -(1e8 + 1.0), 0.0, 1.0];
        assert_roots(
            &coeffs,
            1e-7,
            &[(-1e4, 1), (-1.0, 1), (1.0, 1), (1e4, 1)],
            1e-14,
        );
    }

    #[test]
    fn test_wide_coefficient_range_clustered_quartic() {
        // Two pairs of roots near +-2946 (coefficients from 1 to 7.5e13);
        // reference roots computed in 50-digit arithmetic.
        let coeffs = [
            75357446168894.52,
            28.9983424635691,
            -17361733.368892364,
            -2.273736754432321e-13,
            1.0000000000000004,
        ];
        let expected = [
            -2946.42549007165,
            -2946.236617325543,
            2946.267830797849,
            2946.394276599329,
        ];
        let roots = real_roots(&coeffs, 1e-7);
        assert_eq!(roots.len(), 4, "{roots:?}");
        for (r, e) in roots.iter().zip(expected) {
            assert!((r.value - e).abs() < 1e-9 * e.abs(), "{roots:?}");
            assert_eq!(r.multiplicity, 1);
        }
    }

    #[test]
    fn test_extreme_coefficient_magnitudes() {
        // (x-2)(x-3) scaled by 1e300 and 1e-300: no overflow, no false
        // double root from underflow.
        for k in [1e300, 1e-300] {
            assert_roots(&[6.0 * k, -5.0 * k, k], 1e-7, &[(2.0, 1), (3.0, 1)], 1e-14);
        }
        // Roots 1e-150 and 1e150.
        assert_roots(&[1.0, -1e150, 1.0], 1e-7, &[(1e-150, 1), (1e150, 1)], 1e-14);
        // (x-1)^2 (x-3)^2 scaled by 1e280.
        let c = [9.0, -24.0, 22.0, -8.0, 1.0].map(|v| v * 1e280);
        assert_roots(&c, 1e-7, &[(1.0, 2), (3.0, 2)], 1e-7);
    }

    #[test]
    fn test_non_finite_input_yields_nothing() {
        assert!(real_roots(&[f64::NAN, 1.0, 1.0], 1e-7).is_empty());
        assert!(real_roots(&[1.0, f64::INFINITY], 1e-7).is_empty());
        assert!(real_roots(&[0.0, 0.0, 0.0], 1e-7).is_empty());
        assert!(real_roots(&[3.0], 1e-7).is_empty());
    }

    #[test]
    fn test_close_roots_stay_distinct_below_band() {
        // (x-1)(x-1.001)(x-2) and roots 0, 1e-6, 1 resolve at tol 1e-7.
        assert_roots(
            &[-2.002, 5.003, -4.001, 1.0],
            1e-7,
            &[(1.0, 1), (1.001, 1), (2.0, 1)],
            1e-11,
        );
        let roots = real_roots(&[0.0, 1e-6, -1.000001, 1.0], 1e-7);
        assert_eq!(values(&roots).len(), 3, "{roots:?}");
        assert!((roots[1].value - 1e-6).abs() < 1e-18, "{roots:?}");
    }

    #[test]
    fn test_scale_invariance() {
        // (x - k)(x - 2k)(x^2 + k^2) at any magnitude k: two simple roots
        // to full relative accuracy (tol 0: only the rounding-noise band).
        for k in [1e-60, 1e-3, 1.0, 1e3, 1e70] {
            let k2 = k * k;
            let coeffs = [2.0 * k2 * k2, -3.0 * k2 * k, 3.0 * k2, -3.0 * k, 1.0];
            assert_roots(&coeffs, 0.0, &[(k, 1), (2.0 * k, 1)], 1e-14);
        }
    }

    #[test]
    fn test_near_tangent_complex_pair_reports_double() {
        // (x - 3)^2 + 1e-14: complex pair 1e-7 off the axis, a tangency at
        // tol 1e-7; 1e-3 off the axis is a clear miss at tol 1e-12.
        let coeffs = [9.0 + 1e-14, -6.0, 1.0];
        assert_roots(&coeffs, 1e-7, &[(3.0, 2)], 1e-12);
        let wide = [9.0 + 1e-6, -6.0, 1.0];
        assert!(real_roots(&wide, 1e-12).is_empty());
    }
}
