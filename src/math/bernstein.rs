//! Polynomials in Bernstein form over `[0, 1]`: products and root
//! bracketing by subdivision.
//!
//! The Bernstein basis suits B-spline pieces: a span's control points are
//! Bernstein coefficients, products stay in the basis, and the number of
//! sign changes of the coefficients bounds the number of roots (Descartes'
//! rule of signs), so de Casteljau halving brackets every root.
//!
//! Products work on the *scaled* coefficients `C(n, k) * c_k`, for which
//! multiplying polynomials is a plain convolution.

/// Coefficient capacity: products of B-spline pieces up to degree 25.
pub(crate) const CAPACITY: usize = 80;

/// Scales Bernstein coefficients in place: `c_k <- C(n, k) * c_k`.
pub(crate) fn scale(c: &mut [f64]) {
    let n = c.len() - 1;
    let mut binomial = 1.0;
    for (k, ck) in c.iter_mut().enumerate() {
        *ck *= binomial;
        binomial = binomial * (n - k) as f64 / (k + 1) as f64;
    }
}

/// Undoes [`scale`]: `c_k <- c_k / C(n, k)`.
pub(crate) fn unscale(c: &mut [f64]) {
    let n = c.len() - 1;
    let mut binomial = 1.0;
    for (k, ck) in c.iter_mut().enumerate() {
        *ck /= binomial;
        binomial = binomial * (n - k) as f64 / (k + 1) as f64;
    }
}

/// Adds `factor` times the product of the scaled polynomials `a` (degree
/// `m = a.len() - 1`) and `b` (degree `n`) to the scaled degree-`m + n`
/// polynomial `out`.
pub(crate) fn add_product(a: &[f64], b: &[f64], factor: f64, out: &mut [f64]) {
    for (i, &ai) in a.iter().enumerate() {
        let ai = factor * ai;
        for (o, &bj) in out[i..].iter_mut().zip(b) {
            *o += ai * bj;
        }
    }
}

/// Brackets the roots where the polynomial with (unscaled) Bernstein
/// coefficients `c` rises through zero: calls `visit(s0, s1, s)` for
/// subintervals of `[0, 1]` each holding one such root, `s` estimating it
/// (where the control polygon rises through zero). Coefficients
/// within `tiny` of zero count as zero, so a polynomial that is noise
/// throughout yields nothing. `N` bounds the coefficient count.
///
/// A subinterval whose coefficients change sign once holds exactly one
/// root; more changes are halved until they resolve. A root landing on a
/// cut, and roots too close to separate after `MAX_DEPTH` halvings (a
/// multiple root, say), are reported as possible rising roots.
pub(crate) fn rising_roots<const N: usize>(
    c: &[f64],
    tiny: f64,
    mut visit: impl FnMut(f64, f64, f64),
) {
    bracket::<N>(c, 0.0, 1.0, tiny, 0, &mut visit);
}

fn bracket<const N: usize>(
    c: &[f64],
    s0: f64,
    s1: f64,
    tiny: f64,
    depth: u32,
    visit: &mut impl FnMut(f64, f64, f64),
) {
    const MAX_DEPTH: u32 = 40;
    let mut signs = c.iter().filter(|x| x.abs() > tiny).map(|&x| x > 0.0);
    let Some(first) = signs.next() else {
        return;
    };
    let mut changes = 0;
    let mut previous = first;
    for s in signs {
        changes += usize::from(s != previous);
        previous = s;
    }
    match changes {
        0 => {}
        1 => {
            if !first {
                visit(s0, s1, s0 + (s1 - s0) * polygon_crossing(c, tiny));
            }
        }
        _ if depth == MAX_DEPTH => visit(s0, s1, 0.5 * (s0 + s1)),
        _ => {
            let n = c.len();
            let mut left = [0.0; N];
            let mut right = [0.0; N];
            halve(c, &mut left[..n], &mut right[..n]);
            let mid = 0.5 * (s0 + s1);
            // A root exactly at the cut is a zero end coefficient of both
            // halves, invisible to either: bracket it across the cut.
            if right[0].abs() <= tiny {
                visit(0.5 * (s0 + mid), 0.5 * (mid + s1), mid);
            }
            bracket::<N>(&left[..n], s0, mid, tiny, depth + 1, visit);
            bracket::<N>(&right[..n], mid, s1, tiny, depth + 1, visit);
        }
    }
}

/// Where the control polygon of `c`, with one sign change from negative
/// to positive, rises through zero (in `[0, 1]`).
fn polygon_crossing(c: &[f64], tiny: f64) -> f64 {
    let n = (c.len() - 1) as f64;
    let mut below = 0;
    for (k, &ck) in c.iter().enumerate() {
        if ck > tiny {
            let (y0, y1) = (c[below], ck);
            let x0 = below as f64 + (k - below) as f64 * (-y0 / (y1 - y0)).clamp(0.0, 1.0);
            return x0 / n;
        }
        if ck < -tiny {
            below = k;
        }
    }
    0.5
}

/// Restricts the Bernstein control points `c` of a polynomial piece over
/// `[0, 1]` to `[s0, s1]` (`0 <= s0 < s1 <= 1`), in place, rescaled to
/// `[0, 1]`: de Casteljau keeps the part right of `s0`, then the part of
/// that left of `s1`.
pub(crate) fn restrict<const D: usize>(c: &mut [[f64; D]], s0: f64, s1: f64) {
    let n = c.len() - 1;
    let lerp = |a: [f64; D], b: [f64; D], s: f64| std::array::from_fn(|k| a[k] + (b[k] - a[k]) * s);
    if s0 > 0.0 {
        for r in 1..=n {
            for i in 0..=n - r {
                c[i] = lerp(c[i], c[i + 1], s0);
            }
        }
    }
    if s1 < 1.0 {
        let s = (s1 - s0) / (1.0 - s0);
        for r in 1..=n {
            for i in (r..=n).rev() {
                c[i] = lerp(c[i - 1], c[i], s);
            }
        }
    }
}

/// De Casteljau subdivision at `1/2`: the coefficients of `c` over
/// `[0, 1/2]` and `[1/2, 1]`, each rescaled to `[0, 1]`.
fn halve(c: &[f64], left: &mut [f64], right: &mut [f64]) {
    let n = c.len();
    right.copy_from_slice(c);
    for (r, l) in left.iter_mut().enumerate() {
        *l = right[0];
        for j in 0..n - 1 - r {
            right[j] = 0.5 * (right[j] + right[j + 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluates Bernstein coefficients at `s` (de Casteljau).
    fn eval(c: &[f64], s: f64) -> f64 {
        let mut b = c.to_vec();
        for r in 1..b.len() {
            for j in 0..b.len() - r {
                b[j] = (1.0 - s) * b[j] + s * b[j + 1];
            }
        }
        b[0]
    }

    /// Unscaled Bernstein coefficients of the product of `a` and `b`.
    fn product(a: &[f64], b: &[f64]) -> Vec<f64> {
        let (mut a, mut b) = (a.to_vec(), b.to_vec());
        scale(&mut a);
        scale(&mut b);
        let mut out = vec![0.0; a.len() + b.len() - 1];
        add_product(&a, &b, 1.0, &mut out);
        unscale(&mut out);
        out
    }

    #[test]
    fn test_product_matches_pointwise_product() {
        let a = [1.0, -2.0, 0.5, 3.0];
        let b = [0.3, 4.0, -1.0];
        let out = product(&a, &b);
        for s in [0.0, 0.1, 0.45, 0.8, 1.0] {
            assert!((eval(&out, s) - eval(&a, s) * eval(&b, s)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_halve_preserves_the_polynomial() {
        let c = [2.0, -1.0, 0.25, 4.0, -3.0];
        let (mut left, mut right) = ([0.0; 5], [0.0; 5]);
        halve(&c, &mut left, &mut right);
        for s in [0.0, 0.3, 0.7, 1.0] {
            assert!((eval(&left, s) - eval(&c, 0.5 * s)).abs() < 1e-12);
            assert!((eval(&right, s) - eval(&c, 0.5 + 0.5 * s)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_restrict_preserves_the_polynomial() {
        let c = [2.0, -1.0, 0.25, 4.0, -3.0];
        for (s0, s1) in [(0.2, 0.7), (0.0, 0.4), (0.35, 1.0), (0.0, 1.0)] {
            let mut piece = c.map(|x| [x]);
            restrict(&mut piece, s0, s1);
            let piece = piece.map(|[x]| x);
            for s in [0.0, 0.3, 0.7, 1.0] {
                let expected = eval(&c, s0 + (s1 - s0) * s);
                assert!((eval(&piece, s) - expected).abs() < 1e-12);
            }
        }
    }

    /// Bernstein coefficients of `prod (s - r_i)` over `[0, 1]`.
    fn from_roots(roots: &[f64]) -> Vec<f64> {
        roots
            .iter()
            .fold(vec![1.0], |c, &r| product(&c, &[-r, 1.0 - r]))
    }

    #[test]
    fn test_rising_roots_brackets_each_rising_root_once() {
        // Positive at 0: falling through 0.2 and 0.62, rising through 0.6
        // and 0.9.
        let c = from_roots(&[0.2, 0.6, 0.62, 0.9]);
        let mut found = Vec::new();
        rising_roots::<CAPACITY>(&c, 1e-15, |s0, s1, _| found.push((s0, s1)));
        assert_eq!(found.len(), 2, "{found:?}");
        assert!(0.2 < found[0].0 && found[0].0 <= 0.6 && 0.6 <= found[0].1 && found[0].1 < 0.62);
        assert!(0.62 < found[1].0 && found[1].0 <= 0.9 && 0.9 <= found[1].1);
    }

    #[test]
    fn test_rising_roots_brackets_a_root_on_a_cut() {
        // Three sign changes but one real root, rising through 0.5: the
        // first cut.
        let c = [-3.0, 5.0 / 3.0, -5.0 / 3.0, 3.0];
        let mut found = Vec::new();
        rising_roots::<CAPACITY>(&c, 1e-15, |s0, s1, _| found.push((s0, s1)));
        assert!(
            found.iter().any(|&(s0, s1)| s0 < 0.5 && 0.5 < s1),
            "{found:?}"
        );
    }

    #[test]
    fn test_rising_roots_ignores_noise() {
        let c = [1e-18, -1e-18, 1e-18, -1e-18];
        let mut count = 0;
        rising_roots::<CAPACITY>(&c, 1e-15, |_, _, _| count += 1);
        assert_eq!(count, 0);
    }
}
