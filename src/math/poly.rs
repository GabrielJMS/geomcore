//! Real roots of low-degree polynomials with multiplicity clustering.
//!
//! Closed forms (stable quadratic, Cardano/trigonometric cubic, Ferrari
//! quartic with a biquadratic shortcut), two Newton polish passes on the
//! original polynomial, then tolerance clustering. Near-multiple roots
//! survive as multiplicity counts instead of collapsing — tangencies
//! downstream depend on it.

/// A real root with its multiplicity (clustered near-coincidences add up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RealRoot {
    /// Root location.
    pub(crate) value: f64,
    /// How many roots coincided here (1 = simple).
    pub(crate) multiplicity: u32,
}

/// Evaluate the ascending-coefficient polynomial at `x` (Horner).
pub(crate) fn eval(coeffs: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Derivative coefficients (ascending) of `coeffs`.
pub(crate) fn deriv(coeffs: &[f64]) -> Vec<f64> {
    coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, &c)| i as f64 * c)
        .collect()
}

/// Real roots of the ascending-coefficient polynomial, clustered within
/// `tol * (1 + |x|)` with summed multiplicity. Degrees above 4 return an
/// empty vec (callers needing them do not exist yet); the zero polynomial
/// and constants likewise yield no roots.
pub(crate) fn real_roots(coeffs: &[f64], tol: f64) -> Vec<RealRoot> {
    // Trim high-degree near-zero coefficients (relative to the max).
    let scale = coeffs.iter().fold(0.0f64, |m, &c| m.max(c.abs()));
    if scale == 0.0 {
        return Vec::new();
    }
    let mut n = coeffs.len();
    while n > 1 && coeffs[n - 1].abs() <= tol * scale {
        n -= 1;
    }
    let c = &coeffs[..n];
    let mut raw: Vec<f64> = match n - 1 {
        0 => Vec::new(),
        1 => vec![-c[0] / c[1]],
        2 => quadratic_raw(c[2], c[1], c[0], tol),
        3 => cubic_raw(c[3], c[2], c[1], c[0], tol),
        4 => quartic_raw(c[4], c[3], c[2], c[1], c[0], tol),
        _ => Vec::new(),
    };
    if raw.is_empty() {
        return Vec::new();
    }
    // Normalize to monic for polishing.
    let lead = c[n - 1];
    let monic: Vec<f64> = c.iter().map(|&v| v / lead).collect();
    let d1 = deriv(&monic);
    let d2 = deriv(&d1);
    for x in &mut raw {
        for _ in 0..2 {
            let slope = eval(&d1, *x);
            if slope.abs() <= 1e-300 * (1.0 + eval(&monic, *x).abs()) {
                break;
            }
            *x -= eval(&monic, *x) / slope;
        }
    }
    cluster(raw, &monic, &d1, &d2, scale, tol)
}

/// Cancellation-free quadratic roots of `a*x^2 + b*x + c = 0`.
fn quadratic_raw(a: f64, b: f64, c: f64, tol: f64) -> Vec<f64> {
    // Root separation decides tangency (same criterion as the
    // intersection solver): separations below tol*(1+|mid|) graze.
    let disc = b * b - 4.0 * a * c;
    let mid = -b / (2.0 * a);
    let sep = disc.abs().sqrt() / (2.0 * a.abs());
    if sep <= tol * (1.0 + mid.abs()) {
        vec![mid]
    } else if disc < 0.0 {
        Vec::new()
    } else {
        // Stable: q = -0.5*(b + sign(b)*sqrt(disc)), roots q/a, c/q.
        let q = -0.5 * (b + b.signum() * disc.sqrt());
        if q == 0.0 {
            vec![0.0]
        } else {
            vec![q / a, c / q]
        }
    }
}

/// Raw real roots of the cubic `a*x^3 + b*x^2 + c*x + d = 0`.
fn cubic_raw(a: f64, b: f64, c: f64, d: f64, _tol: f64) -> Vec<f64> {
    // Depressed t^3 + p*t + q = 0 with x = t - b/3a.
    let (p, q) = (
        (3.0 * a * c - b * b) / (3.0 * a * a),
        (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a),
    );
    let shift = -b / (3.0 * a);
    let disc = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    if disc > 0.0 {
        let root_disc = disc.sqrt();
        vec![(-q / 2.0 + root_disc).cbrt() + (-q / 2.0 - root_disc).cbrt() + shift]
    } else {
        // Three real roots (casus irreducibilis): trigonometric form.
        // s <= 0 with disc <= 0 forces s == q == 0: triple root at shift.
        let s = -p / 3.0;
        if s <= 0.0 {
            return vec![shift, shift, shift];
        }
        let m = 2.0 * s.sqrt();
        let theta = (-q / 2.0 / (s * s * s).sqrt()).clamp(-1.0, 1.0).acos();
        (0..3)
            .map(|k| m * ((theta + 2.0 * k as f64 * std::f64::consts::PI) / 3.0).cos() + shift)
            .collect()
    }
}

/// Raw real roots of the quartic `a*x^4 + b*x^3 + c*x^2 + d*x + e = 0`.
fn quartic_raw(a: f64, b: f64, c: f64, d: f64, e: f64, tol: f64) -> Vec<f64> {
    // Normalize, then depress x = t - b/4: t^4 + p*t^2 + q*t + r = 0.
    let (b, c, d, e) = (b / a, c / a, d / a, e / a);
    let shift = -b / 4.0;
    let p = c - 3.0 * b * b / 8.0;
    let q = d - b * c / 2.0 + b * b * b / 8.0;
    let r = e - b * d / 4.0 + b * b * c / 16.0 - 3.0 * b * b * b * b / 256.0;
    // Biquadratic shortcut (exact when q vanishes): far more accurate
    // than Ferrari near symmetric configurations.
    if q.abs() <= tol * (1.0 + p.abs() + r.abs()) {
        let mut out = Vec::new();
        for y in quadratic_raw(1.0, p, r, tol) {
            if y > tol * (1.0 + y.abs()) {
                let s = y.sqrt();
                out.push(s + shift);
                out.push(-s + shift);
            } else if y >= -tol * (1.0 + y.abs()) {
                // t^2 = y ~= 0: double root at the shift (quadruple when
                // the resolvent itself doubles at zero, i.e. p ~= 0).
                let times = if p.abs() <= tol * (1.0 + p.abs() + r.abs()) {
                    4
                } else {
                    2
                };
                out.extend(std::iter::repeat_n(shift, times));
            }
        }
        return out;
    }
    // Ferrari: (t^2 + u*t + v)(t^2 - u*t + w); u^2 is the largest real
    // root of m^3 + 2*p*m^2 + (p^2 - 4*r)*m - q^2 = 0 (one always > 0).
    let mut m_roots = cubic_raw(1.0, 2.0 * p, p * p - 4.0 * r, -q * q, tol);
    if m_roots.is_empty() {
        return Vec::new();
    }
    m_roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let u = m_roots[m_roots.len() - 1].max(0.0).sqrt();
    if u <= 0.0 {
        return Vec::new();
    }
    let v = ((p + u * u) - q / u) / 2.0;
    let w = ((p + u * u) + q / u) / 2.0;
    let mut out = quadratic_raw(1.0, u, v, tol);
    out.extend(quadratic_raw(1.0, -u, w, tol));
    out.iter().map(|&t| t + shift).collect()
}

/// Sort, cluster within `tol * (1 + |x|)`, and refine multiplicity: a
/// lone root with a vanishing derivative is a collapsed multiple root.
fn cluster(
    mut raw: Vec<f64>,
    monic: &[f64],
    d1: &[f64],
    d2: &[f64],
    scale: f64,
    tol: f64,
) -> Vec<RealRoot> {
    raw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut roots: Vec<RealRoot> = Vec::new();
    for x in raw {
        let gap = tol * (1.0 + x.abs());
        let merge = match roots.last() {
            Some(last) => (x - last.value).abs() <= gap.max(tol * (1.0 + last.value.abs())),
            None => false,
        };
        if merge {
            let last = roots.last_mut().expect("merge implies a root exists");
            last.value =
                (last.value * last.multiplicity as f64 + x) / (last.multiplicity as f64 + 1.0);
            last.multiplicity += 1;
            continue;
        }
        roots.push(RealRoot {
            value: x,
            multiplicity: 1,
        });
    }
    // Lone roots sitting on a flat slope are collapsed tangencies.
    let degree = monic.len().saturating_sub(1) as u32;
    let total: u32 = roots.iter().map(|r| r.multiplicity).sum();
    for root in &mut roots {
        if root.multiplicity == 1 && total < degree {
            let slope_scale = scale * (1.0 + root.value.abs()).powi(degree as i32);
            if eval(d1, root.value).abs() <= tol * slope_scale {
                root.multiplicity = 2;
                if eval(d2, root.value).abs() <= tol * slope_scale && total + 1 < degree {
                    root.multiplicity = 3;
                }
            }
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(roots[0].multiplicity >= 2, "{roots:?}");
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
        // (x^2 - 1)^2 = x^4 - 2x^2 + 1: doubles at +-1.
        let roots = real_roots(&[1.0, 0.0, -2.0, 0.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 2);
        assert!(roots.iter().all(|r| r.multiplicity == 2));
    }

    #[test]
    fn test_quartic_quadruple_root() {
        // (x + 2)^4.
        let roots = real_roots(&[16.0, 32.0, 24.0, 8.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value + 2.0).abs() < 1e-6);
        assert!(roots[0].multiplicity >= 3, "{roots:?}");
    }

    #[test]
    fn test_quartic_two_real_two_complex() {
        // (x^2 + 1)(x - 1)(x - 5) = x^4 - 6x^3 + 6x^2 - 6x + 5.
        let roots = real_roots(&[5.0, -6.0, 6.0, -6.0, 1.0], 1e-9);
        assert_eq!(roots.len(), 2);
        assert!((roots[0].value - 1.0).abs() < 1e-6);
        assert!((roots[1].value - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_leading_zeros_drop_degree() {
        // 0*x^2 + 2x + 1 treated as linear.
        let roots = real_roots(&[1.0, 2.0, 1e-15], 1e-12);
        assert_eq!(roots.len(), 1);
        assert!((roots[0].value + 0.5).abs() < 1e-9);
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
}
