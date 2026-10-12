//! Equations in one curve parameter, solved with a [`Tolerance`]: the
//! quadratic, first-order and double-angle trigonometric forms, and
//! polynomial roots with geometric tangency. The exact intersections
//! reduce to these once a curve's parametrization is substituted into
//! the other shape's implicit equation.

use super::poly::{RealRoot, real_roots_touching};
use crate::{InlineList, Tolerance};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum QuadraticSolution {
    /// Two distinct roots, ordered.
    Two(f64, f64),
    /// A double root (discriminant within tolerance of zero).
    One(f64),
    /// Degenerate linear equation with a single root.
    Linear(f64),
    /// Degenerate `0 = 0` (caller decides coincidence separately).
    Degenerate,
    /// No (real) roots.
    Empty,
}

impl QuadraticSolution {
    /// The same solution with every root moved by `t0` (undoing a
    /// substitution `t = t0 + s` made to centre the parameter).
    pub(crate) fn shifted(self, t0: f64) -> Self {
        match self {
            Self::Two(t1, t2) => Self::Two(t0 + t1, t0 + t2),
            Self::One(t) => Self::One(t0 + t),
            Self::Linear(t) => Self::Linear(t0 + t),
            other => other,
        }
    }

    /// Geometric tangency at the quadratic's vertex `t*` (its critical
    /// point): when `touches(t*)` — the curve point there is within
    /// tolerance of the surface — two close roots or a near-real complex
    /// pair are one double root at `t*`. The tolerance is a distance, so
    /// this (not the roots' separation in parameter space, which input
    /// rounding of `δ` widens to about `sqrt(2*rho*δ)`) decides contact.
    pub(crate) fn touching_at(self, vertex: f64, touches: impl Fn(f64) -> bool) -> Self {
        match self {
            Self::Two(..) | Self::Empty if vertex.is_finite() && touches(vertex) => {
                Self::One(vertex)
            }
            other => other,
        }
    }
}

/// Real roots of the ascending polynomial `poly` in a curve parameter,
/// with geometric tangency. Roots come from [`real_roots`] (clustering
/// within `tol.confusion` in parameter space); then every critical point
/// `t*` of `poly` where `touches(t*)` holds (the curve is within
/// tolerance of the surface there) marks a grazing contact: the roots it
/// connects merge into one root at `t*` counting them, and a complex pair
/// beside it reports as a double root. This is what keeps exact
/// tangencies tangent when input rounding splits their double root by
/// more than the parameter band. See [`real_roots_touching`].
///
/// [`real_roots`]: super::poly::real_roots
pub(crate) fn roots_with_tangency(
    poly: &[f64],
    tol: Tolerance,
    touches: impl Fn(f64) -> bool,
) -> InlineList<RealRoot, 4> {
    real_roots_touching(poly, tol.confusion, touches)
}

/// Midpoint `-b / (2a)` of the roots of `a*t^2 + b*t + c`, or 0 when the
/// equation is (near) linear for [`solve_quadratic`]. Line callers
/// re-form their quadratic about this parameter so the band and the
/// constant term do not depend on where the line's origin happens to be.
pub(crate) fn quadratic_centre(a: f64, b: f64, tol: Tolerance) -> f64 {
    let t0 = -b / (2.0 * a);
    if a.abs() > tol.angular && t0.is_finite() {
        t0
    } else {
        0.0
    }
}

/// Solve `a*t^2 + b*t + c = 0`, classifying with `tol`.
///
/// Near-zero `a` (against `tol.angular`, the coefficients being unit
/// scale for unit-direction lines) falls back to the linear equation.
/// Otherwise the root half-separation `sqrt(|disc|) / (2|a|)` decides:
/// separations within `tol.confusion * (1 + |t0|)` of the double root,
/// widened by the rounding-noise radius of the discriminant
/// (`sqrt(4*eps*(b^2 + 4|ac|)) / (2|a|)`), report grazing contact; wider
/// positive discriminants give two ordered roots, wider negative ones
/// empty. The band grows with `|t0|`: callers whose parameter is an
/// arbitrary offset (a line's) centre it first ([`quadratic_centre`]).
pub(crate) fn solve_quadratic(a: f64, b: f64, c: f64, tol: Tolerance) -> QuadraticSolution {
    if a.abs() <= tol.angular {
        if b.abs() <= tol.confusion {
            return QuadraticSolution::Degenerate;
        }
        return QuadraticSolution::Linear(-c / b);
    }
    let disc = b * b - 4.0 * a * c;
    let mid = -b / (2.0 * a);
    let sep = disc.abs().sqrt() / (2.0 * a.abs());
    let noise = (4.0 * f64::EPSILON * (b * b + 4.0 * (a * c).abs())).sqrt() / (2.0 * a.abs());
    if sep <= tol.confusion * (1.0 + mid.abs()) + noise {
        QuadraticSolution::One(mid)
    } else if disc < 0.0 {
        QuadraticSolution::Empty
    } else {
        // Cancellation-free: the larger-magnitude root first, the other
        // from the product c/a.
        let q = -0.5 * (b + disc.sqrt().copysign(b));
        let (r1, r2) = (q / a, c / q);
        QuadraticSolution::Two(r1.min(r2), r1.max(r2))
    }
}

/// Solve `C*cos t + D*sin t + E = 0`, returning `(parameter, multiplicity)`.
/// Tangency (grazing) reports multiplicity 2; `scale` carries the equation
/// units for the tolerance band. Below-significance equations (amplitude
/// within tolerance) yield nothing — whole-curve coincidence is the
/// caller's precheck.
pub(crate) fn solve_trig_linear(
    c: f64,
    d: f64,
    e: f64,
    scale: f64,
    tol: Tolerance,
) -> InlineList<(f64, u32), 4> {
    let mut out = InlineList::new();
    let amp = c.hypot(d);
    let band = tol.confusion * scale.max(f64::MIN_POSITIVE);
    if amp <= band || e.abs() > amp + band {
        return out;
    }
    let phi = d.atan2(c);
    let t0 = (-e / amp).clamp(-1.0, 1.0).acos();
    if e.abs() >= amp - band {
        out.push((phi + t0, 2));
    } else {
        out.push((phi + t0, 1));
        out.push((phi - t0, 1));
    }
    out
}

/// Solve `A*cos2t + B*sin2t + C*cos t + D*sin t + E = 0`, returning
/// `(parameter, multiplicity)` with parameters in `(-PI, PI]`. Callers
/// filter by surface containment.
///
/// The substitution `t = t0 + 2*atan(u)` turns the equation into a quartic
/// in `u`. Its leading coefficient is the equation's value at `t0 + PI`,
/// the point `u = ∞`, so `t0` is the sample that keeps that value
/// largest: no root then lies near `u = ∞`, where rounding would turn a
/// double root into a complex pair or two roots far apart in `t`.
/// Tangency is decided by distance, not by the double root surviving
/// rounding: `touches(t)` says whether the curve point at `t` is within
/// tolerance of the other shape (see [`roots_with_tangency`]).
pub(crate) fn solve_trig(
    a2: f64,
    b2: f64,
    c1: f64,
    d1: f64,
    e: f64,
    tol: Tolerance,
    touches: impl Fn(f64) -> bool,
) -> InlineList<(f64, u32), 4> {
    use std::f64::consts::{FRAC_1_SQRT_2 as H, FRAC_PI_4, PI};
    // `t0` is a multiple `k` of PI/4: cos and sin of `k*PI/4`, and of
    // `t0 + PI` at index `k + 4`, come from this table.
    const UNIT: [(f64, f64); 8] = [
        (1.0, 0.0),
        (H, H),
        (0.0, 1.0),
        (-H, H),
        (-1.0, 0.0),
        (-H, -H),
        (0.0, -1.0),
        (H, -H),
    ];
    let f_at = |k: usize| {
        let ((cos1, sin1), (cos2, sin2)) = (UNIT[k % 8], UNIT[2 * k % 8]);
        a2 * cos2 + b2 * sin2 + c1 * cos1 + d1 * sin1 + e
    };
    let (mut k, mut largest) = (0, f_at(4).abs());
    for j in 1..8 {
        let v = f_at(j + 4).abs();
        if v.total_cmp(&largest).is_ge() {
            (k, largest) = (j, v);
        }
    }
    let t0 = k as f64 * FRAC_PI_4;
    // The same equation in `s = t - t0`.
    let ((cos1, sin1), (cos2, sin2)) = (UNIT[k], UNIT[2 * k % 8]);
    let (a2, b2) = (a2 * cos2 + b2 * sin2, b2 * cos2 - a2 * sin2);
    let (c1, d1) = (c1 * cos1 + d1 * sin1, d1 * cos1 - c1 * sin1);
    // x(u^4 - 6u^2 + 1) form; ascending u^0..u^4 coefficients.
    let q = [
        a2 + c1 + e,
        4.0 * b2 + 2.0 * d1,
        -6.0 * a2 + 2.0 * e,
        -4.0 * b2 + 2.0 * d1,
        a2 - c1 + e,
    ];
    // `t0` is in `[0, 2*PI)` and `2*atan(u)` in `(-PI, PI)`: one turn
    // back at most brings `t` into `(-PI, PI]`.
    let to_t = |u: f64| {
        let t = t0 + 2.0 * u.atan();
        if t > PI { t - 2.0 * PI } else { t }
    };
    let mut out = InlineList::new();
    for r in &roots_with_tangency(&q, tol, |u| touches(to_t(u))) {
        out.push((to_t(r.value), r.multiplicity));
    }
    out.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_quadratic_two_roots_ordered() {
        // t^2 - 5t + 6 = 0 -> 2, 3.
        assert_eq!(
            solve_quadratic(1.0, -5.0, 6.0, Tolerance::DEFAULT),
            QuadraticSolution::Two(2.0, 3.0)
        );
    }

    #[test]
    fn test_solve_quadratic_tangent_band() {
        // Exact double root.
        assert_eq!(
            solve_quadratic(1.0, -4.0, 4.0, Tolerance::DEFAULT),
            QuadraticSolution::One(2.0)
        );
        // Miss by sqrt(c) = 1e-7, i.e. exactly at tolerance: still grazing.
        assert_eq!(
            solve_quadratic(1.0, 0.0, 1e-14, Tolerance::DEFAULT),
            QuadraticSolution::One(0.0)
        );
        // Miss by sqrt(c) = 3e-5, beyond tolerance: empty.
        assert_eq!(
            solve_quadratic(1.0, 0.0, 1e-9, Tolerance::DEFAULT),
            QuadraticSolution::Empty
        );
        // Clearly negative discriminant misses.
        assert_eq!(
            solve_quadratic(1.0, 0.0, 1.0, Tolerance::DEFAULT),
            QuadraticSolution::Empty
        );
    }

    #[test]
    fn test_solve_quadratic_rounded_double_root_is_tangent() {
        // a*(t - r)^2 with rounded coefficients: the discriminant is pure
        // rounding noise (negative in one case, positive in the other), so
        // even a 1e-12 band reports the double root.
        let tight = Tolerance {
            confusion: 1e-12,
            ..Tolerance::DEFAULT
        };
        for (a, r) in [(7.0, 2.3), (11.0, 0.7)] {
            match solve_quadratic(a, -2.0 * a * r, a * r * r, tight) {
                QuadraticSolution::One(t) => assert!((t - r).abs() < 1e-14),
                other => panic!("expected a double root near {r}, got {other:?}"),
            }
        }
        // A real gap of 1e-6 is still two roots (to the accuracy the
        // rounded constant term allows).
        match solve_quadratic(1.0, -2.0, 1.0 - 1e-12, tight) {
            QuadraticSolution::Two(t1, t2) => {
                assert!((t1 - (1.0 - 1e-6)).abs() < 1e-9);
                assert!((t2 - (1.0 + 1e-6)).abs() < 1e-9);
            }
            other => panic!("expected two roots, got {other:?}"),
        }
    }

    #[test]
    fn test_solve_quadratic_widely_separated_roots_are_accurate() {
        // t^2 - 1e8*t + 1 = 0: roots 1e-8 and 1e8; the small one must not
        // be lost to cancellation.
        match solve_quadratic(1.0, -1e8, 1.0, Tolerance::DEFAULT) {
            QuadraticSolution::Two(t1, t2) => {
                assert!((t1 - 1e-8).abs() < 1e-22);
                assert!((t2 - 1e8).abs() < 1e-7);
            }
            other => panic!("expected two roots, got {other:?}"),
        }
    }

    #[test]
    fn test_solve_quadratic_degenerate_arms() {
        // Linear: 2t + 4 = 0.
        assert_eq!(
            solve_quadratic(0.0, 2.0, 4.0, Tolerance::DEFAULT),
            QuadraticSolution::Linear(-2.0)
        );
        // Fully degenerate.
        assert_eq!(
            solve_quadratic(0.0, 0.0, 1.0, Tolerance::DEFAULT),
            QuadraticSolution::Degenerate
        );
    }
}
