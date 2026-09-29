//! Small iterative solvers: Newton, Gauss-Newton, Brent, and 2x2 systems.
//!
//! All routines are allocation-free, take plain closures and stack
//! arrays, and report `None` (rather than a best effort) when they do
//! not converge — callers seed them densely and keep the best converged
//! candidate.

/// Solve the 2x2 system `a*x = b`, or `None` when near-singular.
pub(crate) fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2], tol: f64) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let scale = a[0][0].abs() * a[1][1].abs() + a[0][1].abs() * a[1][0].abs();
    if det.abs() <= tol * scale.max(f64::MIN_POSITIVE) {
        return None;
    }
    Some([
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ])
}

/// Newton iteration for `f(x) = 0` from `x0` (needs `f` and `df`).
/// Converges on `|dx| <= tol * (1 + |x|)`; gives up after `max_iter`.
pub(crate) fn newton_1d(
    mut f: impl FnMut(f64) -> f64,
    mut df: impl FnMut(f64) -> f64,
    mut x: f64,
    tol: f64,
    max_iter: u32,
) -> Option<f64> {
    for _ in 0..max_iter {
        let (fx, dfx) = (f(x), df(x));
        if dfx.abs() <= f64::MIN_POSITIVE * (1.0 + fx.abs()) {
            return None;
        }
        let dx = fx / dfx;
        x -= dx;
        if dx.abs() <= tol * (1.0 + x.abs()) {
            return Some(x);
        }
    }
    None
}

/// Gauss-Newton step driver for 3-component residuals in 1 parameter
/// (curve projection): minimizes `|r(t)|^2` via
/// `delta = -(J.r) / (J.J)` with backtracking.
pub(crate) fn gauss_newton_1d(
    mut residual: impl FnMut(f64) -> [f64; 3],
    mut jacobian: impl FnMut(f64) -> [f64; 3],
    mut t: f64,
    tol: f64,
    max_iter: u32,
) -> Option<f64> {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut best = dot(residual(t), residual(t));
    for _ in 0..max_iter {
        let (r, j) = (residual(t), jacobian(t));
        let jj = dot(j, j);
        if jj <= f64::MIN_POSITIVE * (1.0 + best) {
            return None;
        }
        let mut step = -dot(j, r) / jj;
        // Backtrack while the step fails to improve.
        let mut improved = false;
        for _ in 0..12 {
            let trial = t + step;
            let value = dot(residual(trial), residual(trial));
            if value < best {
                t = trial;
                best = value;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            return Some(t);
        }
        if step.abs() <= tol * (1.0 + t.abs()) {
            return Some(t);
        }
    }
    None
}

/// Gauss-Newton driver for 3-component residuals in 2 parameters
/// (surface projection): `delta = -(J'J)^{-1} J'r` with backtracking.
pub(crate) fn gauss_newton_2d(
    mut residual: impl FnMut(f64, f64) -> [f64; 3],
    mut jac_u: impl FnMut(f64, f64) -> [f64; 3],
    mut jac_v: impl FnMut(f64, f64) -> [f64; 3],
    mut u: f64,
    mut v: f64,
    tol: f64,
    max_iter: u32,
) -> Option<(f64, f64)> {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut best = dot(residual(u, v), residual(u, v));
    for _ in 0..max_iter {
        let (r, ju, jv) = (residual(u, v), jac_u(u, v), jac_v(u, v));
        let lhs = [[dot(ju, ju), dot(ju, jv)], [dot(ju, jv), dot(jv, jv)]];
        let rhs = [-dot(ju, r), -dot(jv, r)];
        let delta = solve_2x2(lhs, rhs, tol)?;
        let (mut su, mut sv) = (delta[0], delta[1]);
        let mut improved = false;
        for _ in 0..12 {
            let value = dot(residual(u + su, v + sv), residual(u + su, v + sv));
            if value < best {
                u += su;
                v += sv;
                best = value;
                improved = true;
                break;
            }
            su *= 0.5;
            sv *= 0.5;
        }
        if !improved {
            return Some((u, v));
        }
        if su.abs() + sv.abs() <= tol * (2.0 + u.abs() + v.abs()) {
            return Some((u, v));
        }
    }
    None
}

/// Brent's minimum of `f` on `[a, b]` (no bracket requirement beyond the
/// interval): parabolic interpolation with golden-section fallback.
pub(crate) fn brent_minimum(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    const GOLD: f64 = 0.381_966_011_250_105_1;
    let (mut lo, mut hi) = (a.min(b), a.max(b));
    let (mut x, mut w, mut v) = (
        lo + GOLD * (hi - lo),
        lo + GOLD * (hi - lo),
        lo + GOLD * (hi - lo),
    );
    let (mut fx, mut fw, mut fv) = (f(x), f(x), f(x));
    let (mut d, mut e): (f64, f64) = (0.0, 0.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let tol1 = tol * x.abs().max(1.0) + 1e-300;
        let tol2 = 2.0 * tol1;
        if (x - mid).abs() <= tol2 - 0.5 * (hi - lo) {
            return x;
        }
        let mut p = 0.0;
        let mut q = 0.0;
        let mut r = 0.0;
        if e.abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        let u = if p.abs() < 0.5 * q.abs() * r.abs() && p > q * (lo - x) && p < q * (hi - x) {
            d = p / q;
            x + d
        } else {
            e = if x >= mid { lo - x } else { hi - x };
            d = GOLD * e;
            x + d
        };
        let fu = f(u.max(lo).min(hi));
        if fu <= fx {
            if u >= x {
                lo = x;
            } else {
                hi = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_2x2() {
        let x = solve_2x2([[2.0, 1.0], [1.0, 3.0]], [5.0, 6.0], 1e-12).unwrap();
        assert!((x[0] - 1.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
        assert!(solve_2x2([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0], 1e-12).is_none());
    }

    #[test]
    fn test_newton_sqrt2() {
        let x = newton_1d(|x| x * x - 2.0, |x| 2.0 * x, 1.0, 1e-12, 50).unwrap();
        assert!((x - std::f64::consts::SQRT_2).abs() < 1e-12);
        assert!(newton_1d(|_| 1.0, |_| 0.0, 0.0, 1e-12, 10).is_none());
    }

    #[test]
    fn test_gauss_newton_1d_line_fit() {
        // Closest t on the x-axis to (3, 4, 0): t = 3.
        let t = gauss_newton_1d(
            |t| [t - 3.0, -4.0, 0.0],
            |_| [1.0, 0.0, 0.0],
            0.0,
            1e-12,
            50,
        )
        .unwrap();
        assert!((t - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_gauss_newton_2d_plane_fit() {
        // Closest (u, v) on z = 0 to (1, 2, 3): (1, 2).
        let (u, v) = gauss_newton_2d(
            |u, v| [u - 1.0, v - 2.0, -3.0],
            |_, _| [1.0, 0.0, 0.0],
            |_, _| [0.0, 1.0, 0.0],
            0.0,
            0.0,
            1e-12,
            50,
        )
        .unwrap();
        assert!((u - 1.0).abs() < 1e-9);
        assert!((v - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_brent_minimum() {
        // min of (x - 2)^4 + (x - 2)^2 at x = 2.
        let x = brent_minimum(|x| (x - 2.0).powi(4) + (x - 2.0).powi(2), -5.0, 5.0, 1e-10);
        assert!((x - 2.0).abs() < 1e-7);
    }
}
