//! Small iterative solvers: Newton, Brent, and 2x2 to 4x4 linear
//! systems.
//!
//! All routines are allocation-free and take plain closures and stack
//! arrays. The root finders report `None` (rather than a best effort)
//! when they do not converge — callers seed them densely and keep the
//! best converged candidate; the polishing and minimising routines never
//! fail and return their best iterate.

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

/// Newton steps on `f(x) = 0` from `x`, each accepted only while it
/// strictly decreases `|f|`; returns the last accepted `x`.
///
/// Never fails: it polishes a candidate that is already close (a root of
/// a polynomial the equation was reduced to), so the result is never
/// worse than the input.
pub(crate) fn polish_root(
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    mut x: f64,
    max_iter: u32,
) -> f64 {
    for _ in 0..max_iter {
        let slope = df(x);
        if slope == 0.0 {
            break;
        }
        let next = x - f(x) / slope;
        let residual = f(next).abs();
        if residual.is_nan() || residual >= f(x).abs() {
            break;
        }
        x = next;
    }
    x
}

/// Solve the 3x3 system `a*x = b` by Gaussian elimination with partial
/// pivoting, or `None` when near-singular.
pub(crate) fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3], tol: f64) -> Option<[f64; 3]> {
    solve_square(a, b, tol)
}

/// Solve the 4x4 system `a*x = b` by Gaussian elimination with partial
/// pivoting, or `None` when near-singular.
pub(crate) fn solve_4x4(a: [[f64; 4]; 4], b: [f64; 4], tol: f64) -> Option<[f64; 4]> {
    solve_square(a, b, tol)
}

/// Gaussian elimination with partial pivoting; a pivot below `tol` times
/// the sum of `|a|` counts as singular (with `tol = 0`, only a zero pivot).
pub(crate) fn solve_square<const N: usize>(
    mut m: [[f64; N]; N],
    mut b: [f64; N],
    tol: f64,
) -> Option<[f64; N]> {
    let scale: f64 = m.iter().flatten().map(|v| v.abs()).sum();
    for col in 0..N {
        let mut pivot = col;
        for row in col + 1..N {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        m.swap(col, pivot);
        b.swap(col, pivot);
        if m[col][col].abs() <= tol * scale.max(f64::MIN_POSITIVE) {
            return None;
        }
        for row in col + 1..N {
            let factor = m[row][col] / m[col][col];
            let pivot_row = m[col];
            for (k, cell) in m[row].iter_mut().enumerate().skip(col) {
                *cell -= factor * pivot_row[k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; N];
    for row in (0..N).rev() {
        let mut acc = b[row];
        for k in row + 1..N {
            acc -= m[row][k] * x[k];
        }
        x[row] = acc / m[row][row];
    }
    Some(x)
}

/// Newton iteration for `f(x) = 0` in 3 unknowns from `x0`, with
/// backtracking. Converges on `|dx| <= tol * (1 + |x|)`.
pub(crate) fn newton_3d(
    mut f: impl FnMut([f64; 3]) -> [f64; 3],
    mut jac: impl FnMut([f64; 3]) -> [[f64; 3]; 3],
    mut x: [f64; 3],
    tol: f64,
    max_iter: u32,
) -> Option<[f64; 3]> {
    let norm = |v: [f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let mut best = norm(f(x));
    for _ in 0..max_iter {
        let fx = f(x);
        let delta = solve_3x3(jac(x), [-fx[0], -fx[1], -fx[2]], tol)?;
        let (mut s0, mut s1, mut s2) = (delta[0], delta[1], delta[2]);
        let mut improved = false;
        for _ in 0..12 {
            let trial = [x[0] + s0, x[1] + s1, x[2] + s2];
            let value = norm(f(trial));
            if value < best {
                x = trial;
                best = value;
                improved = true;
                break;
            }
            s0 *= 0.5;
            s1 *= 0.5;
            s2 *= 0.5;
        }
        if !improved {
            return Some(x);
        }
        if s0.abs() + s1.abs() + s2.abs() <= tol * (3.0 + x[0].abs() + x[1].abs() + x[2].abs()) {
            return Some(x);
        }
    }
    None
}

/// Box-bounded damped Newton minimisation of a smooth `f(x)` in 2
/// parameters, from `x0`.
///
/// `eval` returns `(f, gradient, hessian)` at a point; `bounds[i]` is
/// `None` for an unbounded (e.g. periodic) parameter. Each step solves
/// the Newton system on the parameters not pinned at a bound by an
/// outward gradient (so a minimum on an edge or corner of the box is
/// reached exactly instead of stalling against the clamp), shifting an
/// indefinite Hessian to positive definite. The step is projected onto
/// the box and halved until `f` decreases (or, once `f` is flat to
/// rounding, the gradient shrinks, so the minimiser is located to full
/// precision rather than `sqrt(eps)`); the accepted trial's derivatives
/// are reused, so each point is evaluated once.
///
/// Unlike the root finders above this never fails: it returns its last
/// (best) iterate once the (damped) step drops below `tol * (1 + |x|)`
/// per parameter without decreasing `f`, or `max_iter` runs out.
pub(crate) fn newton_min_2d(
    mut eval: impl FnMut(f64, f64) -> (f64, [f64; 2], [[f64; 2]; 2]),
    x0: [f64; 2],
    bounds: [Option<(f64, f64)>; 2],
    tol: f64,
    max_iter: u32,
) -> [f64; 2] {
    let project = |x: [f64; 2]| {
        let mut y = x;
        for (yi, b) in y.iter_mut().zip(bounds) {
            if let Some((lo, hi)) = b {
                *yi = yi.clamp(lo, hi);
            }
        }
        y
    };
    let mut x = project(x0);
    let (mut f, mut g, mut h) = eval(x[0], x[1]);
    for _ in 0..max_iter {
        let free = [0, 1].map(|i| match bounds[i] {
            Some((lo, hi)) => !((x[i] <= lo && g[i] > 0.0) || (x[i] >= hi && g[i] < 0.0)),
            None => true,
        });
        let d = newton_direction(g, h, free);
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let y = project([x[0] + alpha * d[0], x[1] + alpha * d[1]]);
            // A step below resolution: converged, or no descent left to find.
            if (0..2).all(|i| (y[i] - x[i]).abs() <= tol * (1.0 + x[i].abs())) {
                break;
            }
            let (fy, gy, hy) = eval(y[0], y[1]);
            // Near the minimum `f` is flat to rounding long before `x`
            // converges; there a step that shrinks the (free) gradient is
            // progress too, so Newton converges on the gradient instead.
            let norm = |g: [f64; 2]| {
                (0..2)
                    .filter(|&i| free[i])
                    .map(|i| g[i] * g[i])
                    .sum::<f64>()
            };
            if fy < f || (fy <= f + flat_band(f) && norm(gy) < norm(g)) {
                (x, f, g, h) = (y, fy, gy, hy);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    x
}

/// Bounded damped Newton minimisation of a smooth `f(x)` in 1 parameter,
/// from `x0`: the one-dimensional [`newton_min_2d`]. `eval` returns
/// `(f, f', f'')`; `bounds` is `None` for an unbounded (e.g. periodic)
/// parameter. Returns the last (best) iterate.
pub(crate) fn newton_min_1d(
    mut eval: impl FnMut(f64) -> (f64, f64, f64),
    x0: f64,
    bounds: Option<(f64, f64)>,
    tol: f64,
    max_iter: u32,
) -> f64 {
    let project = |x: f64| match bounds {
        Some((lo, hi)) => x.clamp(lo, hi),
        None => x,
    };
    let mut x = project(x0);
    let (mut f, mut g, mut h) = eval(x);
    for _ in 0..max_iter {
        let pinned = match bounds {
            Some((lo, hi)) => (x <= lo && g > 0.0) || (x >= hi && g < 0.0),
            None => false,
        };
        if pinned {
            break;
        }
        let d = -g / curvature(h, g);
        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..40 {
            let y = project(x + alpha * d);
            if (y - x).abs() <= tol * (1.0 + x.abs()) {
                break;
            }
            let (fy, gy, hy) = eval(y);
            if fy < f || (fy <= f + flat_band(f) && gy.abs() < g.abs()) {
                (x, f, g, h) = (y, fy, gy, hy);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    x
}

/// Rounding band of an objective value: changes of `f` within it are
/// noise, not information.
fn flat_band(f: f64) -> f64 {
    8.0 * f64::EPSILON * f.abs().max(f64::MIN_POSITIVE)
}

/// Descent direction `-(H + mu I)^{-1} g` restricted to the `free`
/// parameters (zero on the others), with `mu` the smallest shift making
/// the reduced Hessian safely positive definite (`0` when it already is).
fn newton_direction(g: [f64; 2], h: [[f64; 2]; 2], free: [bool; 2]) -> [f64; 2] {
    match free {
        [true, true] => {
            let (a, b, c) = (h[0][0], 0.5 * (h[0][1] + h[1][0]), h[1][1]);
            let mean = 0.5 * (a + c);
            let radius = (0.25 * (a - c) * (a - c) + b * b).sqrt();
            let (lo, hi) = (mean - radius, mean + radius);
            let floor = 1e-10 * (lo.abs() + hi.abs()) + f64::MIN_POSITIVE;
            let mu = if lo > floor { 0.0 } else { floor - lo };
            let (a, c) = (a + mu, c + mu);
            let det = a * c - b * b;
            [-(c * g[0] - b * g[1]) / det, -(a * g[1] - b * g[0]) / det]
        }
        [true, false] => [-g[0] / curvature(h[0][0], g[0]), 0.0],
        [false, true] => [0.0, -g[1] / curvature(h[1][1], g[1])],
        [false, false] => [0.0, 0.0],
    }
}

/// A positive 1D Newton denominator: the curvature when positive, else a
/// tiny one so the step heads downhill and the line search sizes it.
fn curvature(hh: f64, g: f64) -> f64 {
    if hh > 0.0 {
        hh
    } else {
        hh.abs().max(g.abs()).max(f64::MIN_POSITIVE) * 1e-6
    }
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
    fn test_solve_3x3() {
        let x = solve_3x3(
            [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]],
            [1.0, -2.0, 0.0],
            1e-12,
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] + 2.0).abs() < 1e-9);
        assert!((x[2] + 2.0).abs() < 1e-9);
        assert!(solve_3x3([[1.0, 2.0, 3.0]; 3], [1.0, 2.0, 3.0], 1e-12).is_none());
    }

    #[test]
    fn test_solve_4x4() {
        // Needs a row swap: the first pivot is zero.
        let a = [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 3.0],
        ];
        let x = solve_4x4(a, [10.0, 4.0, 6.0, 16.0], 1e-12).unwrap();
        for (xi, want) in x.iter().zip([1.0, 0.0, 3.0, 4.0]) {
            assert!((xi - want).abs() < 1e-12, "{x:?}");
        }
        assert!(solve_4x4([[1.0, 2.0, 3.0, 4.0]; 4], [1.0; 4], 1e-12).is_none());
    }

    #[test]
    fn test_newton_3d() {
        // Intersection of x^2 + y^2 = 2, y = x, z = 1 near (1, 1, 1).
        let x = newton_3d(
            |v| [v[0] * v[0] + v[1] * v[1] - 2.0, v[1] - v[0], v[2] - 1.0],
            |v| {
                [
                    [2.0 * v[0], 2.0 * v[1], 0.0],
                    [-1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            },
            [1.2, 0.8, 0.5],
            1e-12,
            50,
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 1.0).abs() < 1e-9);
        assert!((x[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_polish_root_sqrt2() {
        let x = polish_root(|x| x * x - 2.0, |x| 2.0 * x, 1.4, 8);
        assert!((x - std::f64::consts::SQRT_2).abs() < 1e-15);
        // A flat function leaves the candidate unchanged.
        assert_eq!(polish_root(|_| 1.0, |_| 0.0, 0.5, 8), 0.5);
        // A step that would increase |f| is refused.
        assert_eq!(
            polish_root(|x: f64| x.atan(), |x| 1.0 / (1.0 + x * x), 3.0, 8),
            3.0
        );
    }

    #[test]
    fn test_brent_minimum() {
        // min of (x - 2)^4 + (x - 2)^2 at x = 2.
        let x = brent_minimum(|x| (x - 2.0).powi(4) + (x - 2.0).powi(2), -5.0, 5.0, 1e-10);
        assert!((x - 2.0).abs() < 1e-7);
    }
}
