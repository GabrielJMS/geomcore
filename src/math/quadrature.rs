//! Adaptive Gauss–Kronrod quadrature: the 15-point Kronrod rule and the
//! 7-point Gauss rule it extends share their nodes, and their difference
//! estimates the error of the 7-point one, so the 15-point result is kept
//! where it is small enough and the interval halved elsewhere.

/// Kronrod nodes in `[0, 1)` of the rule over `[-1, 1]`, by decreasing
/// distance from the centre: the odd ones are the Gauss nodes.
const NODES: [f64; 7] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
];

/// Kronrod weights of [`NODES`], then of the centre.
const KRONROD_WEIGHTS: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_83,
];

/// Gauss weights of the odd [`NODES`], then of the centre.
const GAUSS_WEIGHTS: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// Halvings of the interval below which the rule is taken as it comes: an
/// interval a billionth as wide is left only by an integrand that is not
/// smooth there.
const MAX_DEPTH: u32 = 30;

/// The integral of `f` over `[a, b]`, within `max_error` as far as the
/// difference of the two rules tells, halving each interval whose share
/// of `max_error` (in proportion to its width) it misses. An interval
/// stops halving once that difference is down to rounding, or after
/// [`MAX_DEPTH`] halvings, so that any `max_error` ends.
pub(crate) fn integrate(f: &impl Fn(f64) -> f64, a: f64, b: f64, max_error: f64) -> f64 {
    integrate_from(f, a, b, max_error, 0)
}

fn integrate_from(f: &impl Fn(f64) -> f64, a: f64, b: f64, max_error: f64, depth: u32) -> f64 {
    let (kronrod, gauss) = rules(f, a, b);
    let rounding = 50.0 * f64::EPSILON * kronrod.abs();
    if (kronrod - gauss).abs() <= max_error.max(rounding) || depth == MAX_DEPTH {
        return kronrod;
    }
    let middle = 0.5 * (a + b);
    integrate_from(f, a, middle, 0.5 * max_error, depth + 1)
        + integrate_from(f, middle, b, 0.5 * max_error, depth + 1)
}

/// The 15-point Kronrod and 7-point Gauss results over `[a, b]`.
fn rules(f: &impl Fn(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
    let (centre, half) = (0.5 * (a + b), 0.5 * (b - a));
    let middle = f(centre);
    let mut kronrod = KRONROD_WEIGHTS[7] * middle;
    let mut gauss = GAUSS_WEIGHTS[3] * middle;
    for (k, &x) in NODES.iter().enumerate() {
        let pair = f(centre - half * x) + f(centre + half * x);
        kronrod += KRONROD_WEIGHTS[k] * pair;
        if k % 2 == 1 {
            gauss += GAUSS_WEIGHTS[k / 2] * pair;
        }
    }
    (kronrod * half, gauss * half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomials_up_to_degree_22_are_exact() {
        // The 15-point Kronrod rule extending the 7-point Gauss one is
        // exact up to degree 3 * 7 + 1.
        let f = |x: f64| x.powi(22);
        let exact = 1.0 / 23.0;
        let value = integrate(&f, 0.0, 1.0, 1e-15);
        assert!((value - exact).abs() <= 1e-15, "{value} vs {exact}");
    }

    #[test]
    fn test_smooth_integrals_meet_the_error() {
        type Case<'a> = (&'a dyn Fn(f64) -> f64, f64, f64, f64);
        let cases: [Case; 3] = [
            (&|x: f64| x.sin(), 0.0, std::f64::consts::PI, 2.0),
            (
                &|x: f64| (1.0 + x * x).recip(),
                -20.0,
                20.0,
                2.0 * 20f64.atan(),
            ),
            (&|x: f64| x.sqrt(), 0.0, 1.0, 2.0 / 3.0),
        ];
        for (f, a, b, exact) in cases {
            for max_error in [1e-6, 1e-10, 1e-13] {
                let value = integrate(&f, a, b, max_error);
                assert!((value - exact).abs() <= max_error, "{value} vs {exact}");
            }
        }
    }

    #[test]
    fn test_any_error_ends() {
        let f = |x: f64| x.abs().sqrt();
        for max_error in [0.0, -1.0, f64::NAN] {
            let value = integrate(&f, -1.0, 1.0, max_error);
            assert!((value - 4.0 / 3.0).abs() <= 1e-12, "{value}");
        }
    }
}
