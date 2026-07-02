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
//! Evaluation reuses the curve evaluator [`eval_dn`] twice, in two passes:
//!
//! 1. **u-pass** — treat each u-index's *entire* v-row (`n_v * h` floats) as a
//!    single "mega-point" of dimension `n_v * h`, and run [`eval_dn`] in `u`
//!    over the `n_u` mega-points at degree `u_degree`. The curve engine's
//!    per-coordinate recursion and (for periodic curves) modulo-pole-count
//!    wrapping are dimension-agnostic, so a mega-point is handled exactly like
//!    an ordinary pole — the periodic wrap is over the *mega-point count*
//!    `n_u`, which is what we want. This produces `nu + 1` intermediate
//!    v-curves (the value curve and, for `nu == 1`, the `u`-derivative curve),
//!    each with `n_v` poles of stride `h`.
//! 2. **v-pass** — run [`eval_dn`] in `v` at degree `v_degree` on each
//!    intermediate v-curve. The value curve is evaluated to order `nv` (giving
//!    the homogeneous value and, for `nv == 1`, the `v`-derivative); the
//!    `u`-derivative curve is evaluated to order `0` (giving the homogeneous
//!    `u`-derivative).
//!
//! For rational surfaces the whole computation is carried out in homogeneous
//! coordinates (`h = 4`), and the quotient rule is applied per output at the
//! very end: `S = H/w`, `Su = (Hu_xyz - S*Hu_w)/w`, `Sv = (Hv_xyz - S*Hv_w)/w`.

use super::super::curve_math::bspline as curve;
use crate::{Point3, Vector3};

/// Evaluates the surface point `S(u, v)`.
///
/// `poles_homog` is the flat, row-major `[i_u][i_v][h]` pole buffer with `h`
/// coordinates per pole (`h = 3` polynomial, `h = 4` homogeneous). `u_flat` /
/// `v_flat` are the plain [`curve::flat_knots`] of each direction. See the
/// module docs for the two-pass strategy.
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
) -> Point3 {
    debug_assert_eq!(poles_homog.len(), n_u * n_v * h);
    // u-pass, order 0: collapse the u direction into one v-curve of homogeneous
    // poles.
    let mega = n_v * h;
    let mut v_curve = vec![0.0f64; mega];
    curve::eval_dn(
        u,
        u_degree,
        u_periodic,
        u_flat,
        poles_homog,
        mega,
        0,
        &mut v_curve,
    );

    // v-pass, order 0: evaluate the resulting v-curve at v.
    let mut homog = vec![0.0f64; h];
    curve::eval_dn(v, v_degree, v_periodic, v_flat, &v_curve, h, 0, &mut homog);

    if h == 4 {
        let w = homog[3];
        Point3::new(homog[0] / w, homog[1] / w, homog[2] / w)
    } else {
        Point3::new(homog[0], homog[1], homog[2])
    }
}

/// Evaluates the surface value and both first partial derivatives,
/// returning `(S, Su, Sv)`.
///
/// Layout of `poles_homog` and the knot flats is as for [`surface_d0`]. See
/// the module docs for the two-pass strategy; the rational quotient rule is
/// applied inline here (the homogeneous layout of the intermediate results
/// does not match [`curve::rational_derivatives`]'s contiguous
/// `H, H', ...` expectation, so we compute the first-order quotient directly).
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
) -> (Point3, Vector3, Vector3) {
    debug_assert_eq!(poles_homog.len(), n_u * n_v * h);
    let mega = n_v * h;

    // u-pass, order 1: two v-curves — value (k=0) and u-derivative (k=1).
    let mut u_pass = vec![0.0f64; 2 * mega];
    curve::eval_dn(
        u,
        u_degree,
        u_periodic,
        u_flat,
        poles_homog,
        mega,
        1,
        &mut u_pass,
    );
    let (value_curve, du_curve) = u_pass.split_at(mega);

    // v-pass on the value curve, order 1: homogeneous S and Sv.
    let mut val = vec![0.0f64; 2 * h];
    curve::eval_dn(v, v_degree, v_periodic, v_flat, value_curve, h, 1, &mut val);
    let h_s = &val[..h]; // homogeneous S
    let h_sv = &val[h..]; // homogeneous Sv

    // v-pass on the u-derivative curve, order 0: homogeneous Su.
    let mut du = vec![0.0f64; h];
    curve::eval_dn(v, v_degree, v_periodic, v_flat, du_curve, h, 0, &mut du);
    let h_su = &du[..]; // homogeneous Su

    if h == 4 {
        // Rational quotient rule, first order:
        //   S  = H / w
        //   Su = (Hu_xyz - S * Hu_w) / w
        //   Sv = (Hv_xyz - S * Hv_w) / w
        let w = h_s[3];
        let s = Point3::new(h_s[0] / w, h_s[1] / w, h_s[2] / w);
        let su = Vector3::new(
            (h_su[0] - s.x * h_su[3]) / w,
            (h_su[1] - s.y * h_su[3]) / w,
            (h_su[2] - s.z * h_su[3]) / w,
        );
        let sv = Vector3::new(
            (h_sv[0] - s.x * h_sv[3]) / w,
            (h_sv[1] - s.y * h_sv[3]) / w,
            (h_sv[2] - s.z * h_sv[3]) / w,
        );
        (s, su, sv)
    } else {
        let s = Point3::new(h_s[0], h_s[1], h_s[2]);
        let su = Vector3::new(h_su[0], h_su[1], h_su[2]);
        let sv = Vector3::new(h_sv[0], h_sv[1], h_sv[2]);
        (s, su, sv)
    }
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
