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

/// Evaluates the surface points on the tensor grid `us x vs`, returned
/// row-major (`us` outer, `vs` inner).
///
/// The basis functions of each `u` and each `v` are computed once and
/// shared across the grid, so a point costs a single local tensor sum.
#[allow(clippy::too_many_arguments)] // tensor-product evaluation signature
pub(crate) fn surface_grid(
    us: &[f64],
    vs: &[f64],
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
) -> Vec<Point3D> {
    debug_assert_eq!(poles_homog.len(), n_u * n_v * h);
    if u_degree.max(v_degree) < curve::SMALL {
        grid::<{ curve::SMALL }>(
            us,
            vs,
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
        grid::<{ curve::LARGE }>(
            us,
            vs,
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
fn grid<const N: usize>(
    us: &[f64],
    vs: &[f64],
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
) -> Vec<Point3D> {
    let v_windows: Vec<curve::Window<N>> = vs
        .iter()
        .map(|&v| {
            let mut w = curve::Window::<N>::new();
            curve::window(v, v_degree, v_periodic, v_flat, n_v, 0, &mut w);
            w
        })
        .collect();
    let mut out = Vec::with_capacity(us.len() * vs.len());
    let mut wu = curve::Window::<N>::new();
    for &u in us {
        curve::window(u, u_degree, u_periodic, u_flat, n_u, 0, &mut wu);
        for wv in &v_windows {
            let [hs, ..] = if h == 3 {
                sum::<N, 3, 0>(&wu, wv, u_degree, v_degree, poles_homog, n_v)
            } else {
                sum::<N, 4, 0>(&wu, wv, u_degree, v_degree, poles_homog, n_v)
            };
            out.push(if h == 4 {
                Point3D::new(hs[0] / hs[3], hs[1] / hs[3], hs[2] / hs[3])
            } else {
                Point3D::new(hs[0], hs[1], hs[2])
            });
        }
    }
    out
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
