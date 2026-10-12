//! Raising the degree of a B-spline curve or surface direction leaves the
//! shape unchanged: the raised B-spline evaluates to the same points and
//! first and second derivatives, its knots keep their values and their
//! multiplicities rise by the amount, for polynomial and rational, clamped,
//! unclamped and periodic B-splines.

use std::f64::consts::{FRAC_1_SQRT_2, TAU};

use geomcore::{BSplineCurve3D, BSplineSurface, DegreeElevationError, Point3D, Vector3D};

/// Pseudo-random coordinates in `[-2, 2]`, reproducible.
fn noise(i: usize) -> f64 {
    ((i * 7919 + 13) % 1009) as f64 / 1009.0 * 4.0 - 2.0
}

fn noisy_poles(n: usize) -> Vec<Point3D> {
    (0..n)
        .map(|i| Point3D::new(i as f64 + noise(3 * i), noise(3 * i + 1), noise(3 * i + 2)))
        .collect()
}

/// Diagonal of the poles' bounding box: the size errors are measured
/// against.
fn size(poles: &[Point3D]) -> f64 {
    let (mut lo, mut hi) = (poles[0], poles[0]);
    for p in poles {
        lo = Point3D::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = Point3D::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    hi.distance(lo).max(1.0)
}

fn assert_close(a: Vector3D, b: Vector3D, scale: f64, what: &str) {
    let tol = 1e-12 * (scale + a.magnitude());
    assert!((a - b).magnitude() <= tol, "{what}: {a:?} != {b:?}");
}

/// Parameters across `[first, last]`, every knot, and for a periodic curve
/// some beyond the bounds.
fn curve_params(curve: &BSplineCurve3D) -> Vec<f64> {
    let (first, last) = curve.bounds();
    let mut params: Vec<f64> = (0..=97)
        .map(|i| first + (last - first) * i as f64 / 97.0)
        .collect();
    params.extend(
        curve
            .knots()
            .iter()
            .filter(|&&k| (first..=last).contains(&k)),
    );
    if curve.is_periodic() {
        let period = last - first;
        params.extend([
            first - 0.3 * period,
            last + 0.45 * period,
            last + 2.0 * period,
        ]);
    }
    params
}

fn assert_same_curve(raised: &BSplineCurve3D, original: &BSplineCurve3D) {
    assert_eq!(raised.bounds(), original.bounds());
    assert_eq!(raised.is_periodic(), original.is_periodic());
    assert_eq!(raised.is_rational(), original.is_rational());
    let scale = size(original.poles());
    for u in curve_params(original) {
        let (p, d1, d2) = original.eval_d2(u);
        let (q, e1, e2) = raised.eval_d2(u);
        assert_close(
            p - Point3D::ORIGIN,
            q - Point3D::ORIGIN,
            scale,
            &format!("point at {u}"),
        );
        assert_close(d1, e1, scale, &format!("first derivative at {u}"));
        assert_close(d2, e2, scale, &format!("second derivative at {u}"));
    }
}

/// Raises `curve` by each amount and checks the shape, the degree, the
/// knots, the multiplicities and the pole count.
fn assert_elevates(curve: &BSplineCurve3D, amounts: &[usize]) {
    let spans = curve.knots().len() - 1;
    for &amount in amounts {
        let raised = curve.elevate_degree(amount).unwrap();
        assert_eq!(raised.degree(), curve.degree() + amount);
        assert_eq!(raised.knots(), curve.knots());
        let expected: Vec<u32> = curve
            .multiplicities()
            .iter()
            .map(|&m| m + amount as u32)
            .collect();
        assert_eq!(raised.multiplicities(), expected.as_slice());
        assert_eq!(raised.poles().len(), curve.poles().len() + amount * spans);
        assert_eq!(
            raised.weights().map(<[f64]>::len),
            curve.is_rational().then_some(raised.poles().len())
        );
        assert_same_curve(&raised, curve);
    }
}

fn clamped_cubic() -> BSplineCurve3D {
    BSplineCurve3D::new(
        3,
        noisy_poles(11),
        vec![0.0, 0.5, 1.2, 2.0, 3.1, 4.0],
        vec![4, 1, 2, 3, 1, 4],
        false,
    )
    .unwrap()
}

#[test]
fn test_clamped_polynomial_curves_by_any_amount() {
    assert_elevates(&clamped_cubic(), &[1, 2, 3, 5]);
    let quadratic = BSplineCurve3D::new(
        2,
        noisy_poles(32),
        (0..=30).map(|i| i as f64 * 0.37).collect(),
        [vec![3], vec![1; 29], vec![3]].concat(),
        false,
    )
    .unwrap();
    assert_elevates(&quadratic, &[1, 3, 4]);
    let quintic = BSplineCurve3D::new(
        5,
        noisy_poles(12),
        vec![-1.0, 0.0, 0.25, 2.0, 2.5, 3.0],
        vec![6, 1, 2, 1, 2, 6],
        false,
    )
    .unwrap();
    assert_elevates(&quintic, &[1, 2, 6]);
}

#[test]
fn test_degree_one_and_bezier_curves() {
    let polyline = BSplineCurve3D::new(
        1,
        noisy_poles(5),
        vec![0.0, 1.0, 1.5, 3.0, 4.0],
        vec![2, 1, 1, 1, 2],
        false,
    )
    .unwrap();
    assert_elevates(&polyline, &[1, 2, 4]);
    let bezier = BSplineCurve3D::new(4, noisy_poles(5), vec![2.0, 3.0], vec![5, 5], false).unwrap();
    assert_elevates(&bezier, &[1, 3]);
}

#[test]
fn test_raising_a_segment_adds_its_midpoint() {
    let a = Point3D::new(1.0, 2.0, 3.0);
    let b = Point3D::new(5.0, -2.0, 1.0);
    let segment = BSplineCurve3D::new(1, vec![a, b], vec![0.0, 1.0], vec![2, 2], false).unwrap();
    let raised = segment.elevate_degree(1).unwrap();
    assert_eq!(raised.poles(), &[a, Point3D::new(3.0, 0.0, 2.0), b]);
}

#[test]
fn test_rational_curves() {
    // Full circle of radius 2: four rational quadratic quarters.
    let w = FRAC_1_SQRT_2;
    let corners = [
        (2.0, 0.0),
        (2.0, 2.0),
        (0.0, 2.0),
        (-2.0, 2.0),
        (-2.0, 0.0),
        (-2.0, -2.0),
        (0.0, -2.0),
        (2.0, -2.0),
        (2.0, 0.0),
    ];
    let circle = BSplineCurve3D::new_rational(
        2,
        corners
            .iter()
            .map(|&(x, y)| Point3D::new(x, y, 0.0))
            .collect(),
        vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![3, 2, 2, 2, 3],
        false,
    )
    .unwrap();
    assert_elevates(&circle, &[1, 2, 4]);
    let raised = circle.elevate_degree(3).unwrap();
    for i in 0..=200 {
        let p = raised.eval_point(4.0 * i as f64 / 200.0);
        assert!((p.x.hypot(p.y) - 2.0).abs() < 1e-12, "{p:?}");
    }

    let weights = (0..11)
        .map(|i| 0.4 + 0.15 * (i % 5) as f64 + 0.3 * (i % 2) as f64)
        .collect();
    let curve = BSplineCurve3D::new_rational(
        3,
        noisy_poles(11),
        weights,
        vec![0.0, 0.5, 1.2, 2.0, 3.1, 4.0],
        vec![4, 1, 2, 3, 1, 4],
        false,
    )
    .unwrap();
    assert_elevates(&curve, &[1, 2, 5]);
}

#[test]
fn test_periodic_curves() {
    let ring = |n: usize| -> Vec<Point3D> {
        (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                Point3D::new(
                    3.0 * a.cos(),
                    2.0 * a.sin() + 0.3 * noise(i),
                    0.5 * noise(i + 50),
                )
            })
            .collect()
    };
    // Seam and interior knots simple.
    let cubic = BSplineCurve3D::new(
        3,
        ring(6),
        (0..7).map(f64::from).collect(),
        vec![1; 7],
        true,
    )
    .unwrap();
    assert_elevates(&cubic, &[1, 2, 3, 4]);
    // A seam of multiplicity 2, a double interior knot, uneven spans.
    let mults = vec![2, 1, 2, 1, 1, 2];
    let curve = BSplineCurve3D::new(
        3,
        ring(7),
        vec![-0.5, 0.25, 1.0, 2.5, 2.75, 4.0],
        mults.clone(),
        true,
    )
    .unwrap();
    assert_elevates(&curve, &[1, 2, 3]);
    let weights = (0..7).map(|i| 0.6 + 0.25 * (i % 3) as f64).collect();
    let rational = BSplineCurve3D::new_rational(
        3,
        ring(7),
        weights,
        vec![-0.5, 0.25, 1.0, 2.5, 2.75, 4.0],
        mults,
        true,
    )
    .unwrap();
    assert_elevates(&rational, &[1, 2, 3]);
    // A seam that is already a crease (multiplicity = degree).
    let creased = BSplineCurve3D::new(
        2,
        ring(5),
        vec![0.0, 1.0, 1.5, 3.0, 4.0],
        vec![2, 1, 1, 1, 2],
        true,
    )
    .unwrap();
    assert_elevates(&creased, &[1, 3]);
    // Few spans, high degree: the sweep starts up to a period back, and
    // more for the degree 7.
    let one_span = BSplineCurve3D::new(4, ring(3), vec![0.0, 1.0], vec![3, 3], true).unwrap();
    assert_elevates(&one_span, &[1, 2]);
    let two_spans =
        BSplineCurve3D::new(5, ring(2), vec![0.0, 0.7, 2.0], vec![1, 1, 1], true).unwrap();
    assert_elevates(&two_spans, &[1, 3]);
    let septic = BSplineCurve3D::new(7, ring(2), vec![0.0, 0.7, 2.0], vec![1, 1, 1], true).unwrap();
    assert_elevates(&septic, &[1, 2]);
}

#[test]
fn test_unclamped_curve_comes_back_clamped_at_its_bounds() {
    let curve = BSplineCurve3D::new(
        3,
        noisy_poles(8),
        (0..12).map(f64::from).collect(),
        vec![1; 12],
        false,
    )
    .unwrap();
    assert_eq!(curve.bounds(), (3.0, 8.0));
    for amount in [1, 2] {
        let raised = curve.elevate_degree(amount).unwrap();
        let q = 3 + amount as u32;
        assert_eq!(raised.knots(), &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let interior = 1 + amount as u32;
        assert_eq!(
            raised.multiplicities(),
            &[q + 1, interior, interior, interior, interior, q + 1]
        );
        assert_same_curve(&raised, &curve);
    }
}

#[test]
fn test_one_raise_equals_successive_raises() {
    let curve = clamped_cubic();
    let once = curve.elevate_degree(3).unwrap();
    let steps = curve.elevate_degree(1).unwrap().elevate_degree(2).unwrap();
    assert_eq!(once.multiplicities(), steps.multiplicities());
    let scale = size(curve.poles());
    for (a, b) in once.poles().iter().zip(steps.poles()) {
        assert!(a.distance(*b) < 1e-12 * scale, "{a:?} vs {b:?}");
    }
}

#[test]
fn test_zero_amount_returns_a_copy_and_degree_25_is_the_limit() {
    let curve = clamped_cubic();
    assert_eq!(curve.elevate_degree(0).unwrap(), curve);
    let top = curve.elevate_degree(22).unwrap();
    assert_eq!(top.degree(), 25);
    assert_eq!(
        curve.elevate_degree(23),
        Err(DegreeElevationError::DegreeTooHigh)
    );
    assert_eq!(
        top.elevate_degree(1),
        Err(DegreeElevationError::DegreeTooHigh)
    );
    assert_eq!(
        curve.elevate_degree(usize::MAX),
        Err(DegreeElevationError::DegreeTooHigh)
    );
    assert!(!DegreeElevationError::DegreeTooHigh.to_string().is_empty());
    let (first, last) = curve.bounds();
    let scale = size(curve.poles());
    for i in 0..=50 {
        let u = first + (last - first) * i as f64 / 50.0;
        assert!(top.eval_point(u).distance(curve.eval_point(u)) < 1e-12 * scale);
    }
}

// ---- surfaces ----

fn surface_params(surface: &BSplineSurface) -> Vec<(f64, f64)> {
    use geomcore::ParametricSurface;
    let (u0, u1) = surface.u_bounds();
    let (v0, v1) = surface.v_bounds();
    let mut params = Vec::new();
    for i in 0..=13 {
        for j in 0..=11 {
            params.push((
                u0 + (u1 - u0) * i as f64 / 13.0,
                v0 + (v1 - v0) * j as f64 / 11.0,
            ));
        }
    }
    params
}

fn assert_same_surface(raised: &BSplineSurface, original: &BSplineSurface, scale: f64) {
    use geomcore::ParametricSurface;
    assert_eq!(raised.u_bounds(), original.u_bounds());
    assert_eq!(raised.v_bounds(), original.v_bounds());
    assert_eq!(raised.u_period(), original.u_period());
    assert_eq!(raised.v_period(), original.v_period());
    assert_eq!(raised.is_rational(), original.is_rational());
    for (u, v) in surface_params(original) {
        let (p, d1, d2) = original.eval_d2(u, v);
        let (q, e1, e2) = raised.eval_d2(u, v);
        let at = format!("at ({u}, {v})");
        assert_close(p - Point3D::ORIGIN, q - Point3D::ORIGIN, scale, &at);
        for (a, b) in d1.into_iter().zip(e1).chain(d2.into_iter().zip(e2)) {
            assert_close(a, b, scale, &at);
        }
    }
}

fn assert_surface_elevates(surface: &BSplineSurface, scale: f64) {
    for amount in [1, 2, 3] {
        let u = surface.elevate_u_degree(amount).unwrap();
        assert_eq!(
            (u.u_degree(), u.v_degree()),
            (surface.u_degree() + amount, surface.v_degree())
        );
        assert_eq!(
            (u.is_u_periodic(), u.is_v_periodic()),
            (surface.is_u_periodic(), surface.is_v_periodic())
        );
        assert_same_surface(&u, surface, scale);
        let v = surface.elevate_v_degree(amount).unwrap();
        assert_eq!(
            (v.u_degree(), v.v_degree()),
            (surface.u_degree(), surface.v_degree() + amount)
        );
        assert_same_surface(&v, surface, scale);
    }
    let both = surface
        .elevate_u_degree(1)
        .unwrap()
        .elevate_v_degree(2)
        .unwrap();
    assert_same_surface(&both, surface, scale);
}

fn net(rows: usize, cols: usize) -> Vec<Vec<Point3D>> {
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| Point3D::new(i as f64, j as f64, noise(i * cols + j)))
                .collect()
        })
        .collect()
}

#[test]
fn test_clamped_surfaces() {
    let polynomial = BSplineSurface::new(
        3,
        2,
        net(7, 6),
        vec![0.0, 1.0, 1.5, 3.0],
        vec![4, 2, 1, 4],
        vec![0.0, 0.4, 1.0, 2.0],
        vec![3, 1, 2, 3],
        false,
        false,
    )
    .unwrap();
    assert_surface_elevates(&polynomial, 8.0);
    let weights = (0..7)
        .map(|i| {
            (0..6)
                .map(|j| 0.5 + 0.2 * ((i + 2 * j) % 4) as f64)
                .collect()
        })
        .collect();
    let rational = BSplineSurface::new_rational(
        3,
        2,
        net(7, 6),
        weights,
        vec![0.0, 1.0, 1.5, 3.0],
        vec![4, 2, 1, 4],
        vec![0.0, 0.4, 1.0, 2.0],
        vec![3, 1, 2, 3],
        false,
        false,
    )
    .unwrap();
    assert_surface_elevates(&rational, 8.0);
    assert_eq!(
        rational.elevate_u_degree(23).unwrap_err(),
        DegreeElevationError::DegreeTooHigh
    );
    assert_eq!(
        rational.elevate_v_degree(24).unwrap_err(),
        DegreeElevationError::DegreeTooHigh
    );
    assert_eq!(rational.elevate_v_degree(0).unwrap(), rational);
}

#[test]
fn test_periodic_surfaces() {
    // Closed in u (a tube), then closed in v (its transpose).
    let tube: Vec<Vec<Point3D>> = (0..7)
        .map(|i| {
            let a = TAU * i as f64 / 7.0;
            (0..4)
                .map(|j| {
                    let r = 2.0 + 0.3 * noise(4 * i + j);
                    Point3D::new(r * a.cos(), r * a.sin(), j as f64)
                })
                .collect()
        })
        .collect();
    let u_closed = BSplineSurface::new(
        3,
        2,
        tube.clone(),
        vec![0.0, 1.0, 2.0, 2.5, 4.0, 5.0, 6.0],
        vec![1, 1, 2, 1, 1, 1, 1],
        vec![0.0, 1.0, 2.0],
        vec![3, 1, 3],
        true,
        false,
    )
    .unwrap();
    assert_surface_elevates(&u_closed, 6.0);
    let transposed: Vec<Vec<Point3D>> = (0..4)
        .map(|j| tube.iter().map(|row| row[j]).collect())
        .collect();
    let weights = (0..4)
        .map(|i| (0..7).map(|j| 0.7 + 0.2 * ((i + j) % 3) as f64).collect())
        .collect();
    let v_closed = BSplineSurface::new_rational(
        2,
        3,
        transposed,
        weights,
        vec![0.0, 1.0, 2.0],
        vec![3, 1, 3],
        vec![0.0, 1.0, 2.0, 2.5, 4.0, 5.0, 6.0],
        vec![1, 1, 2, 1, 1, 1, 1],
        false,
        true,
    )
    .unwrap();
    assert_surface_elevates(&v_closed, 6.0);
}
