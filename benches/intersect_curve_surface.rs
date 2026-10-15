//! Cost of the numeric curve–surface intersection on B-splines.
//!
//! Times the routed `intersect`, which takes the numeric path for these
//! pairs, for 64 cubic B-spline curves against one
//! bicubic B-spline surface. The surface is a 6×6 net over `[0, 5]^2`
//! with gentle height variation; each curve interpolates six points rising
//! from `z = -2` to `z = 2` along a wavy path between two random points of
//! the surface's footprint, so most curves cross it once. Then the same
//! curves against a cylinder of radius 2 about the vertical axis through
//! the footprint's centre, and 64 circles standing across the surface,
//! each crossing it about twice.
//!
//! Informational only, like `eval_throughput`; not a correctness gate.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::surfaces::BSplineSurface;
use geomcore::{BSplineCurve3D, Circle3D, Cylinder, Intersect, Point3D, Tolerance, Vector3D};

const CURVES: usize = 64;

/// A deterministic stream of numbers in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn surface() -> BSplineSurface {
    let poles = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| Point3D::new(i as f64, j as f64, ((i * j) as f64 * 0.7).sin()))
                .collect()
        })
        .collect();
    let (knots, mults) = (vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4]);
    BSplineSurface::new(
        3,
        3,
        poles,
        knots.clone(),
        mults.clone(),
        knots,
        mults,
        false,
        false,
    )
    .unwrap()
}

fn curves() -> Vec<BSplineCurve3D> {
    let mut rng = Lcg(42);
    (0..CURVES)
        .map(|_| {
            let (x0, y0) = (0.3 + rng.next() * 4.4, 0.3 + rng.next() * 4.4);
            let (x1, y1) = (0.3 + rng.next() * 4.4, 0.3 + rng.next() * 4.4);
            let poles = (0..6)
                .map(|k| {
                    let s = k as f64 / 5.0;
                    Point3D::new(
                        x0 + (x1 - x0) * s + 0.3 * (7.0 * s).sin(),
                        y0 + (y1 - y0) * s,
                        -2.0 + 4.0 * s,
                    )
                })
                .collect();
            BSplineCurve3D::new(3, poles, vec![0.0, 1.0, 2.0, 3.0], vec![4, 1, 1, 4], false)
                .unwrap()
        })
        .collect()
}

/// Circles of radius 0.6 to 1.2 centred over the surface's footprint, in
/// vertical planes of random directions.
fn circles() -> Vec<Circle3D> {
    let mut rng = Lcg(7);
    (0..CURVES)
        .map(|_| {
            let centre = Point3D::new(1.2 + rng.next() * 2.6, 1.2 + rng.next() * 2.6, 0.0);
            let angle = rng.next() * std::f64::consts::PI;
            let normal = Vector3D::new(angle.cos(), angle.sin(), 0.0);
            Circle3D::new(centre, normal, 0.6 + 0.6 * rng.next()).unwrap()
        })
        .collect()
}

fn bench_intersect_curve_surface(c: &mut Criterion) {
    let surface = surface();
    let curves = curves();
    let tol = Tolerance::default();
    let mut group = c.benchmark_group("intersect_curve_surface");
    group.throughput(Throughput::Elements(CURVES as u64));
    group.bench_function("bspline_curve_bspline_surface", |b| {
        b.iter(|| {
            for curve in black_box(&curves) {
                black_box(curve.intersect(&surface, tol));
            }
        })
    });
    let cylinder = Cylinder::new(Point3D::new(2.5, 2.5, 0.0), Vector3D::Z, 2.0).unwrap();
    group.bench_function("bspline_curve_cylinder", |b| {
        b.iter(|| {
            for curve in black_box(&curves) {
                black_box(curve.intersect(&cylinder, tol));
            }
        })
    });
    let circles = circles();
    group.bench_function("circle_bspline_surface", |b| {
        b.iter(|| {
            for circle in black_box(&circles) {
                black_box(circle.intersect(&surface, tol));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_intersect_curve_surface);
criterion_main!(benches);
