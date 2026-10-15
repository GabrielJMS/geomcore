//! Cost of the marching intersection.
//!
//! Times `marching_intersection` on two bicubic 6×6 B-spline surfaces over
//! `[0, 5]^2`, with heights `sin(0.7 i j)` and `0.5 cos(1.3 i) cos(1.1 j)`
//! at their poles `(i, j)`. They cross in six curves: five open across the
//! footprint and one closed loop. Also times a torus of radii 3 and 1
//! about z against the plane through `(0.5, 0, 0.2)` tilted 0.6 radians
//! about y, which cuts it in two closed curves.
//!
//! Informational only, like `eval_throughput`; not a correctness gate.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::surfaces::BSplineSurface;
use geomcore::{Plane, Point3D, Tolerance, Torus, Vector3D, marching_intersection};

fn surface(height: impl Fn(f64, f64) -> f64) -> BSplineSurface {
    let poles = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| {
                    let (x, y) = (i as f64, j as f64);
                    Point3D::new(x, y, height(x, y))
                })
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

fn bench_marching_intersection(c: &mut Criterion) {
    let wavy = surface(|x, y| (0.7 * x * y).sin());
    let bumpy = surface(|x, y| 0.5 * (1.3 * x).cos() * (1.1 * y).cos());
    let tol = Tolerance::default();
    let mut group = c.benchmark_group("marching_intersection");
    group.bench_function("bspline_surface_pair", |b| {
        b.iter(|| black_box(marching_intersection(black_box(&wavy), &bumpy, tol)))
    });
    let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    let tilt: f64 = 0.6;
    let plane = Plane::new(
        Point3D::new(0.5, 0.0, 0.2),
        Vector3D::new(-tilt.sin(), 0.0, tilt.cos()),
    )
    .unwrap();
    group.bench_function("torus_plane_oblique", |b| {
        b.iter(|| black_box(marching_intersection(black_box(&torus), &plane, tol)))
    });
    group.finish();
}

criterion_group!(benches, bench_marching_intersection);
criterion_main!(benches);
