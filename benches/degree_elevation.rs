//! Degree elevation of B-spline curves and surfaces.
//!
//! Clamped curves of degree 2 and 3 with 32 poles and simple interior
//! knots, polynomial and rational, raised by 1 and by 3, and a bicubic 6×6
//! surface raised by 1 in `u`. Times include building the returned
//! B-spline.

use criterion::{Criterion, criterion_group, criterion_main};
use std::f64::consts::TAU;
use std::hint::black_box;

use geomcore::{BSplineCurve3D, BSplineSurface, Point3D};

fn curve(degree: usize, rational: bool) -> BSplineCurve3D {
    let poles: Vec<Point3D> = (0..32)
        .map(|i| {
            let t = i as f64 / 31.0 * TAU;
            Point3D::new(3.0 * t.cos(), 3.0 * t.sin(), 0.4 * i as f64)
        })
        .collect();
    let spans = 32 - degree;
    let knots: Vec<f64> = (0..=spans).map(|i| i as f64).collect();
    let mut mults = vec![1u32; spans + 1];
    mults[0] = degree as u32 + 1;
    mults[spans] = degree as u32 + 1;
    if rational {
        let weights = (0..32).map(|i| 0.6 + 0.1 * (i % 7) as f64).collect();
        BSplineCurve3D::new_rational(degree, poles, weights, knots, mults, false).unwrap()
    } else {
        BSplineCurve3D::new(degree, poles, knots, mults, false).unwrap()
    }
}

fn surface() -> BSplineSurface {
    let poles: Vec<Vec<Point3D>> = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| Point3D::new(i as f64, j as f64, ((i * j) as f64 * 0.7).sin()))
                .collect()
        })
        .collect();
    let knots = vec![0.0, 1.0, 2.0, 3.0];
    let mults = vec![4u32, 1, 1, 4];
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

fn bench_degree_elevation(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree_elevation");
    for degree in [2, 3] {
        for rational in [false, true] {
            let curve = curve(degree, rational);
            let kind = if rational { "rational" } else { "polynomial" };
            for amount in [1, 3] {
                group.bench_function(format!("degree{degree}_{kind}_32_poles_by_{amount}"), |b| {
                    b.iter(|| black_box(&curve).elevate_degree(black_box(amount)).unwrap())
                });
            }
        }
    }
    let surface = surface();
    group.bench_function("bicubic_6x6_u_by_1", |b| {
        b.iter(|| black_box(&surface).elevate_u_degree(black_box(1)).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_degree_elevation);
criterion_main!(benches);
