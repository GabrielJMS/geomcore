//! Cost of the exact conic–quadric intersections.
//!
//! Times a batch of typed `IntersectExact::intersect_exact` calls for each of the 16
//! pairs of a conic (circle, ellipse, parabola, hyperbola) and a quadric
//! (plane, sphere, cylinder, cone). The conics are placed at random near
//! the quadrics, so each batch mixes crossings, tangencies and misses.
//!
//! Informational only, like `eval_throughput`; not a correctness gate.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::curves::{Circle3D, Ellipse3D, Hyperbola3D, Parabola3D};
use geomcore::surfaces::{Cone, Cylinder, Plane, Sphere};
use geomcore::{Frame3D, IntersectExact, Point3D, Tolerance, Vector3D};

const BATCH: usize = 1024;

/// A deterministic offset in `[-1, 1)` for the `i`-th input.
fn jitter(i: usize, salt: u64) -> f64 {
    let mut x = (i as u64 ^ salt).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// A random frame near the origin, for the `i`-th conic.
fn frame(i: usize) -> Frame3D {
    let origin = Point3D::new(1.5 * jitter(i, 1), 1.5 * jitter(i, 2), 1.5 * jitter(i, 3));
    let z = Vector3D::new(jitter(i, 4), jitter(i, 5), 0.2 + jitter(i, 6).abs())
        .normalized()
        .unwrap();
    let x = z.cross(Vector3D::new(0.3, 1.0, 0.2)).normalized().unwrap();
    Frame3D::new(origin, z, x).unwrap()
}

fn circles() -> Vec<Circle3D> {
    (0..BATCH)
        .map(|i| {
            let f = frame(i);
            Circle3D::new(f.origin(), f.z_direction(), 1.0 + jitter(i, 7).abs()).unwrap()
        })
        .collect()
}

fn ellipses() -> Vec<Ellipse3D> {
    (0..BATCH)
        .map(|i| {
            let a = 1.5 + jitter(i, 7).abs();
            Ellipse3D::from_frame(frame(i), a, a * (0.3 + 0.6 * jitter(i, 8).abs())).unwrap()
        })
        .collect()
}

fn parabolas() -> Vec<Parabola3D> {
    (0..BATCH)
        .map(|i| Parabola3D::from_frame(frame(i), 0.2 + jitter(i, 7).abs()).unwrap())
        .collect()
}

fn hyperbolas() -> Vec<Hyperbola3D> {
    (0..BATCH)
        .map(|i| {
            let a = 0.5 + jitter(i, 7).abs();
            Hyperbola3D::from_frame(frame(i), a, 0.3 + jitter(i, 8).abs()).unwrap()
        })
        .collect()
}

/// Times one conic family against each of the four quadrics.
macro_rules! bench_conic {
    ($c:expr, $name:literal, $conics:expr) => {{
        let conics = $conics;
        let tol = Tolerance::default();
        let plane = Plane::new(Point3D::new(0.0, 0.0, 0.3), Vector3D::new(0.2, 0.1, 1.0)).unwrap();
        let sphere = Sphere::new(Point3D::ORIGIN, 1.8).unwrap();
        let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap();
        let cone = Cone::from_frame(Frame3D::WORLD, 0.5, 1.2).unwrap();
        let mut group = $c.benchmark_group(concat!("conic_quadric/", $name));
        group.throughput(Throughput::Elements(BATCH as u64));
        group.bench_function("plane", |b| {
            b.iter(|| {
                for x in black_box(&conics) {
                    black_box(x.intersect_exact(&plane, tol));
                }
            })
        });
        group.bench_function("sphere", |b| {
            b.iter(|| {
                for x in black_box(&conics) {
                    black_box(x.intersect_exact(&sphere, tol));
                }
            })
        });
        group.bench_function("cylinder", |b| {
            b.iter(|| {
                for x in black_box(&conics) {
                    black_box(x.intersect_exact(&cylinder, tol));
                }
            })
        });
        group.bench_function("cone", |b| {
            b.iter(|| {
                for x in black_box(&conics) {
                    black_box(x.intersect_exact(&cone, tol));
                }
            })
        });
        group.finish();
    }};
}

fn bench_conic_quadric(c: &mut Criterion) {
    bench_conic!(c, "circle", circles());
    bench_conic!(c, "ellipse", ellipses());
    bench_conic!(c, "parabola", parabolas());
    bench_conic!(c, "hyperbola", hyperbolas());
}

criterion_group!(benches, bench_conic_quadric);
criterion_main!(benches);
