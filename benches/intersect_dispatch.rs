//! Cost of the `IntersectExact` entry point.
//!
//! For a few representative pairs, times a batch of exact intersections
//! three ways: the typed trait call, the untyped call on `ElementaryCurve`
//! / `ElementarySurface`, and the untyped call on `Curve3D` / `Surface`.
//! The untyped calls add a `match` and the conversion to the general
//! result.
//!
//! Informational only, like `eval_throughput`; not a correctness gate.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use geomcore::curves::{Circle2D, Circle3D, Curve3D, ElementaryCurve, Ellipse3D, Line3D};
use geomcore::surfaces::{Cone, Cylinder, ElementarySurface, Plane, Sphere, Surface, Torus};
use geomcore::{Intersect, IntersectExact, Point2D, Point3D, Tolerance, Vector3D};

const BATCH: usize = 1024;

/// A deterministic offset in `[-1, 1)` for the `i`-th input.
fn jitter(i: usize, salt: u64) -> f64 {
    let mut x = (i as u64 ^ salt).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn point(i: usize, scale: f64) -> Point3D {
    Point3D::new(
        scale * jitter(i, 1),
        scale * jitter(i, 2),
        scale * jitter(i, 3),
    )
}

fn direction(i: usize) -> Vector3D {
    Vector3D::new(jitter(i, 4), jitter(i, 5), 0.2 + jitter(i, 6).abs())
}

/// Times one pair kind four ways over the same batch.
macro_rules! bench_pair {
    ($c:expr, $name:literal, $pairs:expr, $elem_a:ty, $elem_b:ty, $erased_a:ty, $erased_b:ty) => {{
        let pairs = $pairs;
        let elementary: Vec<($elem_a, $elem_b)> = pairs
            .iter()
            .map(|(a, b)| (<$elem_a>::from(*a), <$elem_b>::from(*b)))
            .collect();
        let erased: Vec<($erased_a, $erased_b)> = pairs
            .iter()
            .map(|(a, b)| (<$erased_a>::from(*a), <$erased_b>::from(*b)))
            .collect();
        let tol = Tolerance::default();
        let mut group = $c.benchmark_group(concat!("intersect/", $name));
        group.throughput(Throughput::Elements(BATCH as u64));
        group.bench_function("typed", |b| {
            b.iter(|| {
                for (x, y) in black_box(&pairs) {
                    black_box(x.intersect_exact(y, tol));
                }
            })
        });
        group.bench_function("elementary", |b| {
            b.iter(|| {
                for (x, y) in black_box(&elementary) {
                    black_box(x.intersect_exact(y, tol));
                }
            })
        });
        group.bench_function("erased", |b| {
            b.iter(|| {
                for (x, y) in black_box(&erased) {
                    black_box(x.intersect_exact(y, tol));
                }
            })
        });
        group.finish();
    }};
}

fn bench_intersect_dispatch(c: &mut Criterion) {
    // Planes at random heights against a unit sphere: circles, tangent
    // points and misses.
    let plane_sphere: Vec<(Plane, Sphere)> = (0..BATCH)
        .map(|i| {
            let plane = Plane::new(Point3D::new(0.0, 0.0, 1.2 * jitter(i, 7)), Vector3D::Z);
            (plane.unwrap(), Sphere::new(Point3D::ORIGIN, 1.0).unwrap())
        })
        .collect();
    bench_pair!(
        c,
        "plane_sphere",
        plane_sphere,
        ElementarySurface,
        ElementarySurface,
        Surface,
        Surface
    );

    // Coaxial torus and cone: a quartic in the axial coordinate.
    let torus_cone: Vec<(Torus, Cone)> = (0..BATCH)
        .map(|i| {
            let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap();
            let apex_height = 2.0 * jitter(i, 8);
            let cone = Cone::new(Point3D::new(0.0, 0.0, apex_height), Vector3D::Z, 0.6, 4.0);
            (torus, cone.unwrap())
        })
        .collect();
    bench_pair!(
        c,
        "torus_cone",
        torus_cone,
        ElementarySurface,
        ElementarySurface,
        Surface,
        Surface
    );

    // Random lines against a torus: up to four hits.
    let line_torus: Vec<(Line3D, Torus)> = (0..BATCH)
        .map(|i| {
            let line = Line3D::new(point(i, 3.0), direction(i)).unwrap();
            (
                line,
                Torus::new(Point3D::ORIGIN, Vector3D::Z, 4.0, 1.0).unwrap(),
            )
        })
        .collect();
    bench_pair!(
        c,
        "line_torus",
        line_torus,
        ElementaryCurve,
        ElementarySurface,
        Curve3D,
        Surface
    );

    // Tilted circles against a cylinder: the conic-surface quartic.
    let circle_cylinder: Vec<(Circle3D, Cylinder)> = (0..BATCH)
        .map(|i| {
            let circle = Circle3D::new(point(i, 1.0), direction(i), 2.0).unwrap();
            (
                circle,
                Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.5).unwrap(),
            )
        })
        .collect();
    bench_pair!(
        c,
        "circle_cylinder",
        circle_cylinder,
        ElementaryCurve,
        ElementarySurface,
        Curve3D,
        Surface
    );

    // Lines through the plane of a circle.
    let line_circle: Vec<(Line3D, Circle3D)> = (0..BATCH)
        .map(|i| {
            let origin = Point3D::new(3.0 * jitter(i, 9), -5.0, 0.0);
            let line = Line3D::new(origin, Vector3D::new(jitter(i, 10), 1.0, 0.0)).unwrap();
            (
                line,
                Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap(),
            )
        })
        .collect();
    bench_pair!(
        c,
        "line_circle",
        line_circle,
        ElementaryCurve,
        ElementaryCurve,
        Curve3D,
        Curve3D
    );

    // Coplanar ellipses, shifted and rotated: up to four hits.
    let ellipse_ellipse: Vec<(Ellipse3D, Ellipse3D)> = (0..BATCH)
        .map(|i| {
            let a = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.5);
            let angle = 3.0 * jitter(i, 11);
            let x_direction = Vector3D::new(angle.cos(), angle.sin(), 0.0);
            let center = Point3D::new(jitter(i, 12), jitter(i, 13), 0.0);
            let b = Ellipse3D::new(center, Vector3D::Z, x_direction, 2.5, 1.0);
            (a.unwrap(), b.unwrap())
        })
        .collect();
    bench_pair!(
        c,
        "ellipse_ellipse",
        ellipse_ellipse,
        ElementaryCurve,
        ElementaryCurve,
        Curve3D,
        Curve3D
    );
}

/// Times the routed call against the typed exact call over one batch. The
/// routed result stores only the parameters the solver computed, so the
/// typed call is the fair baseline: the gap is building the result.
macro_rules! bench_routed {
    ($c:expr, $name:literal, $pairs:expr) => {{
        let pairs = $pairs;
        let tol = Tolerance::default();
        let mut group = $c.benchmark_group(concat!("intersect_routed/", $name));
        group.throughput(Throughput::Elements(BATCH as u64));
        group.bench_function("exact", |b| {
            b.iter(|| {
                for (x, y) in black_box(&pairs) {
                    black_box(x.intersect_exact(y, tol));
                }
            })
        });
        group.bench_function("routed", |b| {
            b.iter(|| {
                for (x, y) in black_box(&pairs) {
                    black_box(x.intersect(y, tol));
                }
            })
        });
        group.finish();
    }};
}

fn bench_intersect_routed(c: &mut Criterion) {
    // Lines through the axis region of a cylinder: two hits each.
    let line_cylinder: Vec<(Line3D, Cylinder)> = (0..BATCH)
        .map(|i| {
            let origin = Point3D::new(-3.0, 0.5 * jitter(i, 14), 0.5 * jitter(i, 15));
            let line = Line3D::new(origin, Vector3D::new(1.0, 0.2 * jitter(i, 16), 0.0));
            let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.5);
            (line.unwrap(), cylinder.unwrap())
        })
        .collect();
    bench_routed!(c, "line_cylinder", line_cylinder);

    // Coplanar circles of radius 2 and 1.5 with centres 1 to 2 apart: two hits each.
    let circle_circle: Vec<(Circle3D, Circle3D)> = (0..BATCH)
        .map(|i| {
            let a = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
            let angle = 3.0 * jitter(i, 17);
            let distance = 1.5 + 0.5 * jitter(i, 18);
            let center = Point3D::new(distance * angle.cos(), distance * angle.sin(), 0.0);
            (a, Circle3D::new(center, Vector3D::Z, 1.5).unwrap())
        })
        .collect();
    bench_routed!(c, "circle_circle", circle_circle.clone());

    // The same circles in the plane.
    let circle_circle_2d: Vec<(Circle2D, Circle2D)> = circle_circle
        .iter()
        .map(|(a, b)| {
            let center = |c: &Circle3D| Point2D::new(c.center().x, c.center().y);
            let a = Circle2D::new(center(a), a.radius()).unwrap();
            (a, Circle2D::new(center(b), b.radius()).unwrap())
        })
        .collect();
    bench_routed!(c, "circle_circle_2d", circle_circle_2d);

    // Planes at heights across a unit sphere: a circle each.
    let plane_sphere: Vec<(Plane, Sphere)> = (0..BATCH)
        .map(|i| {
            let plane = Plane::new(Point3D::new(0.0, 0.0, 0.9 * jitter(i, 19)), Vector3D::Z);
            (plane.unwrap(), Sphere::new(Point3D::ORIGIN, 1.0).unwrap())
        })
        .collect();
    bench_routed!(c, "plane_sphere", plane_sphere);
}

criterion_group!(benches, bench_intersect_dispatch, bench_intersect_routed);
criterion_main!(benches);
