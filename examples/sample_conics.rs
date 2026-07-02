//! Samples a circle and an ellipse over a full turn, and demonstrates
//! analytic curve-on-surface parametrization by placing a circle on a
//! cylinder and reading off its `(u, v)` image.
//!
//! Run with `cargo run --example sample_conics`.

use geomrust::curves::{Circle3D, Curve2D, Ellipse3D, ParametricCurve2D};
use geomrust::{Cylinder, Point3, Vector3};
use std::f64::consts::TAU;

const SAMPLE_COUNT: usize = 100;

fn main() {
    let params: Vec<f64> = (0..SAMPLE_COUNT)
        .map(|i| TAU * i as f64 / SAMPLE_COUNT as f64)
        .collect();

    let circle = Circle3D::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    let circle_points = circle.eval_points(&params);

    let ellipse = Ellipse3D::new(Point3::ORIGIN, Vector3::Z, Vector3::X, 3.0, 1.5).unwrap();
    let ellipse_points = ellipse.eval_points(&params);

    println!("Circle (radius 2.0), first/last 3 of {SAMPLE_COUNT} points over [0, TAU):");
    print_head_and_tail(&params, &circle_points);

    println!(
        "\nEllipse (major 3.0, minor 1.5), first/last 3 of {SAMPLE_COUNT} points over [0, TAU):"
    );
    print_head_and_tail(&params, &ellipse_points);

    // Curve-on-surface parametrization: a circle coaxial with a cylinder
    // has an exact 2D image, a horizontal line at v = height in (u, v).
    let cylinder = Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
    let section = Circle3D::new(Point3::new(0.0, 0.0, 3.0), Vector3::Z, 2.0).unwrap();
    let pcurve = section.parametrize_on(cylinder).unwrap();
    match &pcurve {
        Curve2D::Line(line) => {
            println!(
                "\nCircle-on-cylinder pcurve: (u, v) = ({:.3}, {:.3}) + t * ({:.3}, {:.3})",
                line.origin().x,
                line.origin().y,
                line.direction().x,
                line.direction().y
            );
        }
        _ => unreachable!("a coaxial circle on a cylinder always maps to a line"),
    }
    let q = pcurve.eval_point(1.0);
    println!("pcurve(1.0) = ({:.6}, {:.6})", q.x, q.y);
}

fn print_head_and_tail(params: &[f64], points: &[Point3]) {
    let n = points.len();
    for i in 0..3 {
        println!("  u={:.4}  ->  {:?}", params[i], points[i]);
    }
    println!("  ...");
    for i in n - 3..n {
        println!("  u={:.4}  ->  {:?}", params[i], points[i]);
    }
}
