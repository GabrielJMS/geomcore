//! Reversed surfaces stay consistent with everything built on their
//! frame: isoparametric curves, transforms, projection and the parameter
//! maps, on every surface type in a tilted placement.

use geomcore::curves::ParametricCurve3D;
use geomcore::surfaces::{ParametricSurface, Surface};
use geomcore::{
    Axis3D, BSplineSurface, Cone, Cylinder, Frame3D, Plane, Point3D, Sphere, Tolerance, Torus,
    Transform, Vector3D,
};

const PARAMETERS: [(f64, f64); 4] = [(0.0, 0.0), (0.3, 0.2), (2.0, -0.7), (4.5, 1.1)];

fn tilted_frame() -> Frame3D {
    Frame3D::new(
        Point3D::new(1.0, -2.0, 0.5),
        Vector3D::new(0.3, -0.4, 0.86),
        Vector3D::new(1.0, 0.2, 0.0),
    )
    .unwrap()
}

fn bspline() -> BSplineSurface {
    let poles: Vec<Vec<Point3D>> = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| Point3D::new(i as f64, j as f64, ((i + 1) * (j + 2)) as f64 * 0.1))
                .collect()
        })
        .collect();
    BSplineSurface::new(
        2,
        2,
        poles,
        vec![-1.0, 5.0],
        vec![3, 3],
        vec![-2.0, 2.0],
        vec![3, 3],
        false,
        false,
    )
    .unwrap()
}

fn surfaces() -> Vec<Surface> {
    let frame = tilted_frame();
    vec![
        Surface::from(Plane::from_frame(frame)),
        Surface::from(Cylinder::from_frame(frame, 2.0).unwrap()),
        Surface::from(Cone::from_frame(frame, 0.4, 2.0).unwrap()),
        Surface::from(Sphere::from_frame(frame, 2.0).unwrap()),
        Surface::from(Torus::from_frame(frame, 5.0, 1.5).unwrap()),
        Surface::from(bspline()),
    ]
}

fn reversals(surface: &Surface) -> [Surface; 3] {
    [
        surface.reversed_u(),
        surface.reversed_v(),
        surface.reversed_u().reversed_v(),
    ]
}

fn unit_normal(surface: &Surface, u: f64, v: f64) -> Vector3D {
    let (_, [su, sv]) = surface.eval_d1(u, v);
    su.cross(sv).normalized().unwrap()
}

fn transformed(surface: &Surface, transform: Transform) -> Option<Surface> {
    Some(match surface {
        Surface::Plane(s) => Surface::Plane(s.transformed(transform).unwrap()),
        Surface::Cylinder(s) => Surface::Cylinder(s.transformed(transform).unwrap()),
        Surface::Cone(s) => Surface::Cone(s.transformed(transform).unwrap()),
        Surface::Sphere(s) => Surface::Sphere(s.transformed(transform).unwrap()),
        Surface::Torus(s) => Surface::Torus(s.transformed(transform).unwrap()),
        _ => return None,
    })
}

#[test]
fn test_parameter_maps_land_on_the_same_point_and_invert_themselves() {
    for surface in surfaces() {
        let (reversed_u, reversed_v) = (surface.reversed_u(), surface.reversed_v());
        for (u, v) in PARAMETERS {
            let (u, v) = (u.clamp(-1.0, 5.0), v.clamp(-1.0, 1.0));
            let point = surface.eval_point(u, v);
            let on_u = reversed_u.eval_point(surface.u_reversed_parameter(u), v);
            let on_v = reversed_v.eval_point(u, surface.v_reversed_parameter(v));
            assert!(on_u.distance(point) < 1e-12, "{surface:?}: u map at {u}");
            assert!(on_v.distance(point) < 1e-12, "{surface:?}: v map at {v}");
            let back_u = reversed_u.u_reversed_parameter(surface.u_reversed_parameter(u));
            let back_v = reversed_v.v_reversed_parameter(surface.v_reversed_parameter(v));
            assert!((back_u - u).abs() < 1e-14 && (back_v - v).abs() < 1e-14);
        }
    }
}

#[test]
fn test_iso_curves_of_reversed_surfaces_follow_the_reversed_parametrization() {
    for surface in surfaces() {
        for reversed in reversals(&surface) {
            for (u, v) in PARAMETERS {
                let (u, v) = (u.clamp(-1.0, 5.0), v.clamp(-1.0, 1.0));
                let (u_iso, v_iso) = (reversed.u_iso(u).unwrap(), reversed.v_iso(v).unwrap());
                for t in [-0.4, 0.0, 0.7] {
                    let along_v = u_iso.eval_point(t).distance(reversed.eval_point(u, t));
                    let along_u = v_iso.eval_point(t).distance(reversed.eval_point(t, v));
                    assert!(along_v < 1e-12, "{reversed:?}: u_iso({u}) at {t}");
                    assert!(along_u < 1e-12, "{reversed:?}: v_iso({v}) at {t}");
                }
            }
        }
    }
}

#[test]
fn test_rotation_keeps_the_reversed_parametrization() {
    let rotation = Transform::rotation(
        Axis3D::new(Point3D::new(1.0, 0.0, 0.0), Vector3D::new(0.2, 1.0, 0.3)).unwrap(),
        0.7,
    )
    .then(Transform::translation(Vector3D::new(1.0, 2.0, 3.0)));
    for surface in surfaces() {
        for reversed in reversals(&surface) {
            let Some(image) = transformed(&reversed, rotation) else {
                continue;
            };
            for (u, v) in PARAMETERS {
                let expected = rotation.apply_point(reversed.eval_point(u, v));
                assert!(image.eval_point(u, v).distance(expected) < 1e-12);
                let normal = rotation.apply_vector(unit_normal(&reversed, u, v));
                assert!((unit_normal(&image, u, v) - normal).magnitude() < 1e-12);
            }
        }
    }
}

#[test]
fn test_mirror_keeps_the_side_the_normal_points_to() {
    let mirror = Transform::mirror_plane(
        Frame3D::new(
            Point3D::new(0.2, 0.1, -0.3),
            Vector3D::new(0.5, 0.5, 0.7),
            Vector3D::new(1.0, -1.0, 0.0),
        )
        .unwrap(),
    );
    let tol = Tolerance::DEFAULT;
    for surface in surfaces() {
        for reversed in reversals(&surface) {
            let Some(image) = transformed(&reversed, mirror) else {
                continue;
            };
            for (u, v) in PARAMETERS {
                let foot = image.project_point(mirror.apply_point(reversed.eval_point(u, v)), tol);
                assert!(
                    foot.distance < 1e-9,
                    "{reversed:?}: mirrored point off the image"
                );
                let normal = mirror.apply_vector(unit_normal(&reversed, u, v));
                let image_normal = unit_normal(&image, foot.u, foot.v);
                assert!(
                    (image_normal - normal).magnitude() < 1e-9,
                    "{reversed:?} at ({u}, {v})"
                );
            }
        }
    }
}

#[test]
fn test_projection_onto_reversed_surfaces_finds_the_foot() {
    let tol = Tolerance::DEFAULT;
    for surface in surfaces() {
        for reversed in reversals(&surface) {
            for (u, v) in [(0.3, 0.2), (2.0, -0.7)] {
                let foot = reversed.eval_point(u, v);
                let off = foot + 0.1 * unit_normal(&reversed, u, v);
                let projection = reversed.project_point(off, tol);
                let found = reversed.eval_point(projection.u, projection.v);
                assert!(found.distance(foot) < 1e-9, "{reversed:?} at ({u}, {v})");
            }
        }
    }
}
