//! Isoparametric curves of surfaces: shared error type and helpers.
//!
//! `u_iso(u)` fixes `u` and runs along `v`; `v_iso(v)` fixes `v` and runs
//! along `u`. Either is parametrized like the surface, so the iso curve
//! at `t` is the surface point at `(u, t)` or `(t, v)`.

use std::fmt;

use crate::curve_math::bspline::KnotVector;
use crate::{Circle3D, Frame3D, Line3D, Point3D, Vector3D};

/// Error returned when an isoparametric curve cannot be built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceIsoError {
    /// The fixed parameter is NaN or infinite.
    NonFinite,
    /// The fixed parameter lies outside the bounds of a non-periodic
    /// B-spline direction.
    OutOfRange,
}

impl fmt::Display for SurfaceIsoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            SurfaceIsoError::NonFinite => "the fixed parameter is NaN or infinite",
            SurfaceIsoError::OutOfRange => {
                "the fixed parameter is outside the bounds of a non-periodic direction"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for SurfaceIsoError {}

/// The fixed parameter, validated: finite everywhere, wrapped into the
/// period on a periodic B-spline direction (`None`: elementary, where
/// any finite value is accepted) and range-checked on a non-periodic
/// one. Exactly `last` closes the last span instead of wrapping, as in
/// evaluation.
pub(crate) fn check_iso_param(t: f64, knots: Option<&KnotVector>) -> Result<f64, SurfaceIsoError> {
    if !t.is_finite() {
        return Err(SurfaceIsoError::NonFinite);
    }
    let Some(knots) = knots else {
        return Ok(t);
    };
    let (first, last) = knots.bounds();
    if knots.is_periodic() {
        Ok(if t == last {
            last
        } else {
            crate::curve_math::analytic::in_period(t, first, last)
        })
    } else if t < first || t > last {
        Err(SurfaceIsoError::OutOfRange)
    } else {
        Ok(t)
    }
}

/// The line `origin + t * direction`, `direction` being a unit vector. A
/// finite parameter can still carry `origin` past `f64` range.
pub(crate) fn iso_line(origin: Point3D, direction: Vector3D) -> Result<Line3D, SurfaceIsoError> {
    Line3D::new(origin, direction).map_err(|_| SurfaceIsoError::NonFinite)
}

/// The circle `center + radius * (cos(t) * x + sin(t) * y)`, `y` being
/// `normal x x`. A negative `radius` reverses `x` and `y` instead, so the
/// parametrization holds through an apex or a pole, where the radius
/// changes sign, and a zero radius gives the degenerate circle there.
pub(crate) fn iso_circle(
    center: Point3D,
    normal: Vector3D,
    x: Vector3D,
    radius: f64,
) -> Result<Circle3D, SurfaceIsoError> {
    let (x, radius) = if radius < 0.0 {
        (-x, -radius)
    } else {
        (x, radius)
    };
    let frame = Frame3D::new(center, normal, x).map_err(|_| SurfaceIsoError::NonFinite)?;
    Circle3D::from_frame(frame, radius).map_err(|_| SurfaceIsoError::NonFinite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::ParametricCurve3D;
    use crate::surfaces::ParametricSurface;
    use crate::{BSplineSurface, Cone, Curve3D, Cylinder, Plane, Sphere, Surface, Torus};
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    const PARAMS: [f64; 9] = [-7.5, -PI, -1.2, -0.3, 0.0, 0.4, 1.7, PI, 9.0];

    fn tilted_frame() -> Frame3D {
        Frame3D::new(
            Point3D::new(1.0, -2.0, 0.5),
            Vector3D::new(0.3, -0.4, 1.0),
            Vector3D::new(1.0, 0.2, 0.0),
        )
        .unwrap()
    }

    /// Every `u_iso(u)` and `v_iso(v)` over `us` and `vs` reproduces the
    /// surface along `ts`, within `1e-12` of the surface's extent.
    fn assert_isos_match(surface: &Surface, us: &[f64], vs: &[f64], ts: &[f64]) {
        let scale = ts
            .iter()
            .flat_map(|&t| {
                us.iter()
                    .map(move |&u| surface.eval_point(u, t))
                    .chain(vs.iter().map(move |&v| surface.eval_point(t, v)))
            })
            .map(|p| (p - Point3D::ORIGIN).magnitude())
            .fold(1.0, f64::max);
        for &u in us {
            let iso = surface.u_iso(u).unwrap();
            for &t in ts {
                let d = iso.eval_point(t).distance(surface.eval_point(u, t));
                assert!(d <= 1e-12 * scale, "u_iso({u}) at {t}: off by {d}");
            }
        }
        for &v in vs {
            let iso = surface.v_iso(v).unwrap();
            for &t in ts {
                let d = iso.eval_point(t).distance(surface.eval_point(t, v));
                assert!(d <= 1e-12 * scale, "v_iso({v}) at {t}: off by {d}");
            }
        }
    }

    #[test]
    fn elementary_isos_reproduce_the_surface() {
        let frame = tilted_frame();
        let surfaces = [
            Surface::from(Plane::from_frame(frame)),
            Surface::from(Cylinder::from_frame(frame, 2.5).unwrap()),
            Surface::from(Cone::from_frame(frame, 0.6, 1.5).unwrap()),
            Surface::from(Cone::from_frame(frame, -0.6, 1.5).unwrap()),
            Surface::from(Sphere::from_frame(frame, 3.0).unwrap()),
            Surface::from(Torus::from_frame(frame, 5.0, 1.5).unwrap()),
        ];
        for surface in &surfaces {
            assert_isos_match(surface, &PARAMS, &PARAMS, &PARAMS);
        }
    }

    #[test]
    fn elementary_isos_hold_where_the_radius_changes_sign() {
        let frame = tilted_frame();
        // Past the cone's apex, beyond the sphere's poles, and across the
        // inner pinch of a spindle torus.
        let cone = Surface::from(Cone::from_frame(frame, 0.5, 1.0).unwrap());
        assert_isos_match(&cone, &[], &[-6.0, -3.0, -2.5], &PARAMS);
        let sphere = Surface::from(Sphere::from_frame(frame, 2.0).unwrap());
        assert_isos_match(&sphere, &[], &[2.0, PI, -2.5], &PARAMS);
        let spindle = Surface::from(Torus::from_frame(frame, 1.0, 2.0).unwrap());
        assert_isos_match(&spindle, &PARAMS, &[2.5, PI, -2.2], &PARAMS);
    }

    #[test]
    fn degenerate_isos_are_circles_of_radius_zero() {
        let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.5, 1.0).unwrap();
        let apex = -1.0 / 0.5f64.sin();
        let at_apex = cone.v_iso(apex).unwrap();
        assert!(at_apex.radius() < 1e-12);
        assert!(at_apex.center().distance(cone.apex()) < 1e-12);

        let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
        for pole in [FRAC_PI_2, -FRAC_PI_2] {
            let iso = sphere.v_iso(pole).unwrap();
            assert!(iso.radius() < 1e-12);
            assert!(iso.center().distance(sphere.eval_point(0.0, pole)) < 1e-12);
        }

        let horn = Torus::new(Point3D::ORIGIN, Vector3D::Z, 2.0, 2.0).unwrap();
        let pinch = horn.v_iso(PI).unwrap();
        assert!(pinch.radius() < 1e-12);
        assert!(pinch.center().distance(Point3D::ORIGIN) < 1e-12);
    }

    #[test]
    fn elementary_isos_have_the_natural_type() {
        let frame = tilted_frame();
        let cases = [
            (Surface::from(Plane::from_frame(frame)), "line", "line"),
            (
                Surface::from(Cylinder::from_frame(frame, 1.0).unwrap()),
                "line",
                "circle",
            ),
            (
                Surface::from(Cone::from_frame(frame, 0.3, 1.0).unwrap()),
                "line",
                "circle",
            ),
            (
                Surface::from(Sphere::from_frame(frame, 1.0).unwrap()),
                "circle",
                "circle",
            ),
            (
                Surface::from(Torus::from_frame(frame, 3.0, 1.0).unwrap()),
                "circle",
                "circle",
            ),
        ];
        let kind = |curve: Curve3D| match curve {
            Curve3D::Line(_) => "line",
            Curve3D::Circle(_) => "circle",
            _ => "other",
        };
        for (surface, u_kind, v_kind) in cases {
            assert_eq!(kind(surface.u_iso(0.7).unwrap()), u_kind);
            assert_eq!(kind(surface.v_iso(0.7).unwrap()), v_kind);
        }
    }

    #[test]
    fn non_finite_parameters_are_rejected() {
        let frame = tilted_frame();
        let surfaces = [
            Surface::from(Plane::from_frame(frame)),
            Surface::from(Cylinder::from_frame(frame, 1.0).unwrap()),
            Surface::from(Cone::from_frame(frame, 0.3, 1.0).unwrap()),
            Surface::from(Sphere::from_frame(frame, 1.0).unwrap()),
            Surface::from(Torus::from_frame(frame, 3.0, 1.0).unwrap()),
            Surface::from(clamped_patch()),
        ];
        for surface in &surfaces {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(surface.u_iso(bad).unwrap_err(), SurfaceIsoError::NonFinite);
                assert_eq!(surface.v_iso(bad).unwrap_err(), SurfaceIsoError::NonFinite);
            }
        }
    }

    #[test]
    fn a_line_carried_past_f64_range_is_non_finite() {
        let plane = Plane::new(Point3D::new(1e308, 0.0, 0.0), Vector3D::Z).unwrap();
        assert_eq!(
            plane.u_iso(f64::MAX).unwrap_err(),
            SurfaceIsoError::NonFinite
        );
    }

    /// A clamped polynomial patch, degree 3 in `u` and 2 in `v`, with
    /// interior knots in both directions.
    fn clamped_patch() -> BSplineSurface {
        let poles = (0..5)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        let (x, y) = (i as f64, j as f64);
                        Point3D::new(x, y + 0.3 * x, (x * 0.7).sin() * (y - 1.5))
                    })
                    .collect()
            })
            .collect();
        BSplineSurface::new(
            3,
            2,
            poles,
            vec![0.0, 0.4, 1.0],
            vec![4, 1, 4],
            vec![-1.0, 0.5, 2.0],
            vec![3, 1, 3],
            false,
            false,
        )
        .unwrap()
    }

    /// The patch periodic in `u` (degree 2, six rows around a ring) and
    /// rational in `v`.
    fn periodic_rational_tube() -> BSplineSurface {
        let poles = (0..6)
            .map(|i| {
                let a = i as f64 * TAU / 6.0;
                (0..3)
                    .map(|j| {
                        let r = 2.0 + 0.4 * j as f64;
                        Point3D::new(r * a.cos(), r * a.sin(), 1.5 * j as f64)
                    })
                    .collect()
            })
            .collect();
        let weights = (0..6)
            .map(|i| {
                (0..3)
                    .map(|j| 1.0 + 0.1 * i as f64 + 0.5 * j as f64)
                    .collect()
            })
            .collect();
        BSplineSurface::new_rational(
            2,
            2,
            poles,
            weights,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1; 7],
            vec![0.0, 1.0],
            vec![3, 3],
            true,
            false,
        )
        .unwrap()
    }

    fn samples((first, last): (f64, f64)) -> Vec<f64> {
        (0..=12)
            .map(|k| first + (last - first) * k as f64 / 12.0)
            .collect()
    }

    #[test]
    fn bspline_isos_reproduce_the_surface() {
        let patch = clamped_patch();
        let (us, vs) = (samples((0.0, 1.0)), samples((-1.0, 2.0)));
        let surface = Surface::from(patch.clone());
        assert_isos_match(&surface, &us, &vs, &us);
        assert_isos_match(&surface, &us, &vs, &vs);

        let iso = patch.u_iso(0.4).unwrap();
        assert_eq!(iso.degree(), 2);
        assert_eq!(iso.knots(), patch.v_knots());
        assert_eq!(iso.multiplicities(), patch.v_multiplicities());
        assert!(!iso.is_rational());
        let iso = patch.v_iso(0.5).unwrap();
        assert_eq!(iso.degree(), 3);
        assert_eq!(iso.knots(), patch.u_knots());
    }

    #[test]
    fn periodic_rational_isos_reproduce_the_surface_and_wrap() {
        let tube = periodic_rational_tube();
        let surface = Surface::from(tube.clone());
        let us: Vec<f64> = samples((-6.0, 12.0));
        let vs = samples((0.0, 1.0));
        assert_isos_match(&surface, &us, &vs, &vs);
        assert_isos_match(&surface, &us, &vs, &samples((0.0, 6.0)));

        let ring = tube.v_iso(0.3).unwrap();
        assert!(ring.is_periodic());
        assert!(ring.is_rational());
        assert_eq!(tube.u_iso(1.5).unwrap(), tube.u_iso(7.5).unwrap());
        let (start, end) = (tube.u_iso(0.0).unwrap(), tube.u_iso(6.0).unwrap());
        for t in vs {
            assert!(start.eval_point(t).distance(end.eval_point(t)) < 1e-12);
        }
    }

    #[test]
    fn a_surface_rational_in_one_direction_gives_polynomial_isos_across_it() {
        // Weights vary along `u` only: every column carries the same ones,
        // so each `u` iso blends them into one common weight.
        let poles: Vec<Vec<Point3D>> = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| Point3D::new(i as f64, j as f64, (i * j) as f64))
                    .collect()
            })
            .collect();
        let weights = vec![vec![1.0; 3], vec![2.0; 3], vec![1.0; 3]];
        let surface = BSplineSurface::new_rational(
            2,
            2,
            poles,
            weights,
            vec![0.0, 1.0],
            vec![3, 3],
            vec![0.0, 1.0],
            vec![3, 3],
            false,
            false,
        )
        .unwrap();
        let along_v = surface.u_iso(0.3).unwrap();
        assert!(!along_v.is_rational());
        assert!(surface.v_iso(0.3).unwrap().is_rational());
        assert_isos_match(
            &Surface::from(surface),
            &[0.3],
            &[0.3],
            &samples((0.0, 1.0)),
        );
    }

    #[test]
    fn a_collapsed_bspline_edge_gives_a_point_curve() {
        let sphere = Sphere::from_frame(tilted_frame(), 2.0)
            .unwrap()
            .to_bspline();
        let bounds = |knots: &[f64]| (knots[0], knots[knots.len() - 1]);
        let (u_bounds, v_bounds) = (bounds(sphere.u_knots()), bounds(sphere.v_knots()));
        for v in [v_bounds.0, v_bounds.1] {
            let iso = sphere.v_iso(v).unwrap();
            let pole = iso.poles()[0];
            assert!(iso.poles().iter().all(|p| p.distance(pole) < 1e-12));
            assert!(pole.distance(sphere.eval_point(u_bounds.0, v)) < 1e-12);
        }
        let (us, vs) = (samples(u_bounds), samples(v_bounds));
        let surface = Surface::from(sphere);
        assert_isos_match(&surface, &us, &vs, &us);
        assert_isos_match(&surface, &us, &vs, &vs);
    }

    #[test]
    fn bspline_parameters_outside_a_clamped_direction_are_rejected() {
        let patch = clamped_patch();
        assert!(patch.u_iso(0.0).is_ok());
        assert!(patch.u_iso(1.0).is_ok());
        assert!(patch.v_iso(2.0).is_ok());
        assert_eq!(patch.u_iso(-1e-9).unwrap_err(), SurfaceIsoError::OutOfRange);
        assert_eq!(patch.u_iso(1.5).unwrap_err(), SurfaceIsoError::OutOfRange);
        assert_eq!(patch.v_iso(-1.5).unwrap_err(), SurfaceIsoError::OutOfRange);
        assert_eq!(
            periodic_rational_tube().v_iso(1.5).unwrap_err(),
            SurfaceIsoError::OutOfRange
        );
    }
}
