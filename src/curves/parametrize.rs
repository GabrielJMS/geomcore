//! Analytic curve-on-surface parametrization: computing the exact 2D
//! representation `q(t) = (u(t), v(t))` of a 3D curve in a surface's
//! parameter space, such that `surface.eval_point(q(t)) == curve.eval_point(t)`
//! for the same `t`.
//!
//! Only a handful of curve/surface pairs admit a closed-form 2D image (a line
//! or a circle projects to a straight line or a circle in `(u, v)`); every
//! other pair reports [`ParametrizeError::NotAnalytic`]. The per-pair math and
//! failure conditions are the classic elementary-surface projections; see the
//! public `parametrize_on` methods for the user-facing contract.

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fmt;

use crate::curve_math::analytic::{in_period, wrap_to_turn};
use crate::curves::{Circle2D, Circle3D, Curve2D, Line2D, Line3D};
use crate::surfaces::{ParametricSurface, Surface};
use crate::tol;
use crate::{Frame2, Point2, Point3, Vector2, Vector3};

/// Error returned when a 3D curve cannot be given an analytic 2D
/// representation on a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParametrizeError {
    /// No closed-form 2D representation exists for this curve/surface pair.
    NotAnalytic,
    /// The curve does not lie on the surface (within tolerance).
    CurveNotOnSurface,
}

impl fmt::Display for ParametrizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            ParametrizeError::NotAnalytic => {
                "no closed-form 2D representation exists for this curve/surface pair"
            }
            ParametrizeError::CurveNotOnSurface => "the curve does not lie on the surface",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParametrizeError {}

/// Parameters at which the on-surface consistency check is sampled.
const CHECK_PARAMS: [f64; 3] = [0.0, 1.7, 4.1];

// ---- dispatchers ----

/// Computes the analytic 2D image of `line` on `surface`, or an error.
pub(crate) fn line_on_surface(
    line: &Line3D,
    surface: &Surface,
) -> Result<Curve2D, ParametrizeError> {
    let pcurve = match surface {
        Surface::Plane(plane) => line_on_plane(line, plane.frame()),
        Surface::Cylinder(cyl) => line_on_cylinder(line, cyl.frame()),
        Surface::Cone(cone) => {
            line_on_cone(line, cone.frame(), cone.ref_radius(), cone.semi_angle())
        }
        // A line has no closed-form 2D image on these surfaces.
        Surface::Sphere(_) | Surface::Torus(_) | Surface::BSpline(_) => {
            return Err(ParametrizeError::NotAnalytic);
        }
    }?;

    finish_line(surface, pcurve, |t| line.eval_point(t))
}

/// Computes the analytic 2D image of `circle` on `surface`, or an error.
pub(crate) fn circle_on_surface(
    circle: &Circle3D,
    surface: &Surface,
) -> Result<Curve2D, ParametrizeError> {
    match surface {
        Surface::Plane(plane) => {
            let circle2d = circle_on_plane(circle, plane.frame());
            verify_on_surface(surface, &Curve2D::Circle(circle2d), |t| {
                circle.eval_point(t)
            })?;
            // A circle on a plane maps to a circle; no window normalization.
            Ok(Curve2D::Circle(circle2d))
        }
        Surface::Cylinder(cyl) => {
            let pcurve = circle_on_cylinder(circle, cyl.frame())?;
            finish_line(surface, pcurve, |t| circle.eval_point(t))
        }
        Surface::Cone(cone) => {
            let pcurve =
                circle_on_cone(circle, cone.frame(), cone.ref_radius(), cone.semi_angle())?;
            finish_line(surface, pcurve, |t| circle.eval_point(t))
        }
        Surface::Sphere(sphere) => {
            let pcurve = circle_on_sphere(circle, sphere.frame())?;
            finish_line(surface, pcurve, |t| circle.eval_point(t))
        }
        Surface::Torus(torus) => {
            let pcurve = circle_on_torus(
                circle,
                torus.frame(),
                torus.major_radius(),
                torus.minor_radius(),
            )?;
            finish_line(surface, pcurve, |t| circle.eval_point(t))
        }
        Surface::BSpline(_) => Err(ParametrizeError::NotAnalytic),
    }
}

/// Verifies on-surface consistency, then normalizes the pcurve's value at
/// `t = 0` into the surface's canonical parameter window.
fn finish_line(
    surface: &Surface,
    pcurve: Line2D,
    curve_point: impl Fn(f64) -> Point3,
) -> Result<Curve2D, ParametrizeError> {
    verify_on_surface(surface, &Curve2D::Line(pcurve), curve_point)?;
    Ok(Curve2D::Line(normalize_line2d_at_zero(pcurve, surface)))
}

// ---- on-surface verification ----

/// Checks that `surface(pcurve(t)) ~= curve(t)` at the sample parameters.
///
/// This is a geomrust addition on top of the raw projection: the projection
/// math assumes the curve already lies on the surface, so an off-surface curve
/// would otherwise silently produce a wrong-but-plausible 2D image. The
/// tolerance scales with the characteristic size of the configuration so it
/// stays meaningful for both tiny and large geometries.
fn verify_on_surface(
    surface: &Surface,
    pcurve: &Curve2D,
    curve_point: impl Fn(f64) -> Point3,
) -> Result<(), ParametrizeError> {
    use crate::curves::ParametricCurve2D;

    let mut scale: f64 = 1.0;
    for &t in &CHECK_PARAMS {
        scale = scale.max((curve_point(t) - Point3::ORIGIN).magnitude());
    }
    let tolerance = 1e-6 * scale;

    for &t in &CHECK_PARAMS {
        let uv = pcurve.eval_point(t);
        let on_surface = surface.eval_point(uv.x, uv.y);
        if on_surface.distance(curve_point(t)) > tolerance {
            return Err(ParametrizeError::CurveNotOnSurface);
        }
    }
    Ok(())
}

// ---- plane ----

/// 2D image of a line lying in a plane: local x/y coordinates of the origin
/// and the projected (renormalized) direction.
fn line_on_plane(line: &Line3D, frame: crate::Frame3) -> Result<Line2D, ParametrizeError> {
    let origin = eval_point2d_plane(frame, line.origin());
    let d = line.direction();
    let dir = Vector2::new(d.dot(frame.x_direction()), d.dot(frame.y_direction()));
    // A near-zero projected direction means the line points out of the plane,
    // i.e. it is not on the surface.
    let dir = dir
        .normalized()
        .ok_or(ParametrizeError::CurveNotOnSurface)?;
    Line2D::new(origin, dir).map_err(|_| ParametrizeError::CurveNotOnSurface)
}

/// 2D image of a circle lying in a plane: the circle's frame mapped to local
/// plane coordinates, radius preserved.
fn circle_on_plane(circle: &Circle3D, frame: crate::Frame3) -> Circle2D {
    let center = eval_point2d_plane(frame, circle.center());
    let cf = circle.frame();
    let x_dir = eval_dir2d_plane(frame, cf.x_direction());
    let y_dir = eval_dir2d_plane(frame, cf.y_direction());
    let frame2 = Frame2::new(center, x_dir, y_dir)
        .expect("circle frame axes stay orthonormal under a plane projection");
    Circle2D::from_frame(frame2, circle.radius())
        .expect("circle radius is non-negative by construction")
}

/// Local (x, y) coordinates of `p` in the plane `frame`.
fn eval_point2d_plane(frame: crate::Frame3, p: Point3) -> Point2 {
    let d = p - frame.origin();
    Point2::new(d.dot(frame.x_direction()), d.dot(frame.y_direction()))
}

/// Local (x, y) components of the direction `d` in the plane `frame`.
fn eval_dir2d_plane(frame: crate::Frame3, d: Vector3) -> Vector2 {
    Vector2::new(d.dot(frame.x_direction()), d.dot(frame.y_direction()))
}

// ---- cylinder ----

/// 2D image of a line on a cylinder: succeeds only when the line is parallel
/// to the cylinder axis (a vertical iso-`u` line in `(u, v)`).
fn line_on_cylinder(line: &Line3D, frame: crate::Frame3) -> Result<Line2D, ParametrizeError> {
    let axis = frame.z_direction();
    if line.direction().cross(axis).square_magnitude() > tol::ANGULAR * tol::ANGULAR {
        return Err(ParametrizeError::NotAnalytic);
    }
    let (x, y, z) = frame.local_coordinates(line.origin());
    let u = angle_or_zero(y, x);
    let v = z;
    let sign = if line.direction().dot(axis) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    Line2D::new(Point2::new(u, v), Vector2::new(0.0, sign))
        .map_err(|_| ParametrizeError::NotAnalytic)
}

/// 2D image of a circle on a cylinder: succeeds only when the circle's normal
/// is parallel to the cylinder axis (a horizontal iso-`v` line in `(u, v)`).
fn circle_on_cylinder(circle: &Circle3D, frame: crate::Frame3) -> Result<Line2D, ParametrizeError> {
    let axis = frame.z_direction();
    let normal = circle.normal();
    if axis.cross(normal).square_magnitude() > tol::ANGULAR * tol::ANGULAR {
        return Err(ParametrizeError::NotAnalytic);
    }
    let u = wrap_to_turn(
        frame
            .x_direction()
            .angle_with_ref(circle.frame().x_direction(), axis),
    );
    let v = (circle.center() - frame.origin()).dot(axis);
    let du = if normal.dot(axis) > 0.0 { 1.0 } else { -1.0 };
    Line2D::new(Point2::new(u, v), Vector2::new(du, 0.0)).map_err(|_| ParametrizeError::NotAnalytic)
}

// ---- cone ----

/// 2D image of a line on a cone: succeeds only when the line is a generator
/// (parallel to the `v`-isoline through its own foot point).
fn line_on_cone(
    line: &Line3D,
    frame: crate::Frame3,
    ref_r: f64,
    semi_angle: f64,
) -> Result<Line2D, ParametrizeError> {
    let apex = cone_apex(frame, ref_r, semi_angle);
    let mut foot = line.origin();
    let mut delta_v = 0.0;
    if foot.distance(apex) < tol::CONFUSION {
        // At the apex the v-isoline is degenerate; step along the line to get
        // a well-defined foot point and remember the unit shift to undo.
        foot = foot + line.direction();
        delta_v = 1.0;
    }
    let (u, v) = crate::surface_math::analytic::cone_parameters(&frame, ref_r, semi_angle, foot);
    let sv = crate::surface_math::analytic::cone_derivative(&frame, ref_r, semi_angle, u, v, 0, 1);
    if line.direction().cross(sv).square_magnitude() > tol::ANGULAR * tol::ANGULAR {
        return Err(ParametrizeError::NotAnalytic);
    }
    let sign = if line.direction().dot(sv) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    Line2D::new(Point2::new(u, v - delta_v * sign), Vector2::new(0.0, sign))
        .map_err(|_| ParametrizeError::NotAnalytic)
}

/// 2D image of a circle on a cone: succeeds only when the circle's normal is
/// parallel to the cone axis (a horizontal iso-`v` line in `(u, v)`).
fn circle_on_cone(
    circle: &Circle3D,
    frame: crate::Frame3,
    ref_r: f64,
    semi_angle: f64,
) -> Result<Line2D, ParametrizeError> {
    let axis = frame.z_direction();
    let normal = circle.normal();
    if axis.cross(normal).square_magnitude() > tol::ANGULAR * tol::ANGULAR {
        return Err(ParametrizeError::NotAnalytic);
    }
    let cf = circle.frame();
    let x = frame.x_direction().dot(cf.x_direction());
    let y = frame.y_direction().dot(cf.x_direction());
    let z = (circle.center() - frame.origin()).dot(axis);
    let u = if x.abs() <= tol::ANGULAR && y.abs() <= tol::ANGULAR {
        0.0
    } else if -ref_r > z * semi_angle.tan() {
        // Below the apex the point lies on the opposite nappe, so `u` is
        // measured from the mirrored branch.
        wrap_to_turn((-y).atan2(-x))
    } else {
        wrap_to_turn(y.atan2(x))
    };
    let v = z / semi_angle.cos();
    let du = if normal.dot(axis) > 0.0 { 1.0 } else { -1.0 };
    Line2D::new(Point2::new(u, v), Vector2::new(du, 0.0)).map_err(|_| ParametrizeError::NotAnalytic)
}

/// The cone's apex point in world coordinates: the axis point where the
/// section radius shrinks to zero.
fn cone_apex(frame: crate::Frame3, ref_r: f64, semi_angle: f64) -> Point3 {
    let v_apex = -ref_r / semi_angle.sin();
    frame.origin() + (v_apex * semi_angle.cos()) * frame.z_direction()
}

// ---- sphere ----

/// 2D image of a circle on a sphere: succeeds only for a meridian (great
/// circle through the poles) or a parallel (horizontal circle).
fn circle_on_sphere(circle: &Circle3D, frame: crate::Frame3) -> Result<Line2D, ParametrizeError> {
    let z = frame.z_direction();
    let cf = circle.frame();
    let normal = circle.normal();

    let is_iso_u = normal.dot(z).abs() <= tol::CONFUSION
        && circle.center().distance(frame.origin()) <= tol::CONFUSION;
    let is_iso_v = cf.x_direction().dot(z).abs() <= tol::CONFUSION
        && cf.y_direction().dot(z).abs() <= tol::CONFUSION;

    if is_iso_u {
        Ok(sphere_meridian(circle, frame))
    } else if is_iso_v {
        sphere_parallel(circle, frame)
    } else {
        Err(ParametrizeError::NotAnalytic)
    }
}

/// Meridian (iso-`u`) image: a vertical line in `(u, v)` derived from the 2D
/// images of the circle's x and y axis endpoints, with seam/pole handling.
fn sphere_meridian(circle: &Circle3D, frame: crate::Frame3) -> Line2D {
    let cf = circle.frame();
    let mut p1 = eval_point2d_sphere(frame, cf.x_direction());
    let mut p2 = eval_point2d_sphere(frame, cf.y_direction());

    if (p1.y - FRAC_PI_2).abs() < tol::P_CONFUSION || (p1.y + FRAC_PI_2).abs() < tol::P_CONFUSION {
        // p1 sits on a pole where u is undefined; borrow p2's u.
        p1.x = p2.x;
    } else if ((p1.x - p2.x).abs() - PI).abs() < tol::P_CONFUSION {
        // The two endpoints straddle the seam; fold p2 onto p1's meridian.
        p2.x = p1.x;
        p2.y = if p2.y < 0.0 { -PI - p2.y } else { PI - p2.y };
    } else {
        p2.x = p1.x;
    }

    let dir = (p2 - p1)
        .normalized()
        .expect("meridian endpoints are distinct in v");
    Line2D::new(p1, dir).expect("meridian direction is unit length")
}

/// Parallel (iso-`v`) image: a horizontal line in `(u, v)` at constant
/// latitude, running in `+u` or `-u` by the circle's orientation.
fn sphere_parallel(circle: &Circle3D, frame: crate::Frame3) -> Result<Line2D, ParametrizeError> {
    let z = frame.z_direction();
    let cf = circle.frame();
    let u = wrap_to_turn(frame.x_direction().angle_with_ref(cf.x_direction(), z));
    // Recover the sphere radius from an actual on-surface point (the circle's
    // own radius is the parallel's radius, not the sphere's).
    let r = circle.eval_point(0.0).distance(frame.origin());
    let height = (circle.center() - frame.origin()).dot(z);
    let v = (height / r).clamp(-1.0, 1.0).asin();
    let du = if circle.normal().dot(z) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    Line2D::new(Point2::new(u, v), Vector2::new(du, 0.0)).map_err(|_| ParametrizeError::NotAnalytic)
}

/// Local sphere `(u, v)` of a unit direction `dir` seen from the sphere frame.
fn eval_point2d_sphere(frame: crate::Frame3, dir: Vector3) -> Point2 {
    let x = dir.dot(frame.x_direction());
    let y = dir.dot(frame.y_direction());
    let z = dir.dot(frame.z_direction());
    let u = angle_or_zero(y, x);
    let v = z.clamp(-1.0, 1.0).asin();
    Point2::new(u, v)
}

/// Canonicalizes a sphere pcurve's value at `t = 0` into the parameter window,
/// folding poles and the `v`/`u` seams (mirror about `v = ±π/2` with a `+π`
/// `u` shift when the reference point spills past a pole).
fn sphere_set_in_bounds(line: Line2D) -> Line2D {
    let mut line = line;

    // 1. Bring v(0) into [-π, π].
    let y0 = line.eval_point(0.0).y;
    let new_y = in_period(y0, -PI, PI);
    line = translate_line(line, 0.0, new_y - y0);

    // 2. Pole mirroring if v(0) still overshoots ±π/2.
    let p = line.eval_point(0.0);
    let dir = line.direction();
    let north = p.y - FRAC_PI_2 > tol::CONFUSION
        || ((p.y - FRAC_PI_2).abs() < tol::CONFUSION && dir.y > 0.0);
    let south = p.y + FRAC_PI_2 < -tol::CONFUSION
        || ((p.y + FRAC_PI_2).abs() < tol::CONFUSION && dir.y < 0.0);
    if north {
        line = mirror_about_horizontal(line, FRAC_PI_2);
        line = translate_line(line, PI, 0.0);
    } else if south {
        line = mirror_about_horizontal(line, -FRAC_PI_2);
        line = translate_line(line, PI, 0.0);
    } else {
        return line;
    }

    // 3. Bring u(0) into [0, 2π).
    let x0 = line.eval_point(0.0).x;
    let new_x = in_period(x0, 0.0, TAU);
    translate_line(line, new_x - x0, 0.0)
}

/// Reflects a 2D line about the horizontal axis `v = c`: `(u, v) -> (u, 2c-v)`,
/// flipping the direction's `v` component.
fn mirror_about_horizontal(line: Line2D, c: f64) -> Line2D {
    let o = line.origin();
    let d = line.direction();
    Line2D::new(Point2::new(o.x, 2.0 * c - o.y), Vector2::new(d.x, -d.y))
        .expect("mirrored direction keeps unit length")
}

// ---- torus ----

/// 2D image of a circle on a torus: succeeds for a toroidal circle (normal
/// parallel to the torus axis, running around the ring in `u` at fixed `v`)
/// or a poloidal circle (a meridian around the tube, running in `v` at fixed
/// `u`).
fn circle_on_torus(
    circle: &Circle3D,
    frame: crate::Frame3,
    maj: f64,
    min: f64,
) -> Result<Line2D, ParametrizeError> {
    let z = frame.z_direction();
    let oc = circle.center() - frame.origin();
    let normal = circle.normal();

    let is_toroidal = oc.magnitude() < tol::CONFUSION
        || z.cross(normal).square_magnitude() <= tol::ANGULAR * tol::ANGULAR;

    if is_toroidal {
        torus_toroidal(circle, frame, maj, min)
    } else {
        torus_poloidal(circle, frame)
    }
}

/// Toroidal image (the circle runs around the ring at fixed poloidal angle
/// `v`): a horizontal line in `(u, v)`. The `v` latitude is `asin` of the
/// center's axial offset over the minor radius, on the near or far tube half
/// depending on whether the circle radius is below the major radius.
fn torus_toroidal(
    circle: &Circle3D,
    frame: crate::Frame3,
    maj: f64,
    min: f64,
) -> Result<Line2D, ParametrizeError> {
    let z = frame.z_direction();
    let cf = circle.frame();
    let mut p1 = torus_eval_u(frame, cf.x_direction());
    let mut p2 = torus_eval_u(frame, cf.y_direction());

    let oc = circle.center() - frame.origin();
    let axial_fraction = oc.dot(z) / min;
    let mut v = axial_fraction.clamp(-1.0, 1.0).asin();
    if circle.radius() < maj {
        v = PI - v;
    } else if v < 0.0 {
        v += TAU;
    }
    p1.y = v;
    p2.y = v;

    let mut v2d = p2 - p1;
    if (p1.x - p2.x).abs() > PI {
        v2d = -v2d;
    }
    let dir = v2d.normalized().ok_or(ParametrizeError::NotAnalytic)?;
    if p1.x < 0.0 {
        p1.x += TAU;
    }
    Line2D::new(p1, dir).map_err(|_| ParametrizeError::NotAnalytic)
}

/// Poloidal image (meridian circle at fixed toroidal angle `u`): a vertical
/// line in `(u, v)`.
fn torus_poloidal(circle: &Circle3D, frame: crate::Frame3) -> Result<Line2D, ParametrizeError> {
    let z = frame.z_direction();
    let oc = circle.center() - frame.origin();
    let cf = circle.frame();
    let u = wrap_to_turn(frame.x_direction().angle_with_ref(oc, z));
    let mut v1 = oc.angle_with_ref(cf.x_direction(), oc.cross(z));
    if v1 < 0.0 {
        v1 += TAU;
    }
    let mut dir = Vector2::Y;
    if oc.cross(z).dot(cf.x_direction().cross(cf.y_direction())) < 0.0 {
        dir = -dir;
    }
    Line2D::new(Point2::new(u, v1), dir).map_err(|_| ParametrizeError::NotAnalytic)
}

/// Toroidal `u` of a unit direction: the angle of its projection into the
/// torus's equatorial plane, with the `v` slot left at zero for the caller.
fn torus_eval_u(frame: crate::Frame3, dir: Vector3) -> Point2 {
    let x = dir.dot(frame.x_direction());
    let y = dir.dot(frame.y_direction());
    Point2::new(angle_or_zero(y, x), 0.0)
}

// ---- normalization ----

/// Shifts `line`'s value at `t = 0` into `surface`'s canonical parameter
/// window: `u` (and `v` for the torus) wrapped by period multiples; the sphere
/// additionally folds poles/seams via [`sphere_set_in_bounds`].
pub(crate) fn normalize_line2d_at_zero(line: Line2D, surface: &Surface) -> Line2D {
    match surface {
        // A plane has no periodic direction: leave the line as-is.
        Surface::Plane(_) | Surface::BSpline(_) => line,
        Surface::Cylinder(_) | Surface::Cone(_) => wrap_u(line),
        Surface::Sphere(_) => sphere_set_in_bounds(line),
        Surface::Torus(_) => wrap_v(wrap_u(line)),
    }
}

/// Wraps `line`'s `u` at `t = 0` into `[0, 2π)`.
fn wrap_u(line: Line2D) -> Line2D {
    let u0 = line.eval_point(0.0).x;
    let new_u = in_period(u0, 0.0, TAU);
    translate_line(line, new_u - u0, 0.0)
}

/// Wraps `line`'s `v` at `t = 0` into `[0, 2π)`.
fn wrap_v(line: Line2D) -> Line2D {
    let v0 = line.eval_point(0.0).y;
    let new_v = in_period(v0, 0.0, TAU);
    translate_line(line, 0.0, new_v - v0)
}

/// Translates a 2D line's origin by `(du, dv)`, keeping its direction.
fn translate_line(line: Line2D, du: f64, dv: f64) -> Line2D {
    let o = line.origin();
    Line2D::new(Point2::new(o.x + du, o.y + dv), line.direction())
        .expect("translation preserves the unit direction")
}

// ---- small helpers ----

/// `atan2(y, x)` wrapped into `[0, 2π)`, returning `0` when both inputs are
/// below the parametric-space tolerance (the angle is then undefined).
fn angle_or_zero(y: f64, x: f64) -> f64 {
    if x.abs() < tol::P_CONFUSION && y.abs() < tol::P_CONFUSION {
        0.0
    } else {
        wrap_to_turn(y.atan2(x))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FRAC_PI_2, PI, TAU, mirror_about_horizontal, sphere_meridian, sphere_set_in_bounds,
    };
    use crate::curve_math::analytic::in_period;
    use crate::curves::{Curve2D, ParametricCurve2D, ParametrizeError};
    use crate::tol;
    use crate::{
        Circle3D, Cylinder, Frame3, Line2D, Line3D, Plane, Point2, Point3, Sphere, Vector2, Vector3,
    };

    #[test]
    fn test_circle_on_cylinder_radius_mismatch_is_not_on_surface() {
        // A radius-1 circle cannot lie on a radius-2 cylinder even though its
        // axis is coaxial: the projection would be geometrically consistent
        // (normal parallel to axis) but off-surface.
        let cylinder = Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        let circle = Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vector3::Z, 1.0).unwrap();
        assert_eq!(
            circle.parametrize_on(cylinder),
            Err(ParametrizeError::CurveNotOnSurface)
        );
    }

    #[test]
    fn test_line_skew_to_cylinder_axis_is_not_analytic() {
        // A line not parallel to the axis has no straight-line 2D image: the
        // geometric failure condition fires first, before any on-surface test.
        let cylinder = Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        let line = Line3D::new(Point3::new(2.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 1.0)).unwrap();
        assert_eq!(
            line.parametrize_on(cylinder),
            Err(ParametrizeError::NotAnalytic)
        );
    }

    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)]
    fn test_parametrize_on_accepts_surface_by_reference() {
        // The public method takes `impl Into<Surface>`, which `&Cylinder`
        // satisfies via `From<&Cylinder>` (Cylinder is Copy, so clippy would
        // rather drop the borrow; pinning the borrowed form is the point).
        let cylinder = Cylinder::new(Point3::ORIGIN, Vector3::Z, 2.0).unwrap();
        let circle = Circle3D::new(Point3::new(0.0, 0.0, 3.0), Vector3::Z, 2.0).unwrap();
        assert!(circle.parametrize_on(&cylinder).is_ok());
    }

    #[test]
    fn test_line_on_plane_perpendicular_is_not_on_surface() {
        // A line pointing out of the plane projects to a zero-length 2D
        // direction, which is reported as off-surface rather than a garbage
        // parametrization.
        let plane = Plane::new(Point3::ORIGIN, Vector3::Z).unwrap();
        let line = Line3D::new(Point3::ORIGIN, Vector3::Z).unwrap();
        assert_eq!(
            line.parametrize_on(plane),
            Err(ParametrizeError::CurveNotOnSurface)
        );
    }

    #[test]
    fn test_parametrize_error_display_and_is_std_error() {
        fn takes_error(_e: &dyn std::error::Error) {}
        takes_error(&ParametrizeError::NotAnalytic);
        assert!(!ParametrizeError::NotAnalytic.to_string().is_empty());
        assert!(!ParametrizeError::CurveNotOnSurface.to_string().is_empty());
    }

    // ---- sphere pole-mirror branch (sphere_set_in_bounds lines 397-410) ----
    //
    // The branch mirrors a meridian pcurve about v = +-pi/2 when q(0) sits
    // at a pole boundary AND the line's v-direction continues past it. The
    // reviewer's suggested construction (circle-frame x_direction() pointing
    // exactly at a pole) does NOT trigger it: `sphere_meridian` derives the
    // pcurve's v-direction from `p2 - p1`, where `p2` comes from the
    // circle's y_direction() — orthogonal to x_direction(), and therefore
    // always closer to the equator. That forces `dir.y` to point *away* from
    // whichever pole `p1` sits at, the opposite of what the mirror condition
    // needs. Exhaustive random search over orthonormal circle frames
    // confirms this construction never triggers the branch.
    //
    // The actual trigger is the *seam-fold* sub-case (line 336): tilt the
    // circle's x_direction() a hair off the pole (a few 1e-7 rad) so
    // `sphere_meridian`'s own pole snap-to (`tol::P_CONFUSION = 1e-9`)
    // does *not* fire, but the endpoints still straddle the u-seam
    // (`|p1.x - p2.x| ~ pi`), which folds `p2` across the pole and makes the
    // line's v-direction continue *past* it. The resulting `p1.y` then lands
    // just inside `sphere_set_in_bounds`'s looser boundary tolerance
    // (`tol::CONFUSION = 1e-7`) with the matching direction sign, which is
    // exactly the condition on lines 393-396.
    #[test]
    fn test_sphere_meridian_pole_mirror_north() {
        let sphere = Sphere::new(Point3::ORIGIN, 2.0).unwrap();

        // circle normal = +X (perpendicular to the sphere's z-axis, so this
        // is a meridian); x_hint tilts x_direction() a hair off the north
        // pole and just past the u-seam relative to y_direction(), which is
        // what makes sphere_meridian fold p2 across the pole (the seam-fold
        // sub-case) instead of snapping p1 to it (the pole sub-case).
        let eps = 2e-7_f64;
        let phi = 15.0_f64.to_radians();
        let x_hint = Vector3::new(phi.cos() * eps.sin(), phi.sin() * eps.sin(), eps.cos());
        let frame = Frame3::new(Point3::ORIGIN, Vector3::X, x_hint).unwrap();
        let circle = Circle3D::from_frame(frame, 2.0).unwrap();

        let sf = sphere.frame();
        let raw = sphere_meridian(&circle, sf);
        // q(0) sits just short of the north pole, with the v-direction
        // continuing past it — the trigger condition on line 394.
        assert!(FRAC_PI_2 - raw.origin().y > 0.0);
        assert!(FRAC_PI_2 - raw.origin().y < tol::CONFUSION);
        assert_close(raw.direction().y, 1.0);

        let normalized = sphere_set_in_bounds(raw);
        // Observable proof the mirror branch ran: v is reflected about
        // pi/2 (staying within tol::CONFUSION of the pole, on the other
        // side of it from the raw line) and u picks up the +pi shift
        // (wrapped into [0, 2pi)).
        assert_close(
            normalized.origin().x,
            in_period(raw.origin().x + PI, 0.0, TAU),
        );
        assert_close(normalized.direction().y, -raw.direction().y);
        assert!((normalized.origin().y - FRAC_PI_2).abs() < tol::CONFUSION);
        assert!((0.0..TAU + 1e-9).contains(&normalized.origin().x));

        // The consistency invariant is the real proof the mirror math is
        // right: it must hold across t values that cross the pole.
        let pcurve = circle.parametrize_on(sphere).unwrap();
        assert_eq!(pcurve, Curve2D::Line(normalized));
        for t in [0.0_f64, 0.5, 1.7, 3.0, 4.5] {
            let uv = pcurve.eval_point(t);
            let on_surface = sphere.eval_point(uv.x, uv.y);
            let expected = circle.eval_point(t);
            assert!(
                on_surface.distance(expected) < 1e-9,
                "t={t}: sphere.eval_point(u,v)={on_surface:?} != circle.eval_point(t)={expected:?}"
            );
        }
    }

    #[test]
    fn test_sphere_meridian_pole_mirror_south() {
        let sphere = Sphere::new(Point3::ORIGIN, 2.0).unwrap();

        // South twin: negating the whole north x_hint reflects the
        // construction through the sphere's center, landing x_direction()
        // just short of the south pole with the matching seam-fold geometry.
        let eps = 2e-7_f64;
        let phi = 15.0_f64.to_radians();
        let x_hint = -Vector3::new(phi.cos() * eps.sin(), phi.sin() * eps.sin(), eps.cos());
        let frame = Frame3::new(Point3::ORIGIN, Vector3::X, x_hint).unwrap();
        let circle = Circle3D::from_frame(frame, 2.0).unwrap();

        let sf = sphere.frame();
        let raw = sphere_meridian(&circle, sf);
        assert!(raw.origin().y - (-FRAC_PI_2) > 0.0);
        assert!(raw.origin().y - (-FRAC_PI_2) < tol::CONFUSION);
        assert_close(raw.direction().y, -1.0);

        let normalized = sphere_set_in_bounds(raw);
        // Observable proof the mirror branch ran: v is reflected about
        // -pi/2 and u picks up the +pi shift (wrapped into [0, 2pi)).
        assert_close(
            normalized.origin().x,
            in_period(raw.origin().x + PI, 0.0, TAU),
        );
        assert_close(normalized.direction().y, -raw.direction().y);
        assert!((normalized.origin().y - (-FRAC_PI_2)).abs() < tol::CONFUSION);
        assert!((0.0..TAU + 1e-9).contains(&normalized.origin().x));

        let pcurve = circle.parametrize_on(sphere).unwrap();
        assert_eq!(pcurve, Curve2D::Line(normalized));
        for t in [0.0_f64, 0.5, 1.7, 3.0, 4.5] {
            let uv = pcurve.eval_point(t);
            let on_surface = sphere.eval_point(uv.x, uv.y);
            let expected = circle.eval_point(t);
            assert!(
                on_surface.distance(expected) < 1e-9,
                "t={t}: sphere.eval_point(u,v)={on_surface:?} != circle.eval_point(t)={expected:?}"
            );
        }
    }

    /// Direct unit test of the mirror helper: `mirror_about_horizontal`
    /// reflects `(u, v)` about the horizontal line `v = c`, i.e.
    /// `(u, v) -> (u, 2c - v)`, and flips the sign of the direction's `v`
    /// component. The caller (`sphere_set_in_bounds`) applies the separate
    /// `u += π` shift itself; that translation does not live inside this
    /// helper.
    #[test]
    fn test_mirror_about_horizontal() {
        let line = Line2D::new(Point2::new(1.0, 1.7), Vector2::new(0.0, 1.0)).unwrap();
        let mirrored = mirror_about_horizontal(line, FRAC_PI_2);
        assert_close(mirrored.origin().x, 1.0);
        assert_close(mirrored.origin().y, 2.0 * FRAC_PI_2 - 1.7);
        assert_close(mirrored.direction().x, 0.0);
        assert_close(mirrored.direction().y, -1.0);
    }

    #[test]
    fn test_mirror_about_horizontal_is_involution() {
        // Mirroring twice about the same axis returns the original line.
        let line = Line2D::new(Point2::new(-2.3, 0.4), Vector2::new(0.0, -1.0)).unwrap();
        let once = mirror_about_horizontal(line, FRAC_PI_2);
        let twice = mirror_about_horizontal(once, FRAC_PI_2);
        assert_close(twice.origin().x, line.origin().x);
        assert_close(twice.origin().y, line.origin().y);
        assert_close(twice.direction().x, line.direction().x);
        assert_close(twice.direction().y, line.direction().y);
    }

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }
}
