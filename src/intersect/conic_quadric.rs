//! Exact intersections of a conic with a quadric.
//!
//! Every pair is solved the same way. The quadric's section by the
//! conic's plane is an [`ImplicitConic`] in the conic's frame coordinates
//! `(α, β)`; substituting the conic's parametrization into it gives one
//! equation in the curve parameter:
//!
//! - circle or ellipse, `(a cos t, b sin t)`: a double-angle trigonometric
//!   equation, first order when the section has no second-degree terms
//!   left (a plane, or a sphere cut by a circle);
//! - parabola, `(t²/4f, t)`: a quartic in `t`, quadratic for a plane;
//! - hyperbola, `(a cosh t, b sinh t)`: a quartic in `u = eᵗ`, quadratic
//!   for a plane.
//!
//! Each quadric writes its section once and each conic its substitution
//! once, instead of one derivation per pair. The same module decides when
//! the whole conic lies on the quadric and keeps only hits verified on it.

use super::{ConicSurfaceHit, ConicSurfaceIntersection, ImplicitConic};
use crate::curves::ParametricCurve3D;
use crate::math::{
    QuadraticSolution, roots_with_tangency, solve_quadratic, solve_trig, solve_trig_linear,
};
use crate::surfaces::ParametricSurface;
use crate::{
    Circle3D, Cone, Cylinder, Ellipse3D, Frame3D, Hyperbola3D, Parabola3D, Plane, Point3D, Sphere,
    Tolerance, Vector3D,
};

/// The shape of a conic in its own frame: what its parametrization
/// substitutes into a section.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ConicShape {
    /// `(a cos t, b sin t)` about the center; a circle has `a = b`.
    Ellipse { a: f64, b: f64 },
    /// `(t²/4f, t)` about the apex.
    Parabola { focal: f64 },
    /// `(a cosh t, b sinh t)` about the center.
    Hyperbola { a: f64, b: f64 },
}

impl ConicShape {
    /// Five distinct parameters, enough to decide whether the whole conic
    /// lies on a quadric (see [`conic_lies_on`]).
    fn samples(self) -> [f64; 5] {
        match self {
            ConicShape::Ellipse { .. } => CLOSED_CONIC_SAMPLES,
            ConicShape::Parabola { focal: f } => [-8.0 * f, -4.0 * f, 0.0, 4.0 * f, 8.0 * f],
            ConicShape::Hyperbola { .. } => [-1.0, -0.5, 0.0, 0.5, 1.0],
        }
    }
}

/// A conic: a curve with a frame (origin at the center, or at the apex for
/// a parabola) and a [`ConicShape`] in it.
pub(crate) trait Conic: ParametricCurve3D {
    fn frame(&self) -> Frame3D;
    fn shape(&self) -> ConicShape;
}

impl Conic for Circle3D {
    #[inline]
    fn frame(&self) -> Frame3D {
        Circle3D::frame(self)
    }
    #[inline]
    fn shape(&self) -> ConicShape {
        ConicShape::Ellipse {
            a: self.radius(),
            b: self.radius(),
        }
    }
}

impl Conic for Ellipse3D {
    #[inline]
    fn frame(&self) -> Frame3D {
        Ellipse3D::frame(self)
    }
    #[inline]
    fn shape(&self) -> ConicShape {
        ConicShape::Ellipse {
            a: self.major_radius(),
            b: self.minor_radius(),
        }
    }
}

impl Conic for Parabola3D {
    #[inline]
    fn frame(&self) -> Frame3D {
        Parabola3D::frame(self)
    }
    #[inline]
    fn shape(&self) -> ConicShape {
        ConicShape::Parabola {
            focal: self.focal(),
        }
    }
}

impl Conic for Hyperbola3D {
    #[inline]
    fn frame(&self) -> Frame3D {
        Hyperbola3D::frame(self)
    }
    #[inline]
    fn shape(&self) -> ConicShape {
        ConicShape::Hyperbola {
            a: self.major_radius(),
            b: self.minor_radius(),
        }
    }
}

/// A plane, sphere, cylinder or cone, through its section by a plane.
pub(crate) trait Quadric: ParametricSurface + Sized {
    /// Whether every section is first degree, which only the plane's is.
    const LINEAR: bool = false;

    /// The section by the plane through `origin` spanned by the
    /// orthonormal `x` and `y`, in the coordinates `p = origin + αx + βy`.
    fn section(&self, origin: Point3D, x: Vector3D, y: Vector3D) -> ImplicitConic;

    /// For a circle or ellipse of semi-axes `a`, `b` centered at `origin`
    /// whose section leaves a first-order trigonometric equation, the
    /// scale of that equation's terms for [`solve_trig_linear`]; `None`
    /// when the double-angle terms remain.
    fn linear_trig_scale(&self, _origin: Point3D, _a: f64, _b: f64) -> Option<f64> {
        None
    }

    /// Whether the whole conic lies on the surface. Only an ellipse or a
    /// circle can lie on a sphere or a cylinder, so the default checks
    /// those alone.
    fn holds<C: Conic>(&self, conic: &C, tol: Tolerance) -> bool {
        let shape = conic.shape();
        matches!(shape, ConicShape::Ellipse { .. })
            && conic_lies_on(conic, self, shape.samples(), tol)
    }
}

impl Quadric for Plane {
    const LINEAR: bool = true;

    #[inline]
    fn section(&self, origin: Point3D, x: Vector3D, y: Vector3D) -> ImplicitConic {
        let n = self.normal();
        ImplicitConic {
            a2: 0.0,
            b2: 0.0,
            c2: 0.0,
            d1: x.dot(n),
            e1: y.dot(n),
            f0: (origin - self.frame().origin()).dot(n),
        }
    }

    #[inline]
    fn linear_trig_scale(&self, origin: Point3D, a: f64, b: f64) -> Option<f64> {
        Some((origin - self.frame().origin()).magnitude() + a + b)
    }

    /// Any conic lies on a plane parallel to its own through its origin.
    #[inline]
    fn holds<C: Conic>(&self, conic: &C, tol: Tolerance) -> bool {
        let frame = conic.frame();
        frame.z_direction().cross(self.normal()).magnitude() <= tol.angular
            && self.contains(frame.origin(), tol)
    }
}

impl Quadric for Sphere {
    /// `|p - c|² - r²`.
    #[inline]
    fn section(&self, origin: Point3D, x: Vector3D, y: Vector3D) -> ImplicitConic {
        let w = origin - self.center();
        let r = self.radius();
        ImplicitConic {
            a2: 1.0,
            b2: 1.0,
            c2: 0.0,
            d1: 2.0 * x.dot(w),
            e1: 2.0 * y.dot(w),
            f0: w.dot(w) - r * r,
        }
    }

    /// On a circle, `α² + β²` is constant, so the section's second-degree
    /// terms fold into the constant.
    #[inline]
    fn linear_trig_scale(&self, origin: Point3D, a: f64, b: f64) -> Option<f64> {
        if a == b {
            let scale = (origin - self.center()).magnitude() + a + self.radius();
            Some(scale * scale)
        } else {
            None
        }
    }
}

impl Quadric for Cylinder {
    /// `|p - o|² - ((p - o)·d)² - r²` for the axis `(o, d)`.
    #[inline]
    fn section(&self, origin: Point3D, x: Vector3D, y: Vector3D) -> ImplicitConic {
        let d = self.axis().direction();
        let w = origin - self.axis().origin();
        let r = self.radius();
        let (xd, yd, wd) = (x.dot(d), y.dot(d), w.dot(d));
        ImplicitConic {
            a2: 1.0 - xd * xd,
            b2: 1.0 - yd * yd,
            c2: -2.0 * xd * yd,
            d1: 2.0 * (x.dot(w) - wd * xd),
            e1: 2.0 * (y.dot(w) - wd * yd),
            f0: w.dot(w) - wd * wd - r * r,
        }
    }
}

impl Quadric for Cone {
    /// `((p - apex)·d)² - cos²φ |p - apex|²` for the axis direction `d`
    /// and the semi-angle `φ`: both nappes.
    #[inline]
    fn section(&self, origin: Point3D, x: Vector3D, y: Vector3D) -> ImplicitConic {
        let d = self.frame().z_direction();
        let cos = self.semi_angle().cos();
        let c2 = cos * cos;
        let w = origin - self.apex();
        let (xd, yd, wd) = (x.dot(d), y.dot(d), w.dot(d));
        ImplicitConic {
            a2: xd * xd - c2,
            b2: yd * yd - c2,
            c2: 2.0 * xd * yd,
            d1: 2.0 * (wd * xd - c2 * x.dot(w)),
            e1: 2.0 * (wd * yd - c2 * y.dot(w)),
            f0: wd * wd - c2 * w.dot(w),
        }
    }

    /// A cone holds the circles and ellipses that are its plane sections,
    /// and also parabolas and hyperbolas.
    fn holds<C: Conic>(&self, conic: &C, tol: Tolerance) -> bool {
        conic_lies_on(conic, self, conic.shape().samples(), tol)
    }
}

impl ImplicitConic {
    /// Coefficients `(a2, b2, c1, d1, e)` of
    /// `a2*cos2t + b2*sin2t + c1*cos t + d1*sin t + e = 0` for the ellipse
    /// `(a cos t, b sin t)` substituted into the implicit equation (see
    /// [`solve_trig`]).
    #[inline]
    fn ellipse_substitute(&self, a: f64, b: f64) -> (f64, f64, f64, f64, f64) {
        let (qa, qb) = (self.a2 * a * a, self.b2 * b * b);
        (
            (qa - qb) / 2.0,
            self.c2 * a * b / 2.0,
            self.d1 * a,
            self.e1 * b,
            self.f0 + (qa + qb) / 2.0,
        )
    }

    /// Ascending coefficients of the quartic in `t` for the parabola
    /// `(t²/4f, t)` substituted into the implicit equation.
    #[inline]
    fn parabola_substitute(&self, focal: f64) -> [f64; 5] {
        let k = 1.0 / (4.0 * focal);
        [
            self.f0,
            self.e1,
            self.b2 + self.d1 * k,
            self.c2 * k,
            self.a2 * k * k,
        ]
    }

    /// Ascending coefficients of the quartic in `u = eᵗ` for the hyperbola
    /// `(a cosh t, b sinh t)` substituted into the implicit equation and
    /// multiplied by `4u²`.
    #[inline]
    fn hyperbola_substitute(&self, a: f64, b: f64) -> [f64; 5] {
        let (qa, qb, qc) = (self.a2 * a * a, self.b2 * b * b, self.c2 * a * b);
        let (da, eb) = (2.0 * self.d1 * a, 2.0 * self.e1 * b);
        [
            qa + qb - qc,
            da - eb,
            2.0 * (qa - qb) + 4.0 * self.f0,
            da + eb,
            qa + qb + qc,
        ]
    }
}

/// Intersects a conic with a quadric: [`ConicSurfaceIntersection::Coincident`]
/// when the whole conic lies on it, else the hits verified on it.
#[inline]
pub(crate) fn intersect_conic_quadric<C: Conic, S: Quadric>(
    conic: &C,
    quadric: &S,
    tol: Tolerance,
) -> ConicSurfaceIntersection {
    if quadric.holds(conic, tol) {
        return ConicSurfaceIntersection::Coincident;
    }
    let frame = conic.frame();
    let origin = frame.origin();
    let section = quadric.section(origin, frame.x_direction(), frame.y_direction());
    let touches = |t: f64| quadric.contains(conic.eval_point(t), tol);
    let candidates = match conic.shape() {
        ConicShape::Ellipse { a, b } => {
            let (a2, b2, c1, d1, e) = section.ellipse_substitute(a, b);
            match quadric.linear_trig_scale(origin, a, b) {
                Some(scale) => solve_trig_linear(c1, d1, e, scale, tol),
                None => solve_trig(a2, b2, c1, d1, e, tol, touches),
            }
        }
        ConicShape::Parabola { focal } => {
            let q = section.parabola_substitute(focal);
            if S::LINEAR {
                // The quadratic `q[2]*t² + q[1]*t + q[0]`.
                let vertex = -q[1] / (2.0 * q[2]);
                quadratic_candidates(
                    solve_quadratic(q[2], q[1], q[0], tol).touching_at(vertex, touches),
                )
            } else {
                roots_with_tangency(&q, tol, touches)
                    .into_iter()
                    .map(|root| (root.value, root.multiplicity))
                    .collect()
            }
        }
        ConicShape::Hyperbola { a, b } => {
            let q = section.hyperbola_substitute(a, b);
            let touches_u = |u: f64| u > 0.0 && touches(u.ln());
            let roots = if S::LINEAR {
                // `q[0]` and `q[4]` vanish: the quadratic
                // `(q[3]*u² + q[2]*u + q[1]) / 2`.
                let (qa, qb, qc) = (q[3] / 2.0, q[2] / 2.0, q[1] / 2.0);
                quadratic_candidates(
                    solve_quadratic(qa, qb, qc, tol).touching_at(-qb / (2.0 * qa), touches_u),
                )
            } else {
                roots_with_tangency(&q, tol, touches_u)
                    .into_iter()
                    .map(|root| (root.value, root.multiplicity))
                    .collect()
            };
            roots
                .into_iter()
                .filter(|&(u, _)| u > 0.0)
                .map(|(u, multiplicity)| (u.ln(), multiplicity))
                .collect()
        }
    };
    let hits = verify_conic_hits(conic, quadric, candidates, tol);
    if hits.is_empty() {
        ConicSurfaceIntersection::Empty
    } else {
        ConicSurfaceIntersection::Hits(hits)
    }
}

/// The roots of a quadratic as `(parameter, multiplicity)` candidates.
fn quadratic_candidates(solution: QuadraticSolution) -> Vec<(f64, u32)> {
    match solution {
        QuadraticSolution::Two(t1, t2) => vec![(t1, 1), (t2, 1)],
        QuadraticSolution::One(t) => vec![(t, 2)],
        QuadraticSolution::Linear(t) => vec![(t, 1)],
        QuadraticSolution::Empty | QuadraticSolution::Degenerate => Vec::new(),
    }
}

/// Keep trig/quartic candidates verified on the surface. A point far
/// from the origin cannot be placed (nor its surface distance measured)
/// more finely than its coordinates' rounding, so the check's tolerance
/// is at least a few ulps of the point's largest coordinate.
fn verify_conic_hits<C, S>(
    curve: &C,
    surface: &S,
    candidates: Vec<(f64, u32)>,
    tol: Tolerance,
) -> Vec<ConicSurfaceHit>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    candidates
        .into_iter()
        .filter_map(|(t, multiplicity)| {
            let p = curve.eval_point(t);
            let coord = p.x.abs().max(p.y.abs()).max(p.z.abs());
            let tol = Tolerance {
                confusion: tol.confusion.max(16.0 * f64::EPSILON * coord),
                ..tol
            };
            if surface.contains(p, tol) {
                Some(ConicSurfaceHit {
                    parameter: t,
                    point: p,
                    multiplicity,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Five equally spaced parameters around a circle or an ellipse, enough
/// to decide whether the whole curve lies on a quadric (see
/// [`conic_lies_on`]).
const CLOSED_CONIC_SAMPLES: [f64; 5] = [
    0.0,
    0.4 * std::f64::consts::PI,
    0.8 * std::f64::consts::PI,
    1.2 * std::f64::consts::PI,
    1.6 * std::f64::consts::PI,
];

/// Whether the whole conic lies on a quadric surface. On a conic, the
/// surface's equation is a polynomial of degree at most 4 (in `t`,
/// `e^t`, or `tan(t/2)` for the parabola, hyperbola, and circle or
/// ellipse), so five distinct `params` on the surface force it to vanish
/// identically. Three would not do for a circle on a cylinder or cone:
/// their section by the circle's plane is a conic, which a circle can
/// cross at three points without lying on it.
fn conic_lies_on<C, S>(curve: &C, surface: &S, params: [f64; 5], tol: Tolerance) -> bool
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    params
        .iter()
        .all(|&t| surface.contains(curve.eval_point(t), tol))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For every conic shape and parameter, the substituted equation
    /// equals the quadric's `implicit` equation, written in world
    /// coordinates independently of [`Quadric::section`], at the conic's
    /// point.
    fn check_substitution<S: Quadric>(quadric: &S, implicit: &dyn Fn(Point3D) -> f64) {
        let frame = Frame3D::new(
            Point3D::new(0.3, -0.4, 0.7),
            Vector3D::new(0.2, -0.5, 1.0),
            Vector3D::new(1.0, 0.3, 0.05),
        )
        .unwrap();
        let (o, x, y) = (frame.origin(), frame.x_direction(), frame.y_direction());
        let section = quadric.section(o, x, y);
        let at = |a: f64, b: f64| o + x * a + y * b;
        let (a, b, focal) = (1.7, 0.9, 0.6);
        for t in [-2.5f64, -1.0, -0.3, 0.0, 0.4, 1.3, 2.9] {
            let close = |got: f64, want: f64| {
                assert!(
                    (got - want).abs() <= 1e-12 * (1.0 + want.abs()) * 16.0,
                    "{got} vs {want} at t = {t}"
                );
            };
            let (a2, b2, c1, d1, e) = section.ellipse_substitute(a, b);
            let trig =
                a2 * (2.0 * t).cos() + b2 * (2.0 * t).sin() + c1 * t.cos() + d1 * t.sin() + e;
            close(trig, implicit(at(a * t.cos(), b * t.sin())));
            let q = section.parabola_substitute(focal);
            let poly = q.iter().rev().fold(0.0, |acc, c| acc * t + c);
            close(poly, implicit(at(t * t / (4.0 * focal), t)));
            let u = t.exp();
            let q = section.hyperbola_substitute(a, b);
            let poly = q.iter().rev().fold(0.0, |acc, c| acc * u + c) / (4.0 * u * u);
            close(poly, implicit(at(a * t.cosh(), b * t.sinh())));
        }
    }

    #[test]
    fn test_substitutions_match_the_implicit_equations() {
        let plane =
            Plane::new(Point3D::new(0.1, 0.2, -0.3), Vector3D::new(0.3, -0.2, 1.0)).unwrap();
        let (po, n) = (plane.frame().origin(), plane.normal());
        check_substitution(&plane, &|p| (p - po).dot(n));

        let sphere = Sphere::new(Point3D::new(-0.2, 0.5, 0.1), 1.3).unwrap();
        check_substitution(&sphere, &|p| {
            let w = p - sphere.center();
            w.dot(w) - 1.3 * 1.3
        });

        let cylinder = Cylinder::new(
            Point3D::new(0.4, 0.0, -0.2),
            Vector3D::new(0.1, 1.0, 0.3),
            0.8,
        )
        .unwrap();
        let (co, d) = (cylinder.axis().origin(), cylinder.axis().direction());
        check_substitution(&cylinder, &|p| {
            let w = p - co;
            w.dot(w) - w.dot(d).powi(2) - 0.8 * 0.8
        });

        let cone = Cone::from_frame(
            Frame3D::new(
                Point3D::new(0.2, -0.1, -1.0),
                Vector3D::new(0.2, 0.1, 1.0),
                Vector3D::X,
            )
            .unwrap(),
            0.45,
            0.7,
        )
        .unwrap();
        let (apex, d, c2) = (
            cone.apex(),
            cone.frame().z_direction(),
            0.45f64.cos().powi(2),
        );
        check_substitution(&cone, &|p| {
            let w = p - apex;
            w.dot(d).powi(2) - c2 * w.dot(w)
        });
    }
}
