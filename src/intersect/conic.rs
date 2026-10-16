//! Exact intersections of a conic with a quadric or with another conic.
//!
//! Every pair is solved the same way. The quadric's section by the
//! conic's plane, or the other conic's implicit equation when the two are
//! coplanar, is an [`ImplicitConic`] in the conic's frame coordinates
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
//!
//! Two conics in crossing planes can only meet on the line where the
//! planes cross: the hits of one conic on the other's plane, kept when
//! they lie on the other conic.

use super::results::sort_by_known_parameter;
use super::{
    ConicConicIntersection, ConicSurfaceHit, ConicSurfaceIntersection, CurveCurveHit, ImplicitConic,
};
use crate::curves::ParametricCurve3D;
use crate::math::{
    QuadraticSolution, RealRoot, roots_with_tangency, solve_quadratic, solve_trig,
    solve_trig_linear,
};
use crate::surfaces::ParametricSurface;
use crate::{
    Circle3D, Cone, Cylinder, Ellipse3D, Frame3D, Hyperbola3D, InlineList, IntersectExact,
    Parabola3D, Plane, Point3D, Sphere, Tolerance, Vector3D,
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

    /// How far the conic reaches from its origin: the lever arm of a turn
    /// about it (ADR 0006). Parabolas and hyperbolas are unbounded.
    pub(crate) fn extent(self) -> f64 {
        match self {
            ConicShape::Ellipse { a, b } => a.max(b),
            ConicShape::Parabola { .. } | ConicShape::Hyperbola { .. } => f64::INFINITY,
        }
    }

    /// The implicit equation in the conic's own frame coordinates.
    fn implicit(self) -> ImplicitConic {
        match self {
            ConicShape::Ellipse { a, b } => ImplicitConic::ellipse(a, b),
            ConicShape::Parabola { focal } => ImplicitConic::parabola(focal),
            ConicShape::Hyperbola { a, b } => ImplicitConic::hyperbola(a, b),
        }
    }

    /// Which of two coplanar conics substitutes into the other's implicit
    /// equation: the lower rank. A bounded angle (the trigonometric
    /// equation) beats a polynomial parameter, which beats `u = eᵗ`.
    fn rank(self) -> u8 {
        match self {
            ConicShape::Ellipse { .. } => 0,
            ConicShape::Parabola { .. } => 1,
            ConicShape::Hyperbola { .. } => 2,
        }
    }
}

/// A conic: a curve with a frame (origin at the center, or at the apex for
/// a parabola) and a [`ConicShape`] in it.
pub(crate) trait Conic: ParametricCurve3D {
    fn frame(&self) -> Frame3D;
    fn shape(&self) -> ConicShape;
    /// Whether `point` is within tolerance of the curve.
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool;
    /// The parameter of the curve point nearest `point`.
    fn parameter_of(&self, point: Point3D) -> f64;
    /// The parameter to report for a hit found at `t`: in the curve's
    /// bounds, with the seam of a closed conic snapped to their start.
    #[inline]
    fn parameter_in_bounds(&self, t: f64, _tol: Tolerance) -> f64 {
        t
    }
}

impl Conic for Circle3D {
    #[inline]
    fn frame(&self) -> Frame3D {
        Circle3D::frame(self)
    }
    #[inline]
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Circle3D::contains(self, point, tol)
    }
    #[inline]
    fn parameter_of(&self, point: Point3D) -> f64 {
        Circle3D::parameter_of(self, point)
    }
    #[inline]
    fn parameter_in_bounds(&self, t: f64, tol: Tolerance) -> f64 {
        Circle3D::parameter_in_bounds(self, t, tol)
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
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Ellipse3D::contains(self, point, tol)
    }
    #[inline]
    fn parameter_of(&self, point: Point3D) -> f64 {
        Ellipse3D::parameter_of(self, point)
    }
    #[inline]
    fn parameter_in_bounds(&self, t: f64, tol: Tolerance) -> f64 {
        Ellipse3D::parameter_in_bounds(self, t, tol)
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
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Parabola3D::contains(self, point, tol)
    }
    #[inline]
    fn parameter_of(&self, point: Point3D) -> f64 {
        Parabola3D::parameter_of(self, point)
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
    fn contains(&self, point: Point3D, tol: Tolerance) -> bool {
        Hyperbola3D::contains(self, point, tol)
    }
    #[inline]
    fn parameter_of(&self, point: Point3D) -> f64 {
        Hyperbola3D::parameter_of(self, point)
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

    /// Any conic lies on a plane through its origin whose normal turns
    /// onto its own while moving the conic by at most the confusion
    /// tolerance.
    #[inline]
    fn holds<C: Conic>(&self, conic: &C, tol: Tolerance) -> bool {
        let frame = conic.frame();
        let angular = tol.angular_for_extent(conic.shape().extent());
        frame.z_direction().cross(self.normal()).magnitude() <= angular
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
    /// The same conic in coordinates `(α, β)` of the same plane, where its
    /// own coordinates are `u = u0 + p*α + q*β` and `v = v0 + r*α + s*β`.
    #[inline]
    fn pulled_back(&self, (u0, p, q): (f64, f64, f64), (v0, r, s): (f64, f64, f64)) -> Self {
        let (a2, b2, c2, d1, e1) = (self.a2, self.b2, self.c2, self.d1, self.e1);
        ImplicitConic {
            a2: a2 * p * p + b2 * r * r + c2 * p * r,
            b2: a2 * q * q + b2 * s * s + c2 * q * s,
            c2: 2.0 * (a2 * p * q + b2 * r * s) + c2 * (p * s + q * r),
            d1: 2.0 * (a2 * u0 * p + b2 * v0 * r) + c2 * (u0 * r + v0 * p) + d1 * p + e1 * r,
            e1: 2.0 * (a2 * u0 * q + b2 * v0 * s) + c2 * (u0 * s + v0 * q) + d1 * q + e1 * s,
            f0: a2 * u0 * u0 + b2 * v0 * v0 + c2 * u0 * v0 + d1 * u0 + e1 * v0 + self.f0,
        }
    }

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
    let candidates = section_candidates(
        conic,
        section,
        S::LINEAR,
        |a, b| quadric.linear_trig_scale(origin, a, b),
        touches,
        tol,
    );
    let mut hits = verify_conic_hits(conic, quadric, &candidates, tol);
    if hits.is_empty() {
        return ConicSurfaceIntersection::Empty;
    }
    for h in &mut hits {
        h.parameter = conic.parameter_in_bounds(h.parameter, tol);
    }
    hits.sort_unstable_by(|x, y| x.parameter.total_cmp(&y.parameter));
    ConicSurfaceIntersection::Hits(hits)
}

/// Intersects two conics: [`ConicConicIntersection::Coincident`] when
/// they are the same curve, else their hits, ordered by the first conic's
/// parameter when every hit has one. The conic of lower
/// [`ConicShape::rank`] substitutes into the other's equation, so only its
/// parameter is computed; the hits come back in the callers' order.
pub(crate) fn intersect_conic_conic<A: Conic, B: Conic>(
    first: &A,
    second: &B,
    tol: Tolerance,
) -> ConicConicIntersection {
    let mut hits = if second.shape().rank() < first.shape().rank() {
        let Some(mut hits) = conic_conic_hits(second, first, tol) else {
            return ConicConicIntersection::Coincident;
        };
        for h in hits.iter_mut() {
            (h.first_parameter, h.second_parameter) = (h.second_parameter, h.first_parameter);
        }
        hits
    } else {
        let Some(hits) = conic_conic_hits(first, second, tol) else {
            return ConicConicIntersection::Coincident;
        };
        hits
    };
    if hits.is_empty() {
        return ConicConicIntersection::Empty;
    }
    sort_by_known_parameter(&mut hits, |hit| hit.first_parameter);
    ConicConicIntersection::Hits(hits.to_vec())
}

/// The hits of `conic` on `other`, with `conic`'s parameter first, or
/// `None` when they are the same curve. The solve yields `conic`'s
/// parameter; `other`'s is stored only where the tangency test needed it.
fn conic_conic_hits<A: Conic, B: Conic>(
    conic: &A,
    other: &B,
    tol: Tolerance,
) -> Option<InlineList<CurveCurveHit, 4>> {
    let mut hits = InlineList::new();
    let (frame, other_frame) = (conic.frame(), other.frame());
    let normal = frame.z_direction();
    let hit = |t: f64, p: Point3D, multiplicity: u32| CurveCurveHit {
        first_parameter: Some(conic.parameter_in_bounds(t, tol)),
        point: p,
        second_parameter: None,
        multiplicity,
    };
    // Coplanar when either conic lies in the other's plane: solving in
    // `conic`'s plane then moves neither by more than the confusion
    // tolerance.
    let plane = Plane::from_frame(other_frame);
    if !plane.holds(conic, tol) && !Plane::from_frame(frame).holds(other, tol) {
        if normal.cross(other_frame.z_direction()).magnitude() <= tol.angular {
            // Distinct parallel planes.
            return Some(hits);
        }
        // Crossing planes: the hits of `conic` on the other's plane that
        // lie on the other conic. They are tangent only where both conics
        // run along the line the planes share: `conic` grazes the plane
        // and the other conic's tangent lies in `conic`'s plane.
        let along_tolerance = tol.angular_for_extent(other.shape().extent());
        if let ConicSurfaceIntersection::Hits(on_plane) =
            intersect_conic_quadric(conic, &plane, tol)
        {
            for h in on_plane {
                if other.contains(h.point, point_tolerance(h.point, tol)) {
                    let mut found = hit(h.parameter, h.point, 1);
                    if h.multiplicity >= 2 {
                        // The tangency test needs the other conic's
                        // parameter, so it is kept.
                        let second = other.parameter_in_bounds(other.parameter_of(h.point), tol);
                        found.second_parameter = Some(second);
                        let along = other.eval_derivative(second, 1);
                        if along.dot(normal).abs() <= along_tolerance * along.magnitude() {
                            found.multiplicity = h.multiplicity;
                        }
                    }
                    hits.push(found);
                }
            }
        }
        return Some(hits);
    }
    let origin = frame.origin();
    if same_conic(conic, other, tol) {
        return None;
    }
    // The other conic's own coordinates, affine in `conic`'s.
    let (x, y) = (frame.x_direction(), frame.y_direction());
    let (ox, oy) = (other_frame.x_direction(), other_frame.y_direction());
    let w = origin - other_frame.origin();
    let section = other.shape().implicit().pulled_back(
        (w.dot(ox), x.dot(ox), y.dot(ox)),
        (w.dot(oy), x.dot(oy), y.dot(oy)),
    );
    let touches = |t: f64| other.contains(conic.eval_point(t), tol);
    let candidates = section_candidates(conic, section, false, |_, _| None, touches, tol);
    for &(t, multiplicity) in &candidates {
        let p = conic.eval_point(t);
        if other.contains(p, point_tolerance(p, tol)) {
            hits.push(hit(t, p, multiplicity));
        }
    }
    Some(hits)
}

/// Whether two coplanar conics are the same curve: the same shape and
/// size about the same origin, with the same axis unless the shape is a
/// circle (a parabola or a hyperbola branch opens along its axis, so the
/// axes also point the same way).
fn same_conic<A: Conic, B: Conic>(first: &A, second: &B, tol: Tolerance) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= tol.confusion;
    let (f1, f2) = (first.frame(), second.frame());
    let (x1, x2) = (f1.x_direction(), f2.x_direction());
    let parallel = x1.cross(x2).magnitude() <= tol.angular_for_extent(first.shape().extent());
    let same_way = parallel && x1.dot(x2) > 0.0;
    f1.origin().distance(f2.origin()) <= tol.confusion
        && match (first.shape(), second.shape()) {
            (ConicShape::Ellipse { a: a1, b: b1 }, ConicShape::Ellipse { a: a2, b: b2 }) => {
                close(a1, a2) && close(b1, b2) && (close(a1, b1) || parallel)
            }
            (ConicShape::Parabola { focal: f1 }, ConicShape::Parabola { focal: f2 }) => {
                close(f1, f2) && same_way
            }
            (ConicShape::Hyperbola { a: a1, b: b1 }, ConicShape::Hyperbola { a: a2, b: b2 }) => {
                close(a1, a2) && close(b1, b2) && same_way
            }
            _ => false,
        }
}

/// The tolerance to check a point against: a point far from the origin
/// cannot be placed (nor its distance measured) more finely than its
/// coordinates' rounding, so at least a few ulps of its largest
/// coordinate.
fn point_tolerance(p: Point3D, tol: Tolerance) -> Tolerance {
    let coord = p.x.abs().max(p.y.abs()).max(p.z.abs());
    Tolerance {
        confusion: tol.confusion.max(16.0 * f64::EPSILON * coord),
        ..tol
    }
}

/// Implements [`IntersectExact`] between two conics with
/// [`intersect_conic_conic`].
macro_rules! conic_conic {
    ($($a:ty => $b:ty),* $(,)?) => {$(
        impl IntersectExact<$b> for $a {
            type Output = ConicConicIntersection;

            /// Intersects with another conic; see [`ConicConicIntersection`].
            fn intersect_exact(&self, other: &$b, tol: Tolerance) -> ConicConicIntersection {
                intersect_conic_conic(self, other, tol)
            }
        }
    )*};
}

conic_conic!(
    Circle3D => Ellipse3D,
    Circle3D => Parabola3D,
    Circle3D => Hyperbola3D,
    Ellipse3D => Circle3D,
    Ellipse3D => Ellipse3D,
    Ellipse3D => Parabola3D,
    Ellipse3D => Hyperbola3D,
    Parabola3D => Circle3D,
    Parabola3D => Ellipse3D,
    Parabola3D => Parabola3D,
    Parabola3D => Hyperbola3D,
    Hyperbola3D => Circle3D,
    Hyperbola3D => Ellipse3D,
    Hyperbola3D => Parabola3D,
    Hyperbola3D => Hyperbola3D,
);

/// The hit candidates of `conic` on the `section` in its frame
/// coordinates: its parametrization substituted and solved, with
/// `touches(t)` deciding grazing contact. A `linear` section (a plane's)
/// leaves a quadratic; `trig_scale(a, b)`, for a circle or ellipse of
/// semi-axes `a` and `b`, says whether it leaves a first-order
/// trigonometric equation and its scale.
fn section_candidates<C: Conic>(
    conic: &C,
    section: ImplicitConic,
    linear: bool,
    trig_scale: impl FnOnce(f64, f64) -> Option<f64>,
    touches: impl Fn(f64) -> bool,
    tol: Tolerance,
) -> Candidates {
    match conic.shape() {
        ConicShape::Ellipse { a, b } => {
            let (a2, b2, c1, d1, e) = section.ellipse_substitute(a, b);
            match trig_scale(a, b) {
                Some(scale) => solve_trig_linear(c1, d1, e, scale, tol),
                None => solve_trig(a2, b2, c1, d1, e, tol, touches),
            }
        }
        ConicShape::Parabola { focal } => {
            let q = section.parabola_substitute(focal);
            if linear {
                // The quadratic `q[2]*t² + q[1]*t + q[0]`.
                let vertex = -q[1] / (2.0 * q[2]);
                quadratic_candidates(
                    solve_quadratic(q[2], q[1], q[0], tol).touching_at(vertex, touches),
                )
            } else {
                root_candidates(&roots_with_tangency(&q, tol, touches))
            }
        }
        ConicShape::Hyperbola { a, b } => {
            let q = section.hyperbola_substitute(a, b);
            let touches_u = |u: f64| u > 0.0 && touches(u.ln());
            let u_roots = if linear {
                // `q[0]` and `q[4]` vanish: the quadratic
                // `(q[3]*u² + q[2]*u + q[1]) / 2`.
                let (qa, qb, qc) = (q[3] / 2.0, q[2] / 2.0, q[1] / 2.0);
                quadratic_candidates(
                    solve_quadratic(qa, qb, qc, tol).touching_at(-qb / (2.0 * qa), touches_u),
                )
            } else {
                root_candidates(&roots_with_tangency(&q, tol, touches_u))
            };
            let mut candidates = Candidates::new();
            for &(u, multiplicity) in u_roots.iter().filter(|&&(u, _)| u > 0.0) {
                candidates.push((u.ln(), multiplicity));
            }
            candidates
        }
    }
}

/// Hit candidates: `(parameter, multiplicity)`, at most four.
type Candidates = InlineList<(f64, u32), 4>;

/// Polynomial roots as hit candidates.
fn root_candidates(roots: &[RealRoot]) -> Candidates {
    let mut out = Candidates::new();
    for root in roots {
        out.push((root.value, root.multiplicity));
    }
    out
}

/// The roots of a quadratic as hit candidates.
fn quadratic_candidates(solution: QuadraticSolution) -> Candidates {
    let mut out = Candidates::new();
    match solution {
        QuadraticSolution::Two(t1, t2) => {
            out.push((t1, 1));
            out.push((t2, 1));
        }
        QuadraticSolution::One(t) => out.push((t, 2)),
        QuadraticSolution::Linear(t) => out.push((t, 1)),
        QuadraticSolution::Empty | QuadraticSolution::Degenerate => {}
    }
    out
}

/// Keep trig/quartic candidates verified on the surface, each with the
/// [`point_tolerance`] of its point.
fn verify_conic_hits<C, S>(
    curve: &C,
    surface: &S,
    candidates: &[(f64, u32)],
    tol: Tolerance,
) -> Vec<ConicSurfaceHit>
where
    C: ParametricCurve3D,
    S: ParametricSurface,
{
    candidates
        .iter()
        .filter_map(|&(t, multiplicity)| {
            let p = curve.eval_point(t);
            if surface.contains(p, point_tolerance(p, tol)) {
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

    /// The hits of `first` on `second`, each checked to sit at the
    /// parameters it stores, and the same hits with the parameters swapped
    /// when the order is reversed.
    fn conic_hits<A, B>(first: &A, second: &B) -> Vec<CurveCurveHit>
    where
        A: Conic + IntersectExact<B, Output = ConicConicIntersection>,
        B: Conic + IntersectExact<A, Output = ConicConicIntersection>,
    {
        let tol = Tolerance::DEFAULT;
        let ConicConicIntersection::Hits(hits) = first.intersect_exact(second, tol) else {
            panic!("expected hits");
        };
        let ConicConicIntersection::Hits(back) = second.intersect_exact(first, tol) else {
            panic!("expected hits the other way round");
        };
        assert_eq!(hits.len(), back.len());
        // The substituted conic's parameter is always computed.
        assert!(
            hits.iter()
                .all(|h| h.first_parameter.is_some() || h.second_parameter.is_some())
        );
        for h in &hits {
            let b = back
                .iter()
                .min_by(|p, q| {
                    p.point
                        .distance(h.point)
                        .total_cmp(&q.point.distance(h.point))
                })
                .unwrap();
            if let Some(t) = h.first_parameter {
                assert!(first.eval_point(t).distance(h.point) < 1e-9);
            }
            if let Some(t) = h.second_parameter {
                assert!(second.eval_point(t).distance(h.point) < 1e-9);
            }
            assert!(h.point.distance(b.point) < 1e-9);
            // Equal ranks substitute the receiver, so each order may
            // compute a different conic's parameter; stored ones agree.
            for (x, y) in [
                (h.first_parameter, b.second_parameter),
                (h.second_parameter, b.first_parameter),
            ] {
                if let (Some(x), Some(y)) = (x, y) {
                    assert!((x - y).abs() < 1e-9);
                }
            }
            assert_eq!(h.multiplicity, b.multiplicity);
        }
        hits
    }

    /// The hits' points, sorted by `x` (to rounding) then `y`, and their
    /// multiplicities.
    fn points_and_multiplicities(mut hits: Vec<CurveCurveHit>) -> Vec<(Point3D, u32)> {
        hits.sort_by(|p, q| {
            let (p, q) = (p.point, q.point);
            if (p.x - q.x).abs() > 1e-9 {
                p.x.total_cmp(&q.x)
            } else {
                p.y.total_cmp(&q.y)
            }
        });
        hits.iter().map(|h| (h.point, h.multiplicity)).collect()
    }

    fn assert_hits(hits: Vec<CurveCurveHit>, want: &[(f64, f64, u32)]) {
        let got = points_and_multiplicities(hits);
        assert_eq!(got.len(), want.len(), "{got:?}");
        for ((p, m), &(x, y, wm)) in got.iter().zip(want) {
            assert!(
                p.distance(Point3D::new(x, y, 0.0)) < 1e-9,
                "{p:?} vs ({x}, {y})"
            );
            assert_eq!(*m, wm, "multiplicity at {p:?}");
        }
    }

    fn ellipse(center: (f64, f64), x_axis: Vector3D, a: f64, b: f64) -> Ellipse3D {
        Ellipse3D::new(
            Point3D::new(center.0, center.1, 0.0),
            Vector3D::Z,
            x_axis,
            a,
            b,
        )
        .unwrap()
    }

    fn parabola(apex: (f64, f64), x_axis: Vector3D, focal: f64) -> Parabola3D {
        Parabola3D::new(
            Point3D::new(apex.0, apex.1, 0.0),
            Vector3D::Z,
            x_axis,
            focal,
        )
        .unwrap()
    }

    fn hyperbola(center: (f64, f64), x_axis: Vector3D, a: f64, b: f64) -> Hyperbola3D {
        Hyperbola3D::new(
            Point3D::new(center.0, center.1, 0.0),
            Vector3D::Z,
            x_axis,
            a,
            b,
        )
        .unwrap()
    }

    #[test]
    fn test_pulled_back_matches_the_other_conic() {
        let frame = Frame3D::new(
            Point3D::new(0.3, -0.4, 0.0),
            Vector3D::Z,
            Vector3D::new(1.0, 0.7, 0.0),
        )
        .unwrap();
        let other = Frame3D::new(
            Point3D::new(-1.1, 0.6, 0.0),
            Vector3D::Z,
            Vector3D::new(-0.2, 1.0, 0.0),
        )
        .unwrap();
        let (x, y) = (frame.x_direction(), frame.y_direction());
        let (ox, oy) = (other.x_direction(), other.y_direction());
        let w = frame.origin() - other.origin();
        let implicit = ConicShape::Hyperbola { a: 1.3, b: 0.8 }.implicit();
        let pulled = implicit.pulled_back(
            (w.dot(ox), x.dot(ox), y.dot(ox)),
            (w.dot(oy), x.dot(oy), y.dot(oy)),
        );
        let value = |c: &ImplicitConic, u: f64, v: f64| {
            c.a2 * u * u + c.b2 * v * v + c.c2 * u * v + c.d1 * u + c.e1 * v + c.f0
        };
        for (alpha, beta) in [(0.0, 0.0), (1.5, -0.3), (-2.2, 0.9), (0.4, 3.1)] {
            let p = frame.origin() + x * alpha + y * beta;
            let (u, v) = ((p - other.origin()).dot(ox), (p - other.origin()).dot(oy));
            let want = value(&implicit, u, v);
            let got = value(&pulled, alpha, beta);
            assert!(
                (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn test_ellipse_meets_one_hyperbola_branch() {
        // x²/9 + y²/4 = 1 and x² − 4y² = 1 meet at (±√153/5, ±√128/10);
        // the branch x > 0 holds two of them.
        let (x, y) = (153f64.sqrt() / 5.0, 128f64.sqrt() / 10.0);
        let e = ellipse((0.0, 0.0), Vector3D::X, 3.0, 2.0);
        let h = hyperbola((0.0, 0.0), Vector3D::X, 1.0, 0.5);
        assert_hits(conic_hits(&e, &h), &[(x, -y, 1), (x, y, 1)]);
        // An ellipse meeting only the mirror branch.
        let left = ellipse((-3.0, 0.0), Vector3D::X, 2.5, 1.0);
        assert_eq!(
            left.intersect_exact(&h, Tolerance::DEFAULT),
            ConicConicIntersection::Empty
        );
    }

    #[test]
    fn test_coplanar_conic_pairs_cross() {
        // y² = 4x against x²/4 + y² = 1: x² + 16x − 4 = 0.
        let x = -8.0 + 68f64.sqrt();
        let y = (4.0 * x).sqrt();
        let e = ellipse((0.0, 0.0), Vector3D::X, 2.0, 1.0);
        let p = parabola((0.0, 0.0), Vector3D::X, 1.0);
        assert_hits(conic_hits(&e, &p), &[(x, -y, 1), (x, y, 1)]);
        // y² = 4x against x² − y² = 1 (x > 0): x² − 4x − 1 = 0.
        let x = 2.0 + 5f64.sqrt();
        let y = (4.0 * x).sqrt();
        let h = hyperbola((0.0, 0.0), Vector3D::X, 1.0, 1.0);
        assert_hits(conic_hits(&p, &h), &[(x, -y, 1), (x, y, 1)]);
        // y² = 4x against x = y²/4 shifted and opening the other way:
        // x = 2 − y²/4 meets it at x = 1, y = ±2.
        let q = parabola((2.0, 0.0), -Vector3D::X, 1.0);
        assert_hits(conic_hits(&p, &q), &[(1.0, -2.0, 1), (1.0, 2.0, 1)]);
        // x² − y² = 1 against the branch (y + 2)² − x² = 1 opening up:
        // y = −1/2 and x² = 5/4, on the right branch only for x > 0.
        let up = hyperbola((0.0, -2.0), Vector3D::Y, 1.0, 1.0);
        assert_hits(conic_hits(&h, &up), &[(1.25f64.sqrt(), -0.5, 1)]);
        // A circle against a parabola: x² + y² = 4 and y² = 4x.
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let x = -2.0 + 8f64.sqrt();
        let y = (4.0 * x).sqrt();
        assert_hits(conic_hits(&c, &p), &[(x, -y, 1), (x, y, 1)]);
    }

    #[test]
    fn test_coplanar_conics_touch() {
        // Inside the branch x² − 4y² = 4, touching its vertex (2, 0).
        let h = hyperbola((0.0, 0.0), Vector3D::X, 2.0, 1.0);
        let e = ellipse((3.0, 0.0), Vector3D::X, 1.0, 0.5);
        assert_hits(conic_hits(&e, &h), &[(2.0, 0.0, 2)]);
        // Back to back at the apex.
        let p = parabola((0.0, 0.0), Vector3D::X, 1.0);
        let q = parabola((0.0, 0.0), -Vector3D::X, 0.5);
        assert_hits(conic_hits(&p, &q), &[(0.0, 0.0, 2)]);
        // Sharing the vertex (2, 0), then crossing where y² = 48.
        let wide = parabola((2.0, 0.0), Vector3D::X, 1.0);
        let y = 48f64.sqrt();
        assert_hits(
            conic_hits(&wide, &h),
            &[(2.0, 0.0, 2), (14.0, -y, 1), (14.0, y, 1)],
        );
        // Two branches sharing their vertex, one wider.
        let flat = hyperbola((0.0, 0.0), Vector3D::X, 2.0, 2.0);
        assert_hits(conic_hits(&h, &flat), &[(2.0, 0.0, 2)]);
        // The parabola's apex on the ellipse's vertex.
        let e = ellipse((-2.0, 0.0), Vector3D::X, 2.0, 1.0);
        assert_hits(conic_hits(&e, &p), &[(0.0, 0.0, 2)]);
    }

    #[test]
    fn test_same_and_parallel_conics() {
        let tol = Tolerance::DEFAULT;
        let p = parabola((1.0, 2.0), Vector3D::new(1.0, 1.0, 0.0), 0.5);
        assert_eq!(
            p.intersect_exact(&p, tol),
            ConicConicIntersection::Coincident
        );
        let h = hyperbola((1.0, 2.0), Vector3D::new(1.0, 1.0, 0.0), 2.0, 1.0);
        assert_eq!(
            h.intersect_exact(&h, tol),
            ConicConicIntersection::Coincident
        );
        // The mirror branch is another curve.
        let mirror = hyperbola((1.0, 2.0), -Vector3D::new(1.0, 1.0, 0.0), 2.0, 1.0);
        assert_eq!(
            h.intersect_exact(&mirror, tol),
            ConicConicIntersection::Empty
        );
        // A circle and the ellipse that is the same circle.
        let c = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
        let round = ellipse((0.0, 0.0), Vector3D::Y, 2.0, 2.0);
        assert_eq!(
            c.intersect_exact(&round, tol),
            ConicConicIntersection::Coincident
        );
        // Distinct parallel planes.
        let lifted = Parabola3D::new(
            Point3D::new(1.0, 2.0, 0.5),
            Vector3D::Z,
            Vector3D::new(1.0, 1.0, 0.0),
            0.5,
        )
        .unwrap();
        assert_eq!(
            p.intersect_exact(&lifted, tol),
            ConicConicIntersection::Empty
        );
    }

    #[test]
    fn test_conics_in_crossing_planes() {
        let tol = Tolerance::DEFAULT;
        let e = ellipse((0.0, 0.0), Vector3D::X, 3.0, 1.5);
        // An upright parabola through (0, ±1.5, 0), on the line the
        // planes share.
        let p = Parabola3D::new(
            Point3D::new(0.0, 0.0, -2.25),
            Vector3D::X,
            Vector3D::Z,
            0.25,
        )
        .unwrap();
        assert_hits(conic_hits(&e, &p), &[(0.0, -1.5, 1), (0.0, 1.5, 1)]);
        // An upright circle touching the ellipse's vertex (3, 0, 0): both
        // run along y there.
        let touching = Circle3D::new(Point3D::new(3.0, 0.0, 1.0), Vector3D::X, 1.0).unwrap();
        assert_hits(conic_hits(&e, &touching), &[(3.0, 0.0, 2)]);
        // Through the same point, but leaving the ellipse's plane: the
        // ellipse touches the circle's plane there, yet crosses the circle.
        let leaving = Circle3D::new(Point3D::new(3.0, 0.6, 0.8), Vector3D::X, 1.0).unwrap();
        assert_hits(conic_hits(&e, &leaving), &[(3.0, 0.0, 1)]);
        // Missing the line the planes share.
        let away = Circle3D::new(Point3D::new(0.0, 0.0, 5.0), Vector3D::X, 1.0).unwrap();
        assert_eq!(e.intersect_exact(&away, tol), ConicConicIntersection::Empty);
    }
}
