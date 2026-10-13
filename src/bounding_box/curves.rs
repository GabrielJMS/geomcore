//! Boxes of curves: closed form for elementary curves, Bézier control
//! points for B-splines.

use super::sealed::BoundRange3D;
use super::{Bound3D, BoundingBox3D, coordinates, point};
use crate::curve_math::bspline as math;
use crate::curves::{
    BSplineCurve3D, Circle3D, Curve3D, ElementaryCurve, Ellipse3D, Hyperbola3D, Line3D, Parabola3D,
    ParametricCurve3D, TrimmedCurve3D,
};
use crate::math::bernstein;
use crate::{Point3D, Vector3D};
use std::f64::consts::{PI, TAU};

/// The box of the curve `center + a cos t + b sin t` (`curve`) over `t` in
/// `range`, or over a whole turn.
///
/// Coordinate `k` is `c_k + A_k cos(t − φ_k)`, `A_k = |(a_k, b_k)|`,
/// `φ_k = atan2(b_k, a_k)`: it peaks at `φ_k` and bottoms out at
/// `φ_k + π`. Over a range it reaches those only where they fall in it,
/// its other extremes being at the ends.
fn trigonometric_box(
    curve: &impl ParametricCurve3D,
    center: Point3D,
    a: Vector3D,
    b: Vector3D,
    range: Option<(f64, f64)>,
) -> BoundingBox3D {
    let (c, a, b) = (coordinates(center), [a.x, a.y, a.z], [b.x, b.y, b.z]);
    let amplitude: [f64; 3] = std::array::from_fn(|k| a[k].hypot(b[k]));
    let Some((first, last)) = range else {
        return BoundingBox3D {
            min: point(std::array::from_fn(|k| c[k] - amplitude[k])),
            max: point(std::array::from_fn(|k| c[k] + amplitude[k])),
        };
    };
    let ends = BoundingBox3D::new(curve.eval_point(first), curve.eval_point(last));
    let (mut lo, mut hi) = (coordinates(ends.min), coordinates(ends.max));
    let reached = |t: f64| first + (t - first).rem_euclid(TAU) <= last;
    for k in 0..3 {
        let peak = b[k].atan2(a[k]);
        if reached(peak) {
            hi[k] = c[k] + amplitude[k];
        }
        if reached(peak + PI) {
            lo[k] = c[k] - amplitude[k];
        }
    }
    BoundingBox3D {
        min: point(lo),
        max: point(hi),
    }
}

/// The box of `curve` over `[first, last]` from its ends and its points at
/// `stationary`, parameters where a coordinate's derivative vanishes
/// (those outside the range or not finite are skipped). Exact when every
/// coordinate has its extremes there.
fn box_through(
    curve: &impl ParametricCurve3D,
    first: f64,
    last: f64,
    stationary: [f64; 3],
) -> BoundingBox3D {
    let ends = BoundingBox3D::new(curve.eval_point(first), curve.eval_point(last));
    stationary
        .into_iter()
        .filter(|&t| first < t && t < last)
        .fold(ends, |b, t| b.including(curve.eval_point(t)))
}

impl Bound3D for Line3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        None
    }
}

impl Bound3D for Circle3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        Some(circle_box(self, None))
    }
}

impl Bound3D for Ellipse3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        Some(ellipse_box(self, None))
    }
}

impl Bound3D for Parabola3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        None
    }
}

impl Bound3D for Hyperbola3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        None
    }
}

impl Bound3D for BSplineCurve3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        BoundingBox3D::from_points(self.poles().iter().copied())
    }
}

impl Bound3D for Curve3D {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        match self {
            Curve3D::Line(c) => c.bounding_box(),
            Curve3D::Circle(c) => c.bounding_box(),
            Curve3D::Ellipse(c) => c.bounding_box(),
            Curve3D::Parabola(c) => c.bounding_box(),
            Curve3D::Hyperbola(c) => c.bounding_box(),
            Curve3D::BSpline(c) => c.bounding_box(),
        }
    }
}

impl Bound3D for ElementaryCurve {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        match self {
            ElementaryCurve::Line(c) => c.bounding_box(),
            ElementaryCurve::Circle(c) => c.bounding_box(),
            ElementaryCurve::Ellipse(c) => c.bounding_box(),
            ElementaryCurve::Parabola(c) => c.bounding_box(),
            ElementaryCurve::Hyperbola(c) => c.bounding_box(),
        }
    }
}

impl<C: BoundRange3D> Bound3D for TrimmedCurve3D<C> {
    fn bounding_box(&self) -> Option<BoundingBox3D> {
        let (first, last) = self.bounds();
        Some(self.basis().range_box(first, last))
    }
}

fn circle_box(circle: &Circle3D, range: Option<(f64, f64)>) -> BoundingBox3D {
    let frame = circle.frame();
    let r = circle.radius();
    trigonometric_box(
        circle,
        frame.origin(),
        r * frame.x_direction(),
        r * frame.y_direction(),
        range,
    )
}

fn ellipse_box(ellipse: &Ellipse3D, range: Option<(f64, f64)>) -> BoundingBox3D {
    let frame = ellipse.frame();
    trigonometric_box(
        ellipse,
        frame.origin(),
        ellipse.major_radius() * frame.x_direction(),
        ellipse.minor_radius() * frame.y_direction(),
        range,
    )
}

impl BoundRange3D for Line3D {
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        BoundingBox3D::new(self.eval_point(first), self.eval_point(last))
    }
}

impl BoundRange3D for Circle3D {
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        circle_box(self, Some((first, last)))
    }
}

impl BoundRange3D for Ellipse3D {
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        ellipse_box(self, Some((first, last)))
    }
}

impl BoundRange3D for Parabola3D {
    /// Coordinate `k` is `o_k + u² x_k / (4 f) + u y_k`, stationary at
    /// `u = −2 f y_k / x_k`.
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        let frame = self.frame();
        let (x, y) = (frame.x_direction(), frame.y_direction());
        let (x, y) = ([x.x, x.y, x.z], [y.x, y.y, y.z]);
        let f = self.focal();
        box_through(
            self,
            first,
            last,
            std::array::from_fn(|k| -2.0 * f * y[k] / x[k]),
        )
    }
}

impl BoundRange3D for Hyperbola3D {
    /// Coordinate `k` is `o_k + a cosh(u) x_k + b sinh(u) y_k`, stationary
    /// where `tanh u = −b y_k / (a x_k)`, if that is within `(−1, 1)`.
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        let frame = self.frame();
        let (x, y) = (frame.x_direction(), frame.y_direction());
        let (x, y) = ([x.x, x.y, x.z], [y.x, y.y, y.z]);
        let (a, b) = (self.major_radius(), self.minor_radius());
        box_through(
            self,
            first,
            last,
            std::array::from_fn(|k| (-(b * y[k]) / (a * x[k])).atanh()),
        )
    }
}

impl BoundRange3D for BSplineCurve3D {
    /// The union, over the knot spans meeting the range, of the boxes of
    /// the poles of the spans it covers and of the Bézier control points
    /// of the pieces of the spans it cuts, restricted to the range (whose
    /// ends are the range's ends). Each piece lies in the convex hull of
    /// either.
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        let degree = self.degree();
        let start_point = self.eval_point(first);
        let mut bounds = BoundingBox3D::new(start_point, start_point);
        let mut points = [[0.0; 4]; math::LARGE];
        // The middles of the first and last spans the range covers; it
        // covers every span between them.
        let mut covered: Option<(f64, f64)> = None;
        self.for_each_span_in(first, last, |(start, end), (a, b)| {
            let middle = 0.5 * (start + end);
            if a == start && b == end {
                covered = Some((covered.map_or(middle, |(m, _)| m), middle));
                return;
            }
            self.span_bezier_points(middle, &mut points);
            let piece = &mut points[..=degree];
            let length = end - start;
            bernstein::restrict(piece, (a - start) / length, (b - start) / length);
            for &[x, y, z, w] in piece.iter() {
                bounds = bounds.including(Point3D::new(x / w, y / w, z / w));
            }
        });
        match covered {
            Some((m0, m1)) => self
                .run_poles(m0, m1)
                .fold(bounds, BoundingBox3D::including),
            None => bounds,
        }
    }
}

impl BoundRange3D for Curve3D {
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        match self {
            Curve3D::Line(c) => c.range_box(first, last),
            Curve3D::Circle(c) => c.range_box(first, last),
            Curve3D::Ellipse(c) => c.range_box(first, last),
            Curve3D::Parabola(c) => c.range_box(first, last),
            Curve3D::Hyperbola(c) => c.range_box(first, last),
            Curve3D::BSpline(c) => c.range_box(first, last),
        }
    }
}

impl BoundRange3D for ElementaryCurve {
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        match self {
            ElementaryCurve::Line(c) => c.range_box(first, last),
            ElementaryCurve::Circle(c) => c.range_box(first, last),
            ElementaryCurve::Ellipse(c) => c.range_box(first, last),
            ElementaryCurve::Parabola(c) => c.range_box(first, last),
            ElementaryCurve::Hyperbola(c) => c.range_box(first, last),
        }
    }
}

impl<C: BoundRange3D + ?Sized> BoundRange3D for &C {
    fn range_box(&self, first: f64, last: f64) -> BoundingBox3D {
        (**self).range_box(first, last)
    }
}
