//! Elementary curves as exact B-splines: circles and ellipses whole or as
//! arcs, and trimmed lines, parabolas and hyperbolas.
//!
//! The geometry is exact (every B-spline point lies on the curve, up to
//! rounding), but a rational quadratic cannot follow an angle or a
//! hyperbolic parameter: the B-spline keeps the curve's range, and its
//! parameter matches the curve's only at the range ends, the knots between
//! pieces and the middle of each piece.

use std::f64::consts::TAU;
use std::fmt;

use crate::curve_math::conic_bspline::{self, PlaneNet};
use crate::curves::{
    BSplineCurve3D, Circle3D, Curve3D, ElementaryCurve, Ellipse3D, Hyperbola3D, Line3D, Parabola3D,
    ParametricCurve3D, TrimmedCurve3D,
};
use crate::{Frame3D, Point3D};

/// Error returned when a curve or surface cannot be converted to a
/// B-spline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BSplineConversionError {
    /// The curve or surface is unbounded: a line, parabola or hyperbola
    /// converts only once trimmed to a range, and a plane, cylinder or cone
    /// only over finite parameter ranges.
    Unbounded,
    /// The B-spline does not fit in `f64`: a pole overflows (a hyperbola
    /// trimmed beyond `|u| ≈ 710`, a parabola beyond `|u| ≈ 1e154`), or the
    /// range ends are so large that its knots round onto each other.
    NotRepresentable,
    /// A surface parameter range is NaN or empty (`first >= last`), longer
    /// than the period of a periodic direction, or leaves the bounds of a
    /// bounded one (a sphere's `v` beyond `±π/2`).
    InvalidRange,
}

impl fmt::Display for BSplineConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            BSplineConversionError::Unbounded => "the curve or surface is unbounded",
            BSplineConversionError::NotRepresentable => "the B-spline does not fit in f64",
            BSplineConversionError::InvalidRange => "the parameter range does not fit the surface",
        };
        f.write_str(message)
    }
}

impl std::error::Error for BSplineConversionError {}

/// Places a plane net in `frame`, its x coordinates scaled by `sx` and its
/// y coordinates by `sy`.
fn place(
    net: PlaneNet,
    frame: Frame3D,
    sx: f64,
    sy: f64,
) -> Result<BSplineCurve3D, BSplineConversionError> {
    let (origin, x, y) = (frame.origin(), frame.x_direction(), frame.y_direction());
    let poles: Vec<Point3D> = net
        .poles
        .iter()
        .map(|&[a, b]| origin + (sx * a) * x + (sy * b) * y)
        .collect();
    if !poles.iter().all(|p| p.is_finite()) || !net.weights.iter().flatten().all(|w| w.is_finite())
    {
        return Err(BSplineConversionError::NotRepresentable);
    }
    let curve = match net.weights {
        Some(weights) => BSplineCurve3D::new_rational(
            net.degree,
            poles,
            weights,
            net.knots,
            net.mults,
            net.periodic,
        ),
        None => BSplineCurve3D::new(net.degree, poles, net.knots, net.mults, net.periodic),
    };
    curve.map_err(|_| BSplineConversionError::NotRepresentable)
}

/// The net of a circle arc over `[first, last]`, ending on its first pole
/// when the range spans a whole turn.
fn arc_net(first: f64, last: f64) -> PlaneNet {
    let mut net = conic_bspline::circle_arc(first, last);
    if last - first >= TAU * (1.0 - 1e-12) {
        net.close();
    }
    net
}

impl Circle3D {
    /// The circle as an exact rational B-spline of degree 2, periodic over
    /// `[0, 2π]` like the circle: four quarter turns, eight poles, double
    /// knots at `0, π/2, π, 3π/2, 2π`.
    ///
    /// Every point of the B-spline lies on the circle, but its parameter is
    /// not the angle: they agree at the knots and at the middle of each
    /// quarter turn, and differ by at most `0.0161` rad in between.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Circle3D, Point3D, Vector3D};
    /// use std::f64::consts::{FRAC_PI_2, TAU};
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let bspline = circle.to_bspline();
    /// assert!(bspline.is_periodic());
    /// assert_eq!(bspline.bounds(), (0.0, TAU));
    /// let p = bspline.eval_point(FRAC_PI_2);
    /// assert!(p.distance(circle.eval_point(FRAC_PI_2)) < 1e-15);
    /// let q = bspline.eval_point(0.3);
    /// assert!((q.distance(Point3D::ORIGIN) - 2.0).abs() < 1e-15);
    /// ```
    pub fn to_bspline(&self) -> BSplineCurve3D {
        let r = self.radius();
        place(conic_bspline::full_circle(), self.frame(), r, r)
            .expect("a finite circle has finite poles")
    }
}

impl Ellipse3D {
    /// The ellipse as an exact rational B-spline of degree 2, periodic over
    /// `[0, 2π]` like the ellipse: the B-spline of the unit circle (see
    /// [`Circle3D::to_bspline`]) scaled by the radii along the frame's axes.
    ///
    /// Every point of the B-spline lies on the ellipse; its parameter
    /// agrees with the ellipse's at the knots `0, π/2, π, 3π/2, 2π` and at
    /// the middle of each quarter, not in between.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Ellipse3D, Point3D, Vector3D};
    ///
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.0).unwrap();
    /// let bspline = ellipse.to_bspline();
    /// assert!(bspline.is_periodic());
    /// let p = bspline.eval_point(1.0);
    /// assert!(((p.x / 3.0).powi(2) + p.y.powi(2) - 1.0).abs() < 1e-15);
    /// ```
    pub fn to_bspline(&self) -> BSplineCurve3D {
        place(
            conic_bspline::full_circle(),
            self.frame(),
            self.major_radius(),
            self.minor_radius(),
        )
        .expect("a finite ellipse has finite poles")
    }
}

pub(crate) mod sealed {
    use super::BSplineConversionError;
    use crate::BSplineCurve3D;

    /// Exact B-spline of a range of a curve: what a trimmed curve needs
    /// from its basis.
    pub trait BSplineRange3D: crate::curves::ParametricCurve3D {
        /// The B-spline over `[first, last]`, a range accepted by
        /// [`TrimmedCurve3D::new`](crate::curves::TrimmedCurve3D::new).
        fn bspline_range(
            &self,
            first: f64,
            last: f64,
        ) -> Result<BSplineCurve3D, BSplineConversionError>;
    }
}
use sealed::BSplineRange3D;

impl BSplineRange3D for Line3D {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        let poles = vec![self.eval_point(first), self.eval_point(last)];
        if !poles.iter().all(|p| p.is_finite()) {
            return Err(BSplineConversionError::NotRepresentable);
        }
        BSplineCurve3D::new(1, poles, vec![first, last], vec![2, 2], false)
            .map_err(|_| BSplineConversionError::NotRepresentable)
    }
}

impl BSplineRange3D for Circle3D {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        let r = self.radius();
        place(arc_net(first, last), self.frame(), r, r)
    }
}

impl BSplineRange3D for Ellipse3D {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        place(
            arc_net(first, last),
            self.frame(),
            self.major_radius(),
            self.minor_radius(),
        )
    }
}

impl BSplineRange3D for Parabola3D {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        let net = conic_bspline::parabola_arc(self.focal(), first, last);
        place(net, self.frame(), 1.0, 1.0)
    }
}

impl BSplineRange3D for Hyperbola3D {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        place(
            conic_bspline::hyperbola_arc(first, last),
            self.frame(),
            self.major_radius(),
            self.minor_radius(),
        )
    }
}

impl BSplineRange3D for ElementaryCurve {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        match self {
            ElementaryCurve::Line(c) => c.bspline_range(first, last),
            ElementaryCurve::Circle(c) => c.bspline_range(first, last),
            ElementaryCurve::Ellipse(c) => c.bspline_range(first, last),
            ElementaryCurve::Parabola(c) => c.bspline_range(first, last),
            ElementaryCurve::Hyperbola(c) => c.bspline_range(first, last),
        }
    }
}

impl<C: BSplineRange3D + ?Sized> BSplineRange3D for &C {
    fn bspline_range(
        &self,
        first: f64,
        last: f64,
    ) -> Result<BSplineCurve3D, BSplineConversionError> {
        (**self).bspline_range(first, last)
    }
}

impl<C: BSplineRange3D> TrimmedCurve3D<C> {
    /// The piece of an elementary curve as an exact B-spline over the same
    /// range `[first, last]`, starting and ending on the same points.
    ///
    /// The basis is a [`Line3D`], [`Circle3D`], [`Ellipse3D`],
    /// [`Parabola3D`], [`Hyperbola3D`] or [`ElementaryCurve`], owned or
    /// borrowed. A segment becomes a B-spline of degree 1 and a parabola
    /// one polynomial quadratic piece, both with the curve's own
    /// parameter. A hyperbola becomes one rational quadratic piece, and a
    /// circle or ellipse arc rational quadratic pieces of equal sweep, at
    /// most a quarter turn each, joined at knots placed at the angles of
    /// the piece boundaries. Their B-spline parameter agrees with the
    /// curve's at the range ends, those knots and the middle of each piece,
    /// and differs in between (by at most `0.0161` rad on a circle). A
    /// range spanning a whole turn gives a clamped B-spline ending exactly
    /// on its first point, never a periodic one, as a trimmed curve is
    /// never periodic.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::NotRepresentable`] when a pole
    /// overflows `f64` (a hyperbola trimmed beyond `|u| ≈ 710`) or the
    /// range ends are so large that knots round onto each other.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::TrimmedCurve3D;
    /// use geomcore::{Circle3D, Point3D, Tolerance, Vector3D};
    /// use std::f64::consts::PI;
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// // Three quarters of the circle, across the seam at angle 0.
    /// let arc = TrimmedCurve3D::new(circle, -PI / 2.0, PI).unwrap();
    /// let bspline = arc.to_bspline().unwrap();
    /// assert_eq!(bspline.bounds(), (-PI / 2.0, PI));
    /// assert_eq!(bspline.knots().len(), 4);
    /// assert!(bspline.eval_point(0.0).distance(Point3D::new(1.0, 0.0, 0.0)) < 1e-15);
    /// assert!(circle.contains(bspline.eval_point(1.0), Tolerance::DEFAULT));
    /// ```
    pub fn to_bspline(&self) -> Result<BSplineCurve3D, BSplineConversionError> {
        let (first, last) = self.bounds();
        self.basis().bspline_range(first, last)
    }
}

impl ElementaryCurve {
    /// The whole curve as an exact B-spline: a circle or ellipse becomes
    /// its periodic B-spline ([`Circle3D::to_bspline`],
    /// [`Ellipse3D::to_bspline`]).
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] for a line, parabola
    /// or hyperbola; trim it first and use [`TrimmedCurve3D::to_bspline`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::{BSplineConversionError, ElementaryCurve};
    /// use geomcore::{Circle3D, Line3D, Point3D, Vector3D};
    ///
    /// let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// assert!(ElementaryCurve::Circle(circle).to_bspline().unwrap().is_periodic());
    /// let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
    /// assert_eq!(
    ///     ElementaryCurve::Line(line).to_bspline().err(),
    ///     Some(BSplineConversionError::Unbounded)
    /// );
    /// ```
    pub fn to_bspline(&self) -> Result<BSplineCurve3D, BSplineConversionError> {
        match self {
            ElementaryCurve::Circle(c) => Ok(c.to_bspline()),
            ElementaryCurve::Ellipse(c) => Ok(c.to_bspline()),
            ElementaryCurve::Line(_)
            | ElementaryCurve::Parabola(_)
            | ElementaryCurve::Hyperbola(_) => Err(BSplineConversionError::Unbounded),
        }
    }
}

impl Curve3D {
    /// The whole curve as a B-spline: a B-spline is cloned, an elementary
    /// curve converted by [`ElementaryCurve::to_bspline`].
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] for a line, parabola
    /// or hyperbola.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::curves::{BSplineConversionError, Curve3D};
    /// use geomcore::{Ellipse3D, Parabola3D, Point3D, Vector3D};
    ///
    /// let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    /// let bspline = Curve3D::from(ellipse).to_bspline().unwrap();
    /// assert_eq!(Curve3D::from(bspline.clone()).to_bspline(), Ok(bspline));
    /// let parabola = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.0).unwrap();
    /// assert_eq!(
    ///     Curve3D::from(parabola).to_bspline().err(),
    ///     Some(BSplineConversionError::Unbounded)
    /// );
    /// ```
    pub fn to_bspline(&self) -> Result<BSplineCurve3D, BSplineConversionError> {
        match self {
            Curve3D::BSpline(c) => Ok(c.clone()),
            _ => self
                .as_elementary()
                .expect("every other curve is elementary")
                .to_bspline(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discretization::{DiscretizationCriteria, Discretize3D};
    use crate::{Tolerance, Vector3D};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const RELATIVE: f64 = 1e-12;

    fn tilted_frame(origin: Point3D) -> Frame3D {
        Frame3D::new(
            origin,
            Vector3D::new(1.0, -2.0, 0.5),
            Vector3D::new(0.3, 1.0, 2.0),
        )
        .unwrap()
    }

    fn circle(radius: f64) -> Circle3D {
        let center = Point3D::new(0.4 * radius, -0.7 * radius, 0.2 * radius);
        Circle3D::from_frame(tilted_frame(center), radius).unwrap()
    }

    fn ellipse() -> Ellipse3D {
        Ellipse3D::from_frame(tilted_frame(Point3D::new(1.0, 2.0, -1.0)), 5.0, 0.5).unwrap()
    }

    fn parabola() -> Parabola3D {
        Parabola3D::from_frame(tilted_frame(Point3D::new(-1.0, 0.5, 2.0)), 0.75).unwrap()
    }

    fn hyperbola() -> Hyperbola3D {
        Hyperbola3D::from_frame(tilted_frame(Point3D::new(2.0, 1.0, 0.0)), 1.5, 0.8).unwrap()
    }

    fn circle_distance(c: &Circle3D, p: Point3D) -> f64 {
        let (x, y, z) = c.frame().local_coordinates(p);
        z.hypot(x.hypot(y) - c.radius())
    }

    /// The largest of `distance` (an upper bound of the distance from a
    /// point to the original curve) over a dense sample of the B-spline,
    /// its knots and the middles of its spans.
    fn max_deviation(b: &BSplineCurve3D, distance: impl Fn(Point3D) -> f64) -> f64 {
        let (first, last) = b.bounds();
        let n = 4000;
        let dense = (0..=n).map(|i| first + (last - first) * (i as f64 / n as f64));
        let knots = b.knots().to_vec();
        let middles: Vec<f64> = knots.windows(2).map(|k| 0.5 * (k[0] + k[1])).collect();
        dense
            .chain(knots)
            .chain(middles)
            .map(|u| distance(b.eval_point(u)))
            .fold(0.0, f64::max)
    }

    /// The B-spline and the curve agree at the knots and the middle of
    /// every span.
    fn assert_matches_at_knots_and_middles<C: ParametricCurve3D>(
        b: &BSplineCurve3D,
        curve: &C,
        scale: f64,
    ) {
        let knots = b.knots();
        let middles = knots.windows(2).map(|k| 0.5 * (k[0] + k[1]));
        for u in knots.iter().copied().chain(middles) {
            let d = b.eval_point(u).distance(curve.eval_point(u));
            assert!(d <= 1e-15 * scale, "u = {u}: {d}");
        }
    }

    const ARCS: [(f64, f64); 9] = [
        (0.0, 1e-9),
        (0.3, 0.300_01),
        (0.0, FRAC_PI_2),
        (-1.0, 1.0),
        (5.5, 7.0),
        (-5.0, -4.0),
        (1.0, 4.0),
        (-PI, 2.5),
        (0.1, 0.1 + 1.999 * PI),
    ];

    // ---- circles and ellipses ----

    #[test]
    fn test_full_circle_is_periodic_and_exact() {
        for radius in [1e-6, 2.5, 1e6] {
            let c = circle(radius);
            let b = c.to_bspline();
            assert!(b.is_periodic() && b.is_rational());
            assert_eq!(b.degree(), 2);
            assert_eq!(b.bounds(), (0.0, TAU));
            assert_eq!(b.poles().len(), 8);
            assert_eq!(b.period(), Some(TAU));
            let dev = max_deviation(&b, |p| circle_distance(&c, p));
            assert!(dev <= RELATIVE * radius, "radius {radius}: {dev}");
            assert_matches_at_knots_and_middles(&b, &c, radius);
            assert_eq!(b.eval_point(0.0), c.eval_point(0.0));
            let p = b.eval_point(1.0);
            assert!(b.eval_point(1.0 + TAU).distance(p) <= 1e-15 * radius);
            assert!(b.eval_point(1.0 - TAU).distance(p) <= 1e-15 * radius);
        }
    }

    #[test]
    fn test_full_circle_parameter_stays_near_the_angle() {
        let c = circle(1.0);
        let b = c.to_bspline();
        let worst = (0..=1000)
            .map(|i| TAU * i as f64 / 1000.0)
            .map(|u| b.eval_point(u).distance(c.eval_point(u)))
            .fold(0.0, f64::max);
        // A chord of 0.0161 rad, the largest gap between parameter and angle.
        assert!(worst > 0.01 && worst < 0.0162, "{worst}");
    }

    #[test]
    fn test_full_ellipse_is_periodic_and_exact() {
        let e = ellipse();
        let b = e.to_bspline();
        assert!(b.is_periodic());
        assert_eq!(b.bounds(), (0.0, TAU));
        let dev = max_deviation(&b, |p| p.distance(e.eval_point(e.parameter_of(p))));
        assert!(dev <= RELATIVE * 5.0, "{dev}");
        assert_matches_at_knots_and_middles(&b, &e, 5.0);
    }

    #[test]
    fn test_circle_arcs_are_exact_over_their_range() {
        for radius in [1e-6, 1.0, 1e6] {
            let c = circle(radius);
            for (first, last) in ARCS {
                let arc = TrimmedCurve3D::new(c, first, last).unwrap();
                let b = arc.to_bspline().unwrap();
                assert!(!b.is_periodic());
                assert_eq!(b.bounds(), (first, last));
                let pieces = b.knots().len() - 1;
                assert!((last - first) / pieces as f64 <= FRAC_PI_2 * (1.0 + 1e-12));
                let dev = max_deviation(&b, |p| circle_distance(&c, p));
                assert!(
                    dev <= RELATIVE * radius,
                    "{radius} [{first}, {last}]: {dev}"
                );
                assert_matches_at_knots_and_middles(&b, &c, radius);
                assert_eq!(b.eval_point(first), arc.start_point());
                assert!(b.eval_point(last).distance(arc.end_point()) <= 1e-16 * radius);
            }
        }
    }

    #[test]
    fn test_ellipse_arcs_are_exact_over_their_range() {
        let e = ellipse();
        for (first, last) in ARCS {
            let arc = TrimmedCurve3D::new(e, first, last).unwrap();
            let b = arc.to_bspline().unwrap();
            assert_eq!(b.bounds(), (first, last));
            let dev = max_deviation(&b, |p| p.distance(e.eval_point(e.parameter_of(p))));
            assert!(dev <= RELATIVE * 5.0, "[{first}, {last}]: {dev}");
            assert_matches_at_knots_and_middles(&b, &e, 5.0);
            assert_eq!(b.eval_point(first), arc.start_point());
            assert!(b.eval_point(last).distance(arc.end_point()) <= 1e-15);
        }
    }

    #[test]
    fn test_whole_turn_arc_is_clamped_and_closed() {
        let c = circle(2.0);
        for first in [0.0, -1.0, 3.0] {
            let arc = TrimmedCurve3D::new(c, first, first + TAU).unwrap();
            let b = arc.to_bspline().unwrap();
            assert!(!b.is_periodic());
            assert_eq!(b.bounds(), (first, first + TAU));
            assert_eq!(b.knots().len(), 5);
            assert_eq!(b.poles().first(), b.poles().last());
            assert_eq!(b.eval_point(first), b.eval_point(first + TAU));
            assert!(max_deviation(&b, |p| circle_distance(&c, p)) <= RELATIVE * 2.0);
        }
        let e = TrimmedCurve3D::new(ellipse(), 1.0, 1.0 + TAU).unwrap();
        let b = e.to_bspline().unwrap();
        assert_eq!(b.poles().first(), b.poles().last());
    }

    #[test]
    fn test_quarter_turn_pieces() {
        let arc = TrimmedCurve3D::new(circle(1.0), 0.0, PI).unwrap();
        let b = arc.to_bspline().unwrap();
        assert_eq!(b.knots(), &[0.0, FRAC_PI_2, PI]);
        assert_eq!(b.multiplicities(), &[3, 2, 3]);
        let w = b.weights().unwrap();
        assert!((w[1] - FRAC_PI_4.cos()).abs() < 1e-15);
        assert_eq!((w[0], w[2], w[4]), (1.0, 1.0, 1.0));
    }

    // ---- lines, parabolas, hyperbolas ----

    #[test]
    fn test_segment_keeps_the_line_parameter() {
        let line = Line3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::new(1.0, -1.0, 2.0)).unwrap();
        let segment = TrimmedCurve3D::new(line, -2.0, 3.5).unwrap();
        let b = segment.to_bspline().unwrap();
        assert_eq!((b.degree(), b.is_rational()), (1, false));
        assert_eq!(b.bounds(), (-2.0, 3.5));
        assert_eq!(b.eval_point(-2.0), segment.start_point());
        assert_eq!(b.eval_point(3.5), segment.end_point());
        for i in 0..=50 {
            let u = -2.0 + 5.5 * i as f64 / 50.0;
            assert!(b.eval_point(u).distance(line.eval_point(u)) <= 1e-14);
        }
    }

    #[test]
    fn test_parabola_is_one_polynomial_piece_with_its_parameter() {
        let p = parabola();
        for (first, last) in [(-3.0, 5.0), (0.2, 0.200_001), (-1e3, 1e3), (-7.0, -6.0)] {
            let arc = TrimmedCurve3D::new(p, first, last).unwrap();
            let b = arc.to_bspline().unwrap();
            assert_eq!((b.degree(), b.is_rational()), (2, false));
            assert_eq!(b.poles().len(), 3);
            assert_eq!(b.bounds(), (first, last));
            assert_eq!(b.eval_point(first), arc.start_point());
            let scale = (last * last / 3.0)
                .max(last.abs())
                .max(first.abs())
                .max(1.0);
            for i in 0..=200 {
                let u = first + (last - first) * i as f64 / 200.0;
                let d = b.eval_point(u).distance(p.eval_point(u));
                assert!(d <= RELATIVE * scale, "[{first}, {last}] at {u}: {d}");
            }
        }
    }

    #[test]
    fn test_hyperbola_is_one_rational_piece_on_the_branch() {
        let h = hyperbola();
        for (first, last) in [(-3.0, 2.0), (0.1, 0.100_000_1), (-0.5, 0.5), (4.0, 6.0)] {
            let arc = TrimmedCurve3D::new(h, first, last).unwrap();
            let b = arc.to_bspline().unwrap();
            assert_eq!((b.degree(), b.poles().len()), (2, 3));
            assert_eq!(b.bounds(), (first, last));
            let scale = 1.5 * first.abs().max(last.abs()).cosh();
            let dev = max_deviation(&b, |p| p.distance(h.eval_point(h.parameter_of(p))));
            assert!(dev <= RELATIVE * scale, "[{first}, {last}]: {dev}");
            assert_matches_at_knots_and_middles(&b, &h, scale);
            assert_eq!(b.eval_point(first), arc.start_point());
        }
    }

    #[test]
    fn test_hyperbola_far_out_is_not_representable() {
        let arc = TrimmedCurve3D::new(hyperbola(), 700.0, 720.0).unwrap();
        assert_eq!(
            arc.to_bspline().err(),
            Some(BSplineConversionError::NotRepresentable)
        );
    }

    #[test]
    fn test_arc_whose_knots_collapse_is_not_representable() {
        // At 2^54 the parameters are 4 apart: three pieces cannot fit.
        let first = 2f64.powi(54);
        let arc = TrimmedCurve3D::new(circle(1.0), first, first + 4.0).unwrap();
        assert_eq!(
            arc.to_bspline().err(),
            Some(BSplineConversionError::NotRepresentable)
        );
    }

    // ---- run-time and borrowed curves ----

    #[test]
    fn test_unbounded_curves_are_rejected() {
        let line = Line3D::new(Point3D::ORIGIN, Vector3D::X).unwrap();
        let unbounded: [ElementaryCurve; 3] = [line.into(), parabola().into(), hyperbola().into()];
        for curve in unbounded {
            let err = Some(BSplineConversionError::Unbounded);
            assert_eq!(curve.to_bspline().err(), err);
            assert_eq!(Curve3D::from(curve).to_bspline().err(), err);
        }
    }

    #[test]
    fn test_run_time_curves_convert_like_the_concrete_ones() {
        let c = circle(3.0);
        assert_eq!(ElementaryCurve::from(c).to_bspline(), Ok(c.to_bspline()));
        assert_eq!(Curve3D::from(c).to_bspline(), Ok(c.to_bspline()));
        let e = ellipse();
        assert_eq!(Curve3D::from(e).to_bspline(), Ok(e.to_bspline()));
        let b = e.to_bspline();
        assert_eq!(Curve3D::from(b.clone()).to_bspline(), Ok(b));

        let h = hyperbola();
        let concrete = TrimmedCurve3D::new(h, -1.0, 2.0).unwrap().to_bspline();
        let run_time = TrimmedCurve3D::new(ElementaryCurve::from(h), -1.0, 2.0).unwrap();
        assert_eq!(run_time.to_bspline(), concrete);
        let borrowed = TrimmedCurve3D::new(&h, -1.0, 2.0).unwrap();
        assert_eq!(borrowed.to_bspline(), concrete);
    }

    #[test]
    fn test_messages() {
        assert_eq!(
            BSplineConversionError::Unbounded.to_string(),
            "the curve or surface is unbounded"
        );
        assert_eq!(
            BSplineConversionError::NotRepresentable.to_string(),
            "the B-spline does not fit in f64"
        );
        assert_eq!(
            BSplineConversionError::InvalidRange.to_string(),
            "the parameter range does not fit the surface"
        );
    }

    // ---- the B-splines work with the B-spline algorithms ----

    #[test]
    fn test_converted_circle_projects_and_discretizes_like_the_circle() {
        let tol = Tolerance::DEFAULT;
        let c = circle(2.0);
        let b = c.to_bspline();
        let q = c.eval_point(2.0) + 0.5 * c.frame().z_direction();
        let expected = c.project_point(q, tol).distance;
        assert!((b.project_point(q, tol).distance - expected).abs() < 1e-9);
        assert!(b.contains(c.eval_point(4.0), tol));

        let criteria = DiscretizationCriteria::new(1e-3).unwrap();
        let d = b.discretize(criteria);
        assert_eq!(d.points.first(), d.points.last());
        assert!(d.chord_deviations.iter().all(|&dev| dev <= 1e-3));
        assert!(d.points.iter().all(|&p| circle_distance(&c, p) <= 1e-12));

        let arc = TrimmedCurve3D::new(b, 5.0, 7.5).unwrap();
        assert!(arc.contains(c.eval_point(0.0), tol));
    }
}
