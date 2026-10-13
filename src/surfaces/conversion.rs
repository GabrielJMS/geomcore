//! Elementary surfaces as exact B-spline surfaces over parameter ranges.
//!
//! A plane patch is bilinear. The other surfaces turn a profile about their
//! axis, so each is a tensor product of two nets: the unit circle in `u`
//! (rational quadratic pieces of at most a quarter turn, from
//! [`conic_bspline`]) and a profile in `v`, a segment for cylinders and
//! cones and a circle arc for spheres and tori. Pole `(i, j)` is profile
//! pole `j` turned to the angle of circle pole `i`, weighted by the product
//! of their weights. A profile pole on the axis (a cone's apex, a sphere's
//! pole) turns into a row of coincident poles: a collapsed edge.
//!
//! The geometry is exact (every B-spline point lies on the surface, up to
//! rounding). Along a straight direction the B-spline keeps the surface's
//! parameter; along a circular one it matches the angle only at the range
//! ends, the knots between pieces and the middle of each piece.

use std::f64::consts::TAU;

use crate::curve_math::conic_bspline::{self, PlaneNet};
use crate::curves::BSplineConversionError;
use crate::surfaces::{
    BSplineSurface, Cone, Cylinder, ElementarySurface, ParametricSurface, Plane, Sphere, Surface,
    Torus,
};
use crate::{Frame3D, Point3D};

/// Validates `[first, last]` against a direction with the given bounds and
/// period. A periodic direction accepts any start, so a range may cross its
/// seam, but no more than one period.
fn check_range(
    (first, last): (f64, f64),
    bounds: (f64, f64),
    period: Option<f64>,
) -> Result<(), BSplineConversionError> {
    if first.is_nan() || last.is_nan() {
        return Err(BSplineConversionError::InvalidRange);
    }
    if first.is_infinite() || last.is_infinite() {
        return Err(BSplineConversionError::Unbounded);
    }
    if first >= last {
        return Err(BSplineConversionError::InvalidRange);
    }
    match period {
        Some(period) if last - first > period * (1.0 + 1e-12) => {
            Err(BSplineConversionError::InvalidRange)
        }
        Some(_) => Ok(()),
        None if first < bounds.0 || last > bounds.1 => Err(BSplineConversionError::InvalidRange),
        None => Ok(()),
    }
}

fn check_ranges<S: ParametricSurface>(
    surface: &S,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> Result<(), BSplineConversionError> {
    check_range(u_range, surface.u_bounds(), surface.u_period())?;
    check_range(v_range, surface.v_bounds(), surface.v_period())
}

/// The unit circle over the angles `[first, last]`, periodic when the range
/// spans a whole turn.
fn angle_net(first: f64, last: f64) -> PlaneNet {
    let mut net = conic_bspline::circle_arc(first, last);
    // A whole turn computed with rounding errors is still a whole turn.
    if last - first >= TAU * (1.0 - 1e-12) {
        net.close();
        net.make_periodic();
    }
    net
}

/// The straight profile from `start` to `end` over `[first, last]`, with
/// the surface's own parameter.
fn segment(start: [f64; 2], end: [f64; 2], (first, last): (f64, f64)) -> PlaneNet {
    PlaneNet {
        degree: 1,
        poles: vec![start, end],
        weights: None,
        knots: vec![first, last],
        multiplicities: vec![2, 2],
        periodic: false,
    }
}

/// `radius`, or zero when it is within rounding of zero next to `scale`,
/// the size of the terms it was computed from: a profile pole meant to be
/// on the axis lands on it exactly, and its row of poles collapses.
fn on_axis(radius: f64, scale: f64) -> f64 {
    if radius.abs() <= 4.0 * f64::EPSILON * scale {
        0.0
    } else {
        radius
    }
}

/// The poles of `profile` (`[radius, height]` in a half-plane through the
/// axis) turned about the z axis of `frame` to the angle of each pole of
/// the unit `circle`: `poles[i][j]` for circle pole `i` and profile pole
/// `j`.
fn revolved_poles(frame: Frame3D, circle: &PlaneNet, profile: &PlaneNet) -> Vec<Vec<Point3D>> {
    let (origin, x, y, z) = (
        frame.origin(),
        frame.x_direction(),
        frame.y_direction(),
        frame.z_direction(),
    );
    circle
        .poles
        .iter()
        .map(|&[cos, sin]| {
            profile
                .poles
                .iter()
                .map(|&[radius, height]| {
                    origin + (radius * cos) * x + (radius * sin) * y + height * z
                })
                .collect()
        })
        .collect()
}

/// The surface turning `profile` about the z axis of `frame` over the
/// angles `u_range`.
fn revolve(
    frame: Frame3D,
    (first, last): (f64, f64),
    profile: PlaneNet,
) -> Result<BSplineSurface, BSplineConversionError> {
    let circle = angle_net(first, last);
    let poles = revolved_poles(frame, &circle, &profile);
    if !poles.iter().flatten().all(|p| p.is_finite()) {
        return Err(BSplineConversionError::NotRepresentable);
    }
    let weights = circle
        .weights
        .as_ref()
        .expect("circle nets are rational")
        .iter()
        .map(|&u_weight| match &profile.weights {
            Some(v_weights) => v_weights.iter().map(|&w| u_weight * w).collect(),
            None => vec![u_weight; profile.poles.len()],
        })
        .collect();
    BSplineSurface::new_rational(
        circle.degree,
        profile.degree,
        poles,
        weights,
        circle.knots,
        circle.multiplicities,
        profile.knots,
        profile.multiplicities,
        circle.periodic,
        profile.periodic,
    )
    .map_err(|_| BSplineConversionError::NotRepresentable)
}

impl Plane {
    /// The patch `u_range × v_range` of the plane as a bilinear B-spline
    /// surface: degree 1 in both directions, its four poles the plane's
    /// points at the corners. It keeps the plane's parameters.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] when a range end is
    /// infinite, [`BSplineConversionError::InvalidRange`] when a range is
    /// NaN or empty, and [`BSplineConversionError::NotRepresentable`] when
    /// a corner overflows `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Plane, Point3D, Vector3D};
    ///
    /// let plane = Plane::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z).unwrap();
    /// let patch = plane.to_bspline_patch((-1.0, 2.0), (0.0, 3.0)).unwrap();
    /// assert_eq!((patch.u_degree(), patch.v_degree()), (1, 1));
    /// assert_eq!(patch.eval_point(0.5, 2.0), plane.eval_point(0.5, 2.0));
    /// ```
    pub fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Result<BSplineSurface, BSplineConversionError> {
        check_ranges(self, u_range, v_range)?;
        let ((u0, u1), (v0, v1)) = (u_range, v_range);
        let poles = vec![
            vec![self.eval_point(u0, v0), self.eval_point(u0, v1)],
            vec![self.eval_point(u1, v0), self.eval_point(u1, v1)],
        ];
        if !poles.iter().flatten().all(|p| p.is_finite()) {
            return Err(BSplineConversionError::NotRepresentable);
        }
        BSplineSurface::new(
            1,
            1,
            poles,
            vec![u0, u1],
            vec![2, 2],
            vec![v0, v1],
            vec![2, 2],
            false,
            false,
        )
        .map_err(|_| BSplineConversionError::NotRepresentable)
    }
}

impl Cylinder {
    /// The straight generatrix over `v_range`, at the cylinder's radius.
    fn profile(&self, (first, last): (f64, f64)) -> PlaneNet {
        let r = self.radius();
        segment([r, first], [r, last], (first, last))
    }

    /// The patch `u_range × v_range` of the cylinder as an exact rational
    /// B-spline surface, of degree 2 in `u` and 1 in `v`.
    ///
    /// Along `u` the B-spline is made of the circle's rational quadratic
    /// pieces of equal sweep, at most a quarter turn each, joined at knots
    /// placed at the angles of the piece boundaries; a `u_range` spanning a
    /// whole turn, from any start, gives a periodic `u` direction. Every
    /// point of the B-spline lies on the cylinder; its `u` parameter
    /// agrees with the angle at the range ends, those knots and the middle
    /// of each piece, and strays at most `0.0161` rad in between. Its `v`
    /// parameter is the cylinder's.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] when a range end is
    /// infinite, [`BSplineConversionError::InvalidRange`] when a range is
    /// NaN or empty or `u_range` is longer than a turn, and
    /// [`BSplineConversionError::NotRepresentable`] when a pole overflows
    /// `f64` or the range ends are so large that knots round onto each
    /// other.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineConversionError, Cylinder, Point3D, Vector3D};
    /// use std::f64::consts::{FRAC_PI_2, TAU};
    ///
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    /// let tube = cylinder.to_bspline_patch((0.0, TAU), (-1.0, 4.0)).unwrap();
    /// assert!(tube.is_u_periodic() && !tube.is_v_periodic());
    /// let p = tube.eval_point(FRAC_PI_2, 3.0);
    /// assert!(p.distance(cylinder.eval_point(FRAC_PI_2, 3.0)) < 1e-15);
    /// assert_eq!(
    ///     cylinder.to_bspline_patch((0.0, 1.0), (0.0, f64::INFINITY)).err(),
    ///     Some(BSplineConversionError::Unbounded)
    /// );
    /// ```
    pub fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Result<BSplineSurface, BSplineConversionError> {
        check_ranges(self, u_range, v_range)?;
        revolve(self.frame(), u_range, self.profile(v_range))
    }
}

impl Cone {
    /// The straight generatrix over `v_range`, ending exactly on the axis
    /// at the apex.
    fn profile(&self, (first, last): (f64, f64)) -> PlaneNet {
        let (ref_r, alpha) = (self.ref_radius(), self.semi_angle());
        let pole = |v: f64| {
            let radial = v * alpha.sin();
            [
                on_axis(ref_r + radial, ref_r.abs() + radial.abs()),
                v * alpha.cos(),
            ]
        };
        segment(pole(first), pole(last), (first, last))
    }

    /// The patch `u_range × v_range` of the cone as an exact rational
    /// B-spline surface, of degree 2 in `u` and 1 in `v`.
    ///
    /// Built like [`Cylinder::to_bspline_patch`]: along `u` the circle's
    /// rational quadratic pieces of at most a quarter turn (periodic for a
    /// whole turn), whose parameter agrees with the angle at the range
    /// ends, the knots and the middle of each piece; along `v` the cone's
    /// own parameter. A `v_range` ending at the apex gives a collapsed
    /// edge, all its poles at the apex; one crossing it covers both
    /// nappes.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] when a range end is
    /// infinite, [`BSplineConversionError::InvalidRange`] when a range is
    /// NaN or empty or `u_range` is longer than a turn, and
    /// [`BSplineConversionError::NotRepresentable`] when a pole overflows
    /// `f64` or the range ends are so large that knots round onto each
    /// other.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Cone, Point3D, Vector3D};
    /// use std::f64::consts::PI;
    ///
    /// let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, PI / 6.0, 1.0).unwrap();
    /// // From the apex (v = -ref_radius / sin(semi_angle) = -2) up to v = 1.
    /// let patch = cone.to_bspline_patch((0.0, PI), (-2.0, 1.0)).unwrap();
    /// let apex = cone.eval_point(0.0, -2.0);
    /// assert!(patch.eval_point(1.0, -2.0).distance(apex) < 1e-15);
    /// assert!(patch.eval_point(2.5, -2.0).distance(apex) < 1e-15);
    /// ```
    pub fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Result<BSplineSurface, BSplineConversionError> {
        check_ranges(self, u_range, v_range)?;
        revolve(self.frame(), u_range, self.profile(v_range))
    }
}

impl Sphere {
    /// The meridian over the latitudes `v_range`, its end poles exactly on
    /// the axis at the sphere's poles.
    fn profile(&self, (first, last): (f64, f64)) -> PlaneNet {
        let r = self.radius();
        let mut meridian = conic_bspline::circle_arc(first, last);
        for pole in &mut meridian.poles {
            *pole = [r * on_axis(pole[0], 1.0), r * pole[1]];
        }
        meridian
    }

    /// The whole sphere as an exact rational B-spline surface of degree 2
    /// in both directions: periodic in `u` over `[0, 2π]` (four quarter
    /// turns), clamped in `v` over `[-π/2, π/2]` (two quarter turns), its
    /// first and last rows of poles collapsed onto the sphere's poles. See
    /// [`Sphere::to_bspline_patch`].
    ///
    /// # Panics
    ///
    /// Panics if a pole overflows `f64`, for a radius or center beyond
    /// about `1e307`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// use std::f64::consts::{FRAC_PI_2, TAU};
    ///
    /// let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    /// let bspline = sphere.to_bspline();
    /// assert!(bspline.is_u_periodic() && !bspline.is_v_periodic());
    /// let north = bspline.eval_point(1.0, FRAC_PI_2);
    /// assert!(north.distance(Point3D::new(0.0, 0.0, 2.0)) < 1e-15);
    /// let p = bspline.eval_point(0.3, 0.7);
    /// assert!((p.distance(Point3D::ORIGIN) - 2.0).abs() < 1e-15);
    /// ```
    pub fn to_bspline(&self) -> BSplineSurface {
        self.to_bspline_patch(self.u_bounds(), self.v_bounds())
            .expect("a sphere of finite poles converts")
    }

    /// The patch `u_range × v_range` of the sphere as an exact rational
    /// B-spline surface of degree 2 in both directions.
    ///
    /// Each direction is made of circle arcs: rational quadratic pieces of
    /// equal sweep, at most a quarter turn each, joined at knots placed at
    /// the angles of the piece boundaries. Every point of the B-spline lies
    /// on the sphere; its parameters agree with the sphere's at the range
    /// ends, those knots and the middle of each piece, and stray at most
    /// `0.0161` rad in between. A `u_range` spanning a whole turn, from any
    /// start, gives a periodic `u` direction; a `v_range` ending at `±π/2`
    /// gives a collapsed edge, all its poles on the sphere's pole.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] when a range end is
    /// infinite, [`BSplineConversionError::InvalidRange`] when a range is
    /// NaN or empty, `u_range` is longer than a turn or `v_range` leaves
    /// `[-π/2, π/2]`, and [`BSplineConversionError::NotRepresentable`] when
    /// a pole overflows `f64` or the range ends are so large that knots
    /// round onto each other.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Sphere};
    /// use std::f64::consts::{FRAC_PI_2, PI};
    ///
    /// let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    /// // The northern half of the sphere's eastern half.
    /// let patch = sphere.to_bspline_patch((-FRAC_PI_2, FRAC_PI_2), (0.0, FRAC_PI_2)).unwrap();
    /// assert!(!patch.is_u_periodic());
    /// assert!(patch.eval_point(0.0, 0.0).distance(Point3D::new(1.0, 0.0, 0.0)) < 1e-15);
    /// assert!((patch.eval_point(0.4, 1.1).distance(Point3D::ORIGIN) - 1.0).abs() < 1e-15);
    /// assert!(sphere.to_bspline_patch((0.0, PI), (0.0, 2.0)).is_err());
    /// ```
    pub fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Result<BSplineSurface, BSplineConversionError> {
        check_ranges(self, u_range, v_range)?;
        revolve(self.frame(), u_range, self.profile(v_range))
    }
}

impl Torus {
    /// The tube circle over `v_range` (periodic for a whole turn), a pole
    /// meant to be on the axis landing on it exactly.
    fn profile(&self, (first, last): (f64, f64)) -> PlaneNet {
        let (major, minor) = (self.major_radius(), self.minor_radius());
        let mut tube = angle_net(first, last);
        for pole in &mut tube.poles {
            let radial = minor * pole[0];
            *pole = [
                on_axis(major + radial, major + radial.abs()),
                minor * pole[1],
            ];
        }
        tube
    }

    /// The whole torus as an exact rational B-spline surface of degree 2,
    /// periodic in both directions over `[0, 2π]`: four quarter turns each
    /// way, eight by eight poles. See [`Torus::to_bspline_patch`].
    ///
    /// # Panics
    ///
    /// Panics if a pole overflows `f64`, for radii or a center beyond about
    /// `1e307`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    ///
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    /// let bspline = torus.to_bspline();
    /// assert!(bspline.is_u_periodic() && bspline.is_v_periodic());
    /// assert!(bspline.eval_point(0.0, 0.0).distance(Point3D::new(4.0, 0.0, 0.0)) < 1e-15);
    /// ```
    pub fn to_bspline(&self) -> BSplineSurface {
        self.to_bspline_patch(self.u_bounds(), self.v_bounds())
            .expect("a torus of finite poles converts")
    }

    /// The patch `u_range × v_range` of the torus as an exact rational
    /// B-spline surface of degree 2 in both directions.
    ///
    /// Each direction is made of circle arcs, built and parametrised like
    /// [`Sphere::to_bspline_patch`]'s; a range spanning a whole turn, from
    /// any start, gives a periodic direction. On a torus whose tube meets
    /// the axis (major radius at most the minor one), a `v_range` ending on
    /// the axis gives a collapsed edge.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] when a range end is
    /// infinite, [`BSplineConversionError::InvalidRange`] when a range is
    /// NaN, empty or longer than a turn, and
    /// [`BSplineConversionError::NotRepresentable`] when a pole overflows
    /// `f64` or the range ends are so large that knots round onto each
    /// other.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{Point3D, Torus, Vector3D};
    /// use std::f64::consts::{PI, TAU};
    ///
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    /// // The outer half of the tube, all the way round.
    /// let band = torus.to_bspline_patch((0.0, TAU), (-PI / 2.0, PI / 2.0)).unwrap();
    /// assert!(band.is_u_periodic() && !band.is_v_periodic());
    /// let p = band.eval_point(PI, 0.0);
    /// assert!(p.distance(Point3D::new(-4.0, 0.0, 0.0)) < 1e-15);
    /// ```
    pub fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Result<BSplineSurface, BSplineConversionError> {
        check_ranges(self, u_range, v_range)?;
        revolve(self.frame(), u_range, self.profile(v_range))
    }
}

impl ElementarySurface {
    /// The whole surface as an exact B-spline: a sphere or torus becomes
    /// its B-spline ([`Sphere::to_bspline`], [`Torus::to_bspline`]).
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] for a plane, cylinder
    /// or cone; convert a patch of it with
    /// [`to_bspline_patch`](Self::to_bspline_patch) instead. Returns
    /// [`BSplineConversionError::NotRepresentable`] when a pole overflows
    /// `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ElementarySurface;
    /// use geomcore::{BSplineConversionError, Plane, Point3D, Sphere, Vector3D};
    ///
    /// let sphere = Sphere::new(Point3D::ORIGIN, 1.0).unwrap();
    /// assert!(ElementarySurface::Sphere(sphere).to_bspline().unwrap().is_u_periodic());
    /// let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    /// assert_eq!(
    ///     ElementarySurface::Plane(plane).to_bspline().err(),
    ///     Some(BSplineConversionError::Unbounded)
    /// );
    /// ```
    pub fn to_bspline(&self) -> Result<BSplineSurface, BSplineConversionError> {
        match self {
            ElementarySurface::Sphere(s) => s.to_bspline_patch(s.u_bounds(), s.v_bounds()),
            ElementarySurface::Torus(s) => s.to_bspline_patch(s.u_bounds(), s.v_bounds()),
            ElementarySurface::Plane(_)
            | ElementarySurface::Cylinder(_)
            | ElementarySurface::Cone(_) => Err(BSplineConversionError::Unbounded),
        }
    }

    /// The patch `u_range × v_range` of the surface as an exact B-spline,
    /// by [`Plane::to_bspline_patch`], [`Cylinder::to_bspline_patch`],
    /// [`Cone::to_bspline_patch`], [`Sphere::to_bspline_patch`] or
    /// [`Torus::to_bspline_patch`].
    ///
    /// # Errors
    ///
    /// Returns the error of the surface's own conversion.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::surfaces::ElementarySurface;
    /// use geomcore::{Cylinder, Point3D, Vector3D};
    ///
    /// let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    /// let surface = ElementarySurface::Cylinder(cylinder);
    /// assert_eq!(
    ///     surface.to_bspline_patch((0.0, 1.0), (0.0, 2.0)),
    ///     cylinder.to_bspline_patch((0.0, 1.0), (0.0, 2.0))
    /// );
    /// ```
    pub fn to_bspline_patch(
        &self,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Result<BSplineSurface, BSplineConversionError> {
        match self {
            ElementarySurface::Plane(s) => s.to_bspline_patch(u_range, v_range),
            ElementarySurface::Cylinder(s) => s.to_bspline_patch(u_range, v_range),
            ElementarySurface::Cone(s) => s.to_bspline_patch(u_range, v_range),
            ElementarySurface::Sphere(s) => s.to_bspline_patch(u_range, v_range),
            ElementarySurface::Torus(s) => s.to_bspline_patch(u_range, v_range),
        }
    }
}

impl Surface {
    /// The whole surface as a B-spline: a B-spline is cloned, an elementary
    /// surface converted by [`ElementarySurface::to_bspline`].
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConversionError::Unbounded`] for a plane, cylinder
    /// or cone.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineConversionError, Cone, Point3D, Surface, Torus, Vector3D};
    ///
    /// let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 3.0, 1.0).unwrap();
    /// let bspline = Surface::from(torus).to_bspline().unwrap();
    /// assert_eq!(Surface::from(bspline.clone()).to_bspline(), Ok(bspline));
    /// let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.5, 1.0).unwrap();
    /// assert_eq!(
    ///     Surface::from(cone).to_bspline().err(),
    ///     Some(BSplineConversionError::Unbounded)
    /// );
    /// ```
    pub fn to_bspline(&self) -> Result<BSplineSurface, BSplineConversionError> {
        match self {
            Surface::BSpline(s) => Ok(s.clone()),
            _ => self
                .as_elementary()
                .expect("every other surface is elementary")
                .to_bspline(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curves::TrimmedCurve3D;
    use crate::{Circle3D, Tolerance, Vector3D};
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

    /// Ranges of angles: tiny, across the seam, negative, up to just short
    /// of a turn, and a whole turn.
    const ANGLES: [(f64, f64); 9] = [
        (0.0, 1e-9),
        (0.3, 0.300_01),
        (0.0, FRAC_PI_2),
        (-1.0, 1.0),
        (5.5, 7.0),
        (-5.0, -4.0),
        (-PI, 2.5),
        (0.1, 0.1 + 1.999 * PI),
        (-1.0, -1.0 + TAU),
    ];

    /// How a B-spline direction maps to the surface's: the identity along
    /// a straight direction, the angle of the unit circle's B-spline
    /// (#37's, with the same knots) along a circular one.
    enum Direction {
        Straight,
        Circular,
    }

    impl Direction {
        fn map(&self, (first, last): (f64, f64)) -> impl Fn(f64) -> f64 {
            let circle = match self {
                Direction::Straight => None,
                Direction::Circular => {
                    let unit = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
                    Some(
                        TrimmedCurve3D::new(unit, first, last)
                            .unwrap()
                            .to_bspline()
                            .unwrap(),
                    )
                }
            };
            move |t| match &circle {
                None => t,
                Some(c) => {
                    let p = c.eval_point(t);
                    let angle = p.y.atan2(p.x);
                    angle + TAU * ((t - angle) / TAU).round()
                }
            }
        }

        /// The range ends, the knots between pieces and the middles of the
        /// pieces: where the parameters agree.
        fn matching(&self, (first, last): (f64, f64)) -> Vec<f64> {
            let knots = match self {
                Direction::Straight => vec![first, last],
                Direction::Circular => conic_bspline::circle_arc(first, last).knots,
            };
            let middles: Vec<f64> = knots.windows(2).map(|k| 0.5 * (k[0] + k[1])).collect();
            knots.into_iter().chain(middles).collect()
        }
    }

    /// Checks a converted patch against the surface over `u × v`: the
    /// domain; every sampled point on the surface by `distance` (an upper
    /// bound of the distance to it); the B-spline at `(t, s)` equal to the
    /// surface at the angles the circle's B-spline gives `t` and `s` (so the
    /// boundary curves are the surface's at the range ends); and the
    /// parameters equal where they agree.
    fn check_patch<S: ParametricSurface>(
        b: &BSplineSurface,
        surface: &S,
        u: (f64, f64),
        v: (f64, f64),
        directions: (Direction, Direction),
        scale: f64,
        distance: impl Fn(Point3D) -> f64,
    ) {
        assert_eq!(b.u_bounds(), u);
        assert_eq!(b.v_bounds(), v);
        let (map_u, map_v) = (directions.0.map(u), directions.1.map(v));
        let (match_u, match_v) = (directions.0.matching(u), directions.1.matching(v));
        let n = 40;
        let dense = |(first, last): (f64, f64)| {
            (0..=n).map(move |i| first + (last - first) * (i as f64 / n as f64))
        };
        let us: Vec<f64> = dense(u).chain(match_u.iter().copied()).collect();
        let vs: Vec<f64> = dense(v).chain(match_v.iter().copied()).collect();
        for &t in &us {
            for &s in &vs {
                let p = b.eval_point(t, s);
                let d = distance(p);
                assert!(d <= RELATIVE * scale, "({t}, {s}) off the surface by {d}");
                let q = surface.eval_point(map_u(t), map_v(s));
                let d = p.distance(q);
                assert!(d <= RELATIVE * scale, "({t}, {s}) moved by {d}");
            }
        }
        for &t in &match_u {
            for &s in &match_v {
                let d = b.eval_point(t, s).distance(surface.eval_point(t, s));
                assert!(d <= 1e-14 * scale, "parameters differ at ({t}, {s}): {d}");
            }
        }
    }

    fn whole_turn((first, last): (f64, f64)) -> bool {
        last - first >= TAU * (1.0 - 1e-12)
    }

    fn assert_collapsed(poles: &[Vec<Point3D>], j: usize) {
        assert!(poles.iter().all(|row| row[j] == poles[0][j]));
    }

    // ---- planes ----

    #[test]
    fn test_plane_patch_is_bilinear_with_the_plane_parameters() {
        let plane = Plane::from_frame(tilted_frame(Point3D::new(1.0, -2.0, 3.0)));
        for (u, v) in [
            ((-3.0, 5.0), (0.5, 2.0)),
            ((0.2, 0.200_000_001), (-1e-9, 0.0)),
            ((-1e6, 1e6), (-5e5, -4e5)),
        ] {
            let b = plane.to_bspline_patch(u, v).unwrap();
            assert_eq!((b.u_degree(), b.v_degree(), b.is_rational()), (1, 1, false));
            let scale = u.0.abs().max(u.1.abs()).max(v.0.abs()).max(v.1.abs()) + 4.0;
            let distance = |p: Point3D| plane.frame().local_coordinates(p).2.abs();
            let directions = (Direction::Straight, Direction::Straight);
            check_patch(&b, &plane, u, v, directions, scale, distance);
        }
    }

    // ---- cylinders and cones ----

    fn cylinder(radius: f64) -> Cylinder {
        let origin = Point3D::new(0.4 * radius, -0.7 * radius, 0.2 * radius);
        Cylinder::from_frame(tilted_frame(origin), radius).unwrap()
    }

    #[test]
    fn test_cylinder_patches_are_exact() {
        for radius in [1e-6, 1.0, 1e6] {
            let c = cylinder(radius);
            let distance = |p: Point3D| {
                let (x, y, _) = c.frame().local_coordinates(p);
                (x.hypot(y) - radius).abs()
            };
            for u in ANGLES {
                for v in [
                    (-2.0 * radius, 3.0 * radius),
                    (radius, radius * (1.0 + 1e-9)),
                ] {
                    let b = c.to_bspline_patch(u, v).unwrap();
                    assert_eq!((b.u_degree(), b.v_degree()), (2, 1));
                    assert_eq!(b.is_u_periodic(), whole_turn(u));
                    let scale = 4.0 * radius;
                    let directions = (Direction::Circular, Direction::Straight);
                    check_patch(&b, &c, u, v, directions, scale, distance);
                    assert_eq!(b.eval_point(u.0, v.0), c.eval_point(u.0, v.0));
                }
            }
        }
    }

    #[test]
    fn test_whole_turn_cylinder_is_periodic_from_any_start() {
        let c = cylinder(2.0);
        for first in [0.0, -1.0, 3.0] {
            let b = c
                .to_bspline_patch((first, first + TAU), (0.0, 1.0))
                .unwrap();
            assert!(b.is_u_periodic() && !b.is_v_periodic());
            assert_eq!(b.u_period(), Some(TAU));
            let p = b.eval_point(first + 0.3, 0.5);
            assert!(b.eval_point(first + 0.3 + TAU, 0.5).distance(p) <= 1e-14);
            assert!(
                b.eval_point(first, 0.5)
                    .distance(b.eval_point(first + TAU, 0.5))
                    <= 1e-15
            );
        }
    }

    #[test]
    fn test_quarter_turn_pieces_have_the_circle_weights() {
        let profile = cylinder(1.0).profile((0.0, 1.0));
        let circle = angle_net(0.0, PI);
        assert_eq!(circle.knots, vec![0.0, FRAC_PI_2, PI]);
        let w = circle.weights.as_ref().unwrap();
        assert!((w[1] - FRAC_PI_4.cos()).abs() < 1e-15);
        assert_eq!(revolved_poles(Frame3D::WORLD, &circle, &profile).len(), 5);
        assert_eq!(profile.weights, None);
    }

    fn cone(semi_angle: f64) -> Cone {
        Cone::from_frame(tilted_frame(Point3D::new(1.0, 0.5, -2.0)), semi_angle, 2.0).unwrap()
    }

    /// Distance to the double cone, from the point's radius and height in
    /// the cone's frame.
    fn cone_distance(c: &Cone, p: Point3D) -> f64 {
        let (x, y, z) = c.frame().local_coordinates(p);
        let (sin, cos) = (c.semi_angle().sin(), c.semi_angle().cos());
        let rho = x.hypot(y);
        let to_line = |r: f64| ((r - c.ref_radius()) * cos - z * sin).abs();
        to_line(rho).min(to_line(-rho))
    }

    #[test]
    fn test_cone_patches_are_exact_on_both_nappes() {
        for alpha in [0.4, -0.6, 1.5] {
            let c = cone(alpha);
            let apex = -c.ref_radius() / alpha.sin();
            for u in ANGLES {
                for v in [(-1.0, 3.0), (apex, 3.0), (apex - 2.0, apex), (-10.0, 10.0)] {
                    let (v0, v1) = (v.0.min(v.1), v.0.max(v.1));
                    let b = c.to_bspline_patch(u, (v0, v1)).unwrap();
                    let scale = 2.0 + 2.0 * v0.abs().max(v1.abs());
                    let directions = (Direction::Circular, Direction::Straight);
                    check_patch(&b, &c, u, (v0, v1), directions, scale, |p| {
                        cone_distance(&c, p)
                    });
                }
            }
        }
    }

    #[test]
    fn test_cone_patch_collapses_at_the_apex() {
        let c = cone(0.4);
        let apex = -c.ref_radius() / 0.4f64.sin();
        let circle = angle_net(-1.0, 4.0);
        for (v, j) in [((apex, 3.0), 0), ((apex - 2.0, apex), 1)] {
            let profile = c.profile(v);
            assert_eq!(profile.poles[j][0], 0.0);
            let poles = revolved_poles(c.frame(), &circle, &profile);
            assert_collapsed(&poles, j);
            assert!(poles[0][j].distance(c.apex()) <= 1e-14);
            let b = c.to_bspline_patch((-1.0, 4.0), v).unwrap();
            let v_apex = if j == 0 { v.0 } else { v.1 };
            for t in [-1.0, 0.0, 1.3, 4.0] {
                assert!(b.eval_point(t, v_apex).distance(c.apex()) <= 1e-14);
            }
        }
        let away = c.profile((apex + 1e-6, 3.0));
        assert!(away.poles[0][0] > 0.0);
    }

    // ---- spheres and tori ----

    fn sphere(radius: f64) -> Sphere {
        let center = Point3D::new(0.4 * radius, -0.7 * radius, 0.2 * radius);
        Sphere::from_frame(tilted_frame(center), radius).unwrap()
    }

    const LATITUDES: [(f64, f64); 6] = [
        (-FRAC_PI_2, FRAC_PI_2),
        (-FRAC_PI_2, -1.0),
        (0.2, FRAC_PI_2),
        (-0.3, 0.4),
        (FRAC_PI_2 - 1e-9, FRAC_PI_2),
        (0.5, 0.500_000_01),
    ];

    #[test]
    fn test_whole_sphere_is_periodic_with_collapsed_poles() {
        for radius in [1e-6, 2.5, 1e6] {
            let s = sphere(radius);
            let b = s.to_bspline();
            assert!(b.is_u_periodic() && !b.is_v_periodic() && b.is_rational());
            assert_eq!((b.u_degree(), b.v_degree()), (2, 2));
            assert_eq!(b.u_period(), Some(TAU));
            let distance = |p: Point3D| (p.distance(s.center()) - radius).abs();
            let directions = (Direction::Circular, Direction::Circular);
            let (u, v) = ((0.0, TAU), (-FRAC_PI_2, FRAC_PI_2));
            check_patch(&b, &s, u, v, directions, radius, distance);

            let profile = s.profile(v);
            assert_eq!(profile.poles.len(), 5);
            assert_eq!((profile.poles[0][0], profile.poles[4][0]), (0.0, 0.0));
            let poles = revolved_poles(s.frame(), &angle_net(0.0, TAU), &profile);
            assert_collapsed(&poles, 0);
            assert_collapsed(&poles, 4);
            let north = s.eval_point(0.0, FRAC_PI_2);
            assert!(poles[0][4].distance(north) <= 1e-15 * radius);
        }
    }

    #[test]
    fn test_sphere_patches_are_exact() {
        for radius in [1e-6, 1.0, 1e6] {
            let s = sphere(radius);
            let distance = |p: Point3D| (p.distance(s.center()) - radius).abs();
            for u in ANGLES {
                for v in LATITUDES {
                    let b = s.to_bspline_patch(u, v).unwrap();
                    assert_eq!(b.is_u_periodic(), whole_turn(u));
                    let directions = (Direction::Circular, Direction::Circular);
                    check_patch(&b, &s, u, v, directions, radius, distance);
                }
            }
        }
    }

    fn torus(major: f64, minor: f64) -> Torus {
        Torus::from_frame(tilted_frame(Point3D::new(-1.0, 2.0, 0.5)), major, minor).unwrap()
    }

    /// Distance to the torus, the point's radius taken on either side of
    /// the axis for a tube that crosses it.
    fn torus_distance(t: &Torus, p: Point3D) -> f64 {
        let (x, y, z) = t.frame().local_coordinates(p);
        let rho = x.hypot(y);
        let to_tube = |r: f64| ((r - t.major_radius()).hypot(z) - t.minor_radius()).abs();
        to_tube(rho).min(to_tube(-rho))
    }

    #[test]
    fn test_whole_torus_is_periodic_both_ways() {
        let t = torus(3.0, 1.0);
        let b = t.to_bspline();
        assert!(b.is_u_periodic() && b.is_v_periodic());
        assert_eq!((b.u_period(), b.v_period()), (Some(TAU), Some(TAU)));
        let directions = (Direction::Circular, Direction::Circular);
        check_patch(&b, &t, (0.0, TAU), (0.0, TAU), directions, 4.0, |p| {
            torus_distance(&t, p)
        });
        assert_eq!(t.profile((0.0, TAU)).poles.len(), 8);
    }

    #[test]
    fn test_torus_patches_are_exact() {
        // Ring, horn and spindle tori.
        for (major, minor) in [(3.0, 1.0), (1.5, 1.5), (1.0, 2.0), (1e6, 1e-3)] {
            let t = torus(major, minor);
            let scale = major + minor + 3.0;
            for u in ANGLES {
                for v in ANGLES {
                    let b = t.to_bspline_patch(u, v).unwrap();
                    assert_eq!(b.is_u_periodic(), whole_turn(u));
                    assert_eq!(b.is_v_periodic(), whole_turn(v));
                    let directions = (Direction::Circular, Direction::Circular);
                    check_patch(&b, &t, u, v, directions, scale, |p| torus_distance(&t, p));
                }
            }
        }
    }

    #[test]
    fn test_horn_torus_collapses_on_the_axis() {
        let t = torus(1.5, 1.5);
        let profile = t.profile((PI - 1.0, PI));
        assert_eq!(profile.poles[2][0], 0.0);
        let poles = revolved_poles(t.frame(), &angle_net(0.0, 2.0), &profile);
        assert_collapsed(&poles, 2);
        assert!(poles[0][2].distance(t.center()) <= 1e-15);
    }

    // ---- errors ----

    #[test]
    fn test_unbounded_and_invalid_ranges_are_rejected() {
        let plane = Plane::from_frame(Frame3D::WORLD);
        let c = cylinder(1.0);
        let s = sphere(1.0);
        let unbounded = Err(BSplineConversionError::Unbounded);
        let invalid = Err(BSplineConversionError::InvalidRange);
        let everywhere = (f64::NEG_INFINITY, f64::INFINITY);
        assert_eq!(plane.to_bspline_patch(everywhere, (0.0, 1.0)), unbounded);
        assert_eq!(
            plane.to_bspline_patch((0.0, 1.0), (0.0, f64::INFINITY)),
            unbounded
        );
        assert_eq!(c.to_bspline_patch((0.0, 1.0), everywhere), unbounded);
        assert_eq!(
            cone(0.4).to_bspline_patch((0.0, 1.0), (f64::NEG_INFINITY, 0.0)),
            unbounded
        );
        assert_eq!(c.to_bspline_patch((f64::NAN, 1.0), (0.0, 1.0)), invalid);
        assert_eq!(c.to_bspline_patch((1.0, 1.0), (0.0, 1.0)), invalid);
        assert_eq!(c.to_bspline_patch((0.0, 1.0), (2.0, 1.0)), invalid);
        assert_eq!(c.to_bspline_patch((0.0, TAU + 1e-9), (0.0, 1.0)), invalid);
        assert!(
            c.to_bspline_patch((0.0, TAU * (1.0 + 1e-13)), (0.0, 1.0))
                .is_ok()
        );
        assert_eq!(s.to_bspline_patch((0.0, 1.0), (-2.0, 0.0)), invalid);
        assert_eq!(
            s.to_bspline_patch((0.0, 1.0), (0.0, FRAC_PI_2 + 1e-9)),
            invalid
        );
        assert_eq!(
            torus(3.0, 1.0).to_bspline_patch((0.0, 1.0), (0.0, 7.0)),
            invalid
        );
    }

    #[test]
    fn test_overflowing_patches_are_not_representable() {
        let not_representable = Err(BSplineConversionError::NotRepresentable);
        let frame = Frame3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::new(1.0, 1.0, 0.0));
        let plane = Plane::from_frame(frame.unwrap());
        let huge = (-1.5e308, 1.5e308);
        assert_eq!(plane.to_bspline_patch(huge, huge), not_representable);
        // At 2^54 the parameters are 4 apart: three pieces cannot fit.
        let first = 2f64.powi(54);
        let c = cylinder(1.0);
        assert_eq!(
            c.to_bspline_patch((first, first + 4.0), (0.0, 1.0)),
            not_representable
        );
        let far = Sphere::new(Point3D::new(1.5e308, 0.0, 0.0), 1e308).unwrap();
        assert_eq!(
            far.to_bspline_patch((0.0, 1.0), (0.0, 1.0)),
            not_representable
        );
    }

    // ---- run-time surfaces ----

    #[test]
    fn test_run_time_surfaces_convert_like_the_concrete_ones() {
        let s = sphere(2.0);
        assert_eq!(ElementarySurface::from(s).to_bspline(), Ok(s.to_bspline()));
        assert_eq!(Surface::from(s).to_bspline(), Ok(s.to_bspline()));
        let t = torus(3.0, 1.0);
        assert_eq!(Surface::from(t).to_bspline(), Ok(t.to_bspline()));
        let b = t.to_bspline();
        assert_eq!(Surface::from(b.clone()).to_bspline(), Ok(b));

        let (u, v) = ((-1.0, 2.0), (0.0, 0.5));
        let plane = Plane::from_frame(tilted_frame(Point3D::ORIGIN));
        let c = cylinder(1.0);
        let k = cone(0.4);
        let surfaces: [ElementarySurface; 5] =
            [plane.into(), c.into(), k.into(), s.into(), t.into()];
        let concrete = [
            plane.to_bspline_patch(u, v),
            c.to_bspline_patch(u, v),
            k.to_bspline_patch(u, v),
            s.to_bspline_patch(u, v),
            t.to_bspline_patch(u, v),
        ];
        for (surface, expected) in surfaces.iter().zip(concrete) {
            assert!(expected.is_ok());
            assert_eq!(surface.to_bspline_patch(u, v), expected);
        }
        for unbounded in [Surface::from(plane), c.into(), k.into()] {
            let err = Some(BSplineConversionError::Unbounded);
            assert_eq!(unbounded.to_bspline().err(), err);
            assert_eq!(unbounded.as_elementary().unwrap().to_bspline().err(), err);
        }
    }

    // ---- the B-splines work with the B-spline algorithms ----

    #[test]
    fn test_converted_sphere_projects_like_the_sphere() {
        let tol = Tolerance::DEFAULT;
        let s = sphere(2.0);
        let b = s.to_bspline();
        for (u, v) in [(0.3, 0.2), (4.0, -1.2), (1.0, 1.5)] {
            let q = s.eval_point(u, v) + 0.7 * (s.eval_point(u, v) - s.center());
            let expected = s.project_point(q, tol).distance;
            assert!((b.project_point(q, tol).distance - expected).abs() < 1e-9);
            assert!(b.contains(s.eval_point(u + 0.1, v), tol));
        }
        let k = cone(0.4);
        let patch = k.to_bspline_patch((0.0, PI), (-1.0, 2.0)).unwrap();
        assert!(patch.contains(k.eval_point(2.0, 1.5), tol));
        assert!(!patch.contains(k.eval_point(4.0, 1.5), tol));
    }
}
