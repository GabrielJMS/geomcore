use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use geomcore::curves::{Circle3D, Curve2D, Curve3D, ParametricCurve3D, TrimmedCurve3D};
use geomcore::parametrize_numeric;
use geomcore::surfaces::Surface;
use geomcore::{
    CircleSurfaceIntersection, ConicConicIntersection, ConicSurfaceIntersection,
    LineConic3DIntersection, LineConicHit, LineQuadricIntersection, LineTorusHit,
    LineTorusIntersection, Point2D, Point3D, Tolerance, Vector3D,
};
use geomcore::{CurveCurveIntersection, CurveSurfaceIntersection, Intersect, curve_curve_extrema};

use crate::curves::*;
use crate::surfaces::*;
use crate::tolerance;
use crate::values::*;

pub(crate) fn extract_curve3d(obj: &Bound<'_, PyAny>) -> PyResult<Curve3D> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyLine3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyCircle3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyEllipse3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyParabola3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyHyperbola3D>>() {
        return Ok(Curve3D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyBSplineCurve3D>>() {
        // BSplineCurve3D owns heap data (not Copy): clone is required here.
        #[allow(clippy::clone_on_copy)]
        return Ok(Curve3D::from(c.0.clone()));
    }
    Err(PyTypeError::new_err(
        "expected a geomcore 3D curve (Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D or BSplineCurve3D)",
    ))
}

/// A 3D curve passed to a generic function: a whole curve, or a piece of
/// one, whose range bounds the search.
pub(crate) enum CurveArg {
    Whole(Curve3D),
    Trimmed(TrimmedCurve3D),
}

impl ParametricCurve3D for CurveArg {
    fn eval_point(&self, u: f64) -> Point3D {
        match self {
            CurveArg::Whole(c) => c.eval_point(u),
            CurveArg::Trimmed(c) => c.eval_point(u),
        }
    }

    fn eval_derivative(&self, u: f64, order: u32) -> Vector3D {
        match self {
            CurveArg::Whole(c) => c.eval_derivative(u, order),
            CurveArg::Trimmed(c) => c.eval_derivative(u, order),
        }
    }

    fn bounds(&self) -> (f64, f64) {
        match self {
            CurveArg::Whole(c) => c.bounds(),
            CurveArg::Trimmed(c) => c.bounds(),
        }
    }

    fn period(&self) -> Option<f64> {
        match self {
            CurveArg::Whole(c) => c.period(),
            CurveArg::Trimmed(c) => c.period(),
        }
    }

    fn eval_points(&self, us: &[f64]) -> Vec<Point3D> {
        match self {
            CurveArg::Whole(c) => c.eval_points(us),
            CurveArg::Trimmed(c) => c.eval_points(us),
        }
    }
}

pub(crate) fn extract_curve_arg(obj: &Bound<'_, PyAny>) -> PyResult<CurveArg> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyTrimmedCurve3D>>() {
        return Ok(CurveArg::Trimmed(c.0.clone()));
    }
    extract_curve3d(obj).map(CurveArg::Whole).map_err(|_| {
        PyTypeError::new_err(
            "expected a geomcore 3D curve (Line3D, Circle3D, Ellipse3D, Parabola3D, Hyperbola3D, BSplineCurve3D or TrimmedCurve3D)",
        )
    })
}

pub(crate) fn curve3d_to_py(py: Python<'_>, curve: Curve3D) -> PyResult<Py<PyAny>> {
    match curve {
        Curve3D::Line(c) => Ok(PyLine3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Circle(c) => Ok(PyCircle3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Ellipse(c) => Ok(PyEllipse3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Parabola(c) => Ok(PyParabola3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::Hyperbola(c) => Ok(PyHyperbola3D(c).into_pyobject(py)?.into_any().unbind()),
        Curve3D::BSpline(c) => Ok(PyBSplineCurve3D(c).into_pyobject(py)?.into_any().unbind()),
        _ => Err(PyValueError::new_err("unsupported 3d curve variant")),
    }
}

pub(crate) fn extract_curve2d(obj: &Bound<'_, PyAny>) -> PyResult<Curve2D> {
    if let Ok(c) = obj.extract::<PyRef<'_, PyLine2D>>() {
        return Ok(Curve2D::from(c.0));
    }
    if let Ok(c) = obj.extract::<PyRef<'_, PyCircle2D>>() {
        return Ok(Curve2D::from(c.0));
    }
    Err(PyTypeError::new_err(
        "expected a geomcore 2D curve (Line2D or Circle2D)",
    ))
}

/// A generic intersection hit: `(curve parameter, (u, v), point)`.
pub(crate) type CurveHitPy = (Option<f64>, Option<(f64, f64)>, Py<PyAny>);

/// A curve-curve hit as Python sees it: `(first_parameter, Point3D,
/// second_parameter)`, a parameter `None` when the solver did not compute
/// it.
pub(crate) type CurveCurveHitPy = (Option<f64>, Py<PyAny>, Option<f64>);

/// Closest-point enumeration between any two 3D curves.
///
/// Every local minimum of the distance over the curves' parameter
/// domains, including minima at the ends of bounded curves, as
/// `(first_parameter, second_parameter, distance)` tuples ordered by distance.
/// Parallel lines, concentric circles and other continuous families of
/// equal minima report the distance once, with a representative pair.
/// Use the analytic methods when classification matters.
#[pyfunction]
#[pyo3(name = "curve_curve_extrema", signature = (curve_a, curve_b, tol = None))]
pub(crate) fn py_curve_curve_extrema(
    curve_a: &Bound<'_, PyAny>,
    curve_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let tol = tolerance(tol);
    let a = extract_curve_arg(curve_a)?;
    let b = extract_curve_arg(curve_b)?;
    Ok(curve_curve_extrema(&a, &b, tol)
        .iter()
        .map(|e| (e.first_parameter, e.second_parameter, e.distance))
        .collect())
}

/// Generic 3D curve-curve intersection through the routed `intersect`:
/// exact when the pair has a closed form, numeric otherwise.
///
/// Returns `(first_parameter, Point3D, second_parameter)` tuples. A
/// parameter is `None` when the solver did not compute it; recover it with
/// the curve's `parameter_of(point)`. The tuples are ordered by first-curve
/// parameter when every hit has one, and in the solver's order otherwise.
/// Ranges where the curves coincide are not hits and are not returned.
#[pyfunction]
#[pyo3(name = "intersect_curve_curve", signature = (curve_a, curve_b, tol = None))]
pub(crate) fn py_intersect_curve_curve(
    py: Python<'_>,
    curve_a: &Bound<'_, PyAny>,
    curve_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<CurveCurveHitPy>> {
    let tol = tolerance(tol);
    let a = extract_curve_arg(curve_a)?;
    let b = extract_curve_arg(curve_b)?;
    let mut out = Vec::new();
    for h in routed_curve_curve(&a, &b, tol).hits() {
        out.push((
            h.first_parameter,
            PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
            h.second_parameter,
        ));
    }
    Ok(out)
}
/// Generic curve-surface intersection for any curve and surface, through
/// the routed `intersect`: exact when the pair has a closed form, numeric
/// otherwise.
///
/// Returns a `(curve_parameter, (u, v), Point3D)` tuple per hit. A
/// parameter is `None` when the solver did not compute it (the exact path
/// never computes `(u, v)`); recover it with the curve's
/// `parameter_of(point)` or the surface's `parameters_of(point)`. The
/// tuples are ordered by curve parameter when every hit has one. Ranges
/// where the curve lies on the surface are not hits and are not returned.
#[pyfunction]
#[pyo3(name = "intersect_curve_surface", signature = (curve, surface, tol = None))]
pub(crate) fn py_intersect_curve_surface(
    py: Python<'_>,
    curve: &Bound<'_, PyAny>,
    surface: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<CurveHitPy>> {
    let tol = tolerance(tol);
    let c = extract_curve_arg(curve)?;
    let s = extract_surface(surface)?;
    let mut out = Vec::new();
    for h in routed_curve_surface(&c, &s, tol).hits() {
        out.push((
            h.curve_parameter,
            h.surface_parameters,
            PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
        ));
    }
    Ok(out)
}

/// The routed intersection of two curves passed from Python.
pub(crate) fn routed_curve_curve(
    a: &CurveArg,
    b: &CurveArg,
    tol: Tolerance,
) -> CurveCurveIntersection {
    match (a, b) {
        (CurveArg::Whole(a), CurveArg::Whole(b)) => a.intersect(b, tol),
        (CurveArg::Whole(a), CurveArg::Trimmed(b)) => a.intersect(b, tol),
        (CurveArg::Trimmed(a), CurveArg::Whole(b)) => a.intersect(b, tol),
        (CurveArg::Trimmed(a), CurveArg::Trimmed(b)) => a.intersect(b, tol),
    }
}

/// The routed intersection of a curve and a surface passed from Python.
pub(crate) fn routed_curve_surface(
    curve: &CurveArg,
    surface: &Surface,
    tol: Tolerance,
) -> CurveSurfaceIntersection {
    match curve {
        CurveArg::Whole(curve) => curve.intersect(surface, tol),
        CurveArg::Trimmed(curve) => curve.intersect(surface, tol),
    }
}

/// Numeric curve-on-surface parametrization for pairs without closed form.
///
/// Samples the curve uniformly (`n` points) and inverts each sample onto
/// the surface. Returns `(u, v)` tuples; surface evaluation at them
/// recovers the samples where the curve lies on the surface.
#[pyfunction]
#[pyo3(name = "parametrize_numeric", signature = (curve, surface, n = 64, tol = None))]
pub(crate) fn py_parametrize_numeric(
    curve: &Bound<'_, PyAny>,
    surface: &Bound<'_, PyAny>,
    n: usize,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<(f64, f64)>> {
    let tol = tolerance(tol);
    let c = extract_curve_arg(curve)?;
    let s = extract_surface(surface)?;
    Ok(parametrize_numeric(&c, &s, n, tol)
        .iter()
        .map(|p| (p.x, p.y))
        .collect())
}

pub(crate) fn curve2d_to_py(py: Python<'_>, curve: Curve2D) -> PyResult<Py<PyAny>> {
    match curve {
        Curve2D::Line(l) => Ok(PyLine2D(l).into_pyobject(py)?.into_any().unbind()),
        Curve2D::Circle(c) => Ok(PyCircle2D(c).into_pyobject(py)?.into_any().unbind()),
        _ => Err(PyValueError::new_err("unsupported 2d curve variant")),
    }
}

pub(crate) fn circles_to_py(py: Python<'_>, c1: Circle3D, c2: Circle3D) -> PyResult<Py<PyAny>> {
    (
        PyCircle3D(c1).into_pyobject(py)?.into_any().unbind(),
        PyCircle3D(c2).into_pyobject(py)?.into_any().unbind(),
    )
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

pub(crate) fn conic_surface_to_py(
    py: Python<'_>,
    hit: ConicSurfaceIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        ConicSurfaceIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in hits {
                out.push((
                    h.parameter,
                    PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
                    h.multiplicity,
                ));
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        ConicSurfaceIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        ConicSurfaceIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

pub(crate) fn line_conic_hit(py: Python<'_>, h: &LineConicHit) -> PyResult<Py<PyAny>> {
    (
        h.line_parameter,
        PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
        h.conic_parameter,
    )
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

pub(crate) fn line_conic_to_py(
    py: Python<'_>,
    hit: LineConic3DIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        LineConic3DIntersection::Points(h1, h2) => Ok((
            "points".to_string(),
            (line_conic_hit(py, &h1)?, line_conic_hit(py, &h2)?)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        LineConic3DIntersection::Tangent(h) => Ok(("tangent".to_string(), line_conic_hit(py, &h)?)),
        LineConic3DIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

pub(crate) fn conic_conic_to_py(
    py: Python<'_>,
    hit: ConicConicIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        ConicConicIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in &hits {
                out.push(
                    (
                        h.first_parameter,
                        PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
                        h.second_parameter,
                        h.multiplicity,
                    )
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                );
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        ConicConicIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
        ConicConicIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

pub(crate) fn line_torus_hit(py: Python<'_>, h: &LineTorusHit) -> PyResult<Py<PyAny>> {
    (
        h.line_parameter,
        PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
        h.surface_parameters,
        h.multiplicity,
    )
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

pub(crate) fn line_torus_to_py(
    py: Python<'_>,
    hit: LineTorusIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        LineTorusIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in &hits {
                out.push(line_torus_hit(py, h)?);
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        LineTorusIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

pub(crate) fn quadric_hit_to_py(
    py: Python<'_>,
    hit: LineQuadricIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    fn point(py: Python<'_>, t: f64, p: Point3D) -> PyResult<Py<PyAny>> {
        (t, PyPoint3D(p).into_pyobject(py)?.into_any().unbind())
            .into_pyobject(py)
            .map(|o| o.into_any().unbind())
    }
    match hit {
        LineQuadricIntersection::TwoPoints((t1, p1), (t2, p2)) => Ok((
            "two_points".to_string(),
            (point(py, t1, p1)?, point(py, t2, p2)?)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )),
        LineQuadricIntersection::OnePoint(t, p) => Ok(("one_point".to_string(), point(py, t, p)?)),
        LineQuadricIntersection::Tangent(t, p) => Ok(("tangent".to_string(), point(py, t, p)?)),
        LineQuadricIntersection::Empty => Ok(("empty".to_string(), py.None())),
        LineQuadricIntersection::Coincident => Ok(("coincident".to_string(), py.None())),
    }
}

pub(crate) fn point2(py: Python<'_>, t: f64, p: Point2D) -> PyResult<Py<PyAny>> {
    (t, PyPoint2D(p).into_pyobject(py)?.into_any().unbind())
        .into_pyobject(py)
        .map(|o| o.into_any().unbind())
}

pub(crate) fn circle_surface_to_py(
    py: Python<'_>,
    hit: CircleSurfaceIntersection,
) -> PyResult<(String, Py<PyAny>)> {
    match hit {
        CircleSurfaceIntersection::Circle(c) => Ok((
            "circle".to_string(),
            PyCircle3D(c).into_pyobject(py)?.into_any().unbind(),
        )),
        CircleSurfaceIntersection::Hits(hits) => {
            let mut out = Vec::with_capacity(hits.len());
            for h in hits {
                out.push((
                    h.parameter,
                    PyPoint3D(h.point).into_pyobject(py)?.into_any().unbind(),
                    h.multiplicity,
                ));
            }
            Ok((
                "hits".to_string(),
                out.into_pyobject(py)?.into_any().unbind(),
            ))
        }
        CircleSurfaceIntersection::Empty => Ok(("empty".to_string(), py.None())),
    }
}

/// Surface-surface intersection of any two surfaces, through the routed
/// Rust `intersect`.
///
/// Returns one curve per meeting curve (empty when the surfaces miss):
/// a line, circle, ellipse, parabola or hyperbola where a closed form
/// exists, a `BSplineCurve3D` fitted by marching otherwise.
#[pyfunction]
#[pyo3(name = "intersect_marching", signature = (surface_a, surface_b, tol = None))]
pub(crate) fn py_intersect_marching(
    py: Python<'_>,
    surface_a: &Bound<'_, PyAny>,
    surface_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<Vec<Py<PyAny>>> {
    let tol = tolerance(tol);
    let a = extract_surface(surface_a)?;
    let b = extract_surface(surface_b)?;
    a.intersect(&b, tol)
        .curves()
        .iter()
        .map(|piece| curve3d_to_py(py, piece.curve.clone()))
        .collect()
}
