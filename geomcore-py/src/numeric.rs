use crate::errors::invalid_argument;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use geomcore::curves::{
    BSplineCurve3D, Curve2D, Curve3D, ElementaryCurve, ParametricCurve3D, TrimmedCurve3D,
};
use geomcore::parametrize_numeric;
use geomcore::surfaces::Surface;
use geomcore::{CurveCurveIntersection, CurveSurfaceIntersection, Intersect, curve_curve_extrema};
use geomcore::{Point3D, Tolerance, Vector3D};

use crate::curves::*;
use crate::results::PyCurveCurveExtrema;
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

    // Forwarded so that the search sees a B-spline's pieces or an
    // elementary curve's closed form through the wrapper.
    fn as_bspline(&self) -> Option<&BSplineCurve3D> {
        match self {
            CurveArg::Whole(c) => c.as_bspline(),
            CurveArg::Trimmed(c) => c.as_bspline(),
        }
    }

    fn as_elementary(&self) -> Option<ElementaryCurve> {
        match self {
            CurveArg::Whole(c) => c.as_elementary(),
            CurveArg::Trimmed(c) => c.as_elementary(),
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
        _ => Err(invalid_argument("unsupported 3d curve variant")),
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

/// Closest-point enumeration between any two 3D curves.
///
/// Returns a `CurveCurveExtrema`: every local minimum of the distance over
/// the curves' parameter domains, including minima at the ends of bounded
/// curves, closest first, with whether the search proved there is no
/// other. Parallel lines, concentric circles and other continuous
/// families of equal minima report the distance once, with a
/// representative pair. The result holds the curves as `first` and
/// `second`.
#[pyfunction]
#[pyo3(name = "curve_curve_extrema", signature = (curve_a, curve_b, tol = None))]
pub(crate) fn py_curve_curve_extrema(
    curve_a: &Bound<'_, PyAny>,
    curve_b: &Bound<'_, PyAny>,
    tol: Option<PyTolerance>,
) -> PyResult<PyCurveCurveExtrema> {
    let tol = tolerance(tol);
    let a = extract_curve_arg(curve_a)?;
    let b = extract_curve_arg(curve_b)?;
    Ok(PyCurveCurveExtrema {
        result: curve_curve_extrema(&a, &b, tol),
        first: curve_a.clone().unbind(),
        second: curve_b.clone().unbind(),
    })
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
        _ => Err(invalid_argument("unsupported 2d curve variant")),
    }
}
