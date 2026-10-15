//! The routed `intersect` shared by every curve and surface class.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use geomcore::curves::{Curve2D, TrimmedCurve2D};
use geomcore::surfaces::Surface;
use geomcore::{CurveCurveIntersection2D, Intersect, Tolerance};

use crate::curves::PyTrimmedCurve2D;
use crate::numeric::{
    CurveArg, extract_curve_arg, extract_curve2d, routed_curve_curve, routed_curve_surface,
};
use crate::results::{
    PyCurveCurveIntersection, PyCurveCurveIntersection2D, PyCurveSurfaceIntersection,
    PySurfaceSurfaceIntersection,
};
use crate::surfaces::extract_surface;

/// A 2D curve passed from Python: a whole curve or a piece of one.
enum CurveArg2D {
    Whole(Curve2D),
    Trimmed(TrimmedCurve2D),
}

/// An `intersect` operand, read from any curve or surface class.
enum Operand {
    Curve(CurveArg),
    Curve2D(CurveArg2D),
    Surface(Surface),
}

fn extract_operand(obj: &Bound<'_, PyAny>) -> PyResult<Operand> {
    if let Ok(curve) = extract_curve_arg(obj) {
        return Ok(Operand::Curve(curve));
    }
    if let Ok(surface) = extract_surface(obj) {
        return Ok(Operand::Surface(surface));
    }
    if let Ok(trimmed) = obj.extract::<PyRef<'_, PyTrimmedCurve2D>>() {
        return Ok(Operand::Curve2D(CurveArg2D::Trimmed(trimmed.0)));
    }
    if let Ok(curve) = extract_curve2d(obj) {
        return Ok(Operand::Curve2D(CurveArg2D::Whole(curve)));
    }
    Err(PyTypeError::new_err(format!(
        "intersect takes a geomcore curve or surface, not {}",
        obj.get_type().name()?
    )))
}

fn routed_curve_curve_2d(
    a: &CurveArg2D,
    b: &CurveArg2D,
    tol: Tolerance,
) -> CurveCurveIntersection2D {
    match (a, b) {
        (CurveArg2D::Whole(a), CurveArg2D::Whole(b)) => a.intersect(b, tol),
        (CurveArg2D::Whole(a), CurveArg2D::Trimmed(b)) => a.intersect(b, tol),
        (CurveArg2D::Trimmed(a), CurveArg2D::Whole(b)) => a.intersect(b, tol),
        (CurveArg2D::Trimmed(a), CurveArg2D::Trimmed(b)) => a.intersect(b, tol),
    }
}

/// Intersects `first` with `second` through the routed Rust `intersect`
/// and wraps the result with both operands.
pub(crate) fn intersect(
    first: &Bound<'_, PyAny>,
    second: &Bound<'_, PyAny>,
    tol: Tolerance,
) -> PyResult<Py<PyAny>> {
    let py = first.py();
    let a = extract_operand(first)?;
    let b = extract_operand(second)?;
    let (first, second) = (first.clone().unbind(), second.clone().unbind());
    let object = match (&a, &b) {
        (Operand::Curve(a), Operand::Curve(b)) => PyCurveCurveIntersection {
            result: routed_curve_curve(a, b, tol),
            first,
            second,
        }
        .into_pyobject(py)?
        .into_any(),
        (Operand::Curve(curve), Operand::Surface(surface))
        | (Operand::Surface(surface), Operand::Curve(curve)) => PyCurveSurfaceIntersection {
            result: routed_curve_surface(curve, surface, tol),
            first,
            second,
        }
        .into_pyobject(py)?
        .into_any(),
        (Operand::Surface(a), Operand::Surface(b)) => PySurfaceSurfaceIntersection {
            result: a.intersect(b, tol),
            first,
            second,
        }
        .into_pyobject(py)?
        .into_any(),
        (Operand::Curve2D(a), Operand::Curve2D(b)) => PyCurveCurveIntersection2D {
            result: routed_curve_curve_2d(a, b, tol),
            first,
            second,
        }
        .into_pyobject(py)?
        .into_any(),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "cannot intersect a 2D curve with a 3D curve or a surface ({} and {})",
                first.bind(py).get_type().name()?,
                second.bind(py).get_type().name()?
            )));
        }
    };
    Ok(object.unbind())
}
