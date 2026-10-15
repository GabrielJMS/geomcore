//! The routed `intersect` shared by every curve and surface class.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use geomcore::curves::{Curve2D, Curve3D, ParametricCurve3D, TrimmedCurve2D};
use geomcore::surfaces::Surface;
use geomcore::{CurveCurveIntersection2D, Intersect, Tolerance};

use crate::curves::PyTrimmedCurve2D;
use crate::numeric::{
    CurveArg, curve_arg_of, curve2d_of, routed_curve_curve, routed_curve_surface,
};
use crate::results::{
    PyCurveCurveIntersection, PyCurveCurveIntersection2D, PyCurveSurfaceIntersection,
    PySurfaceSurfaceIntersection,
};
use crate::surfaces::surface_of;

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

/// The operand `obj` holds, read by type tests that raise nothing on a
/// miss: the surface classes first when `surfaces_first` (the other
/// operand is a surface), the 3D curve classes first otherwise.
fn operand_of(obj: &Bound<'_, PyAny>, surfaces_first: bool) -> PyResult<Operand> {
    let surface = || surface_of(obj).map(Operand::Surface);
    let curve = || curve_arg_of(obj).map(Operand::Curve);
    let found = if surfaces_first {
        surface().or_else(curve)
    } else {
        curve().or_else(surface)
    };
    if let Some(operand) = found {
        return Ok(operand);
    }
    if let Ok(trimmed) = obj.downcast::<PyTrimmedCurve2D>() {
        return Ok(Operand::Curve2D(CurveArg2D::Trimmed(trimmed.borrow().0)));
    }
    if let Some(curve) = curve2d_of(obj) {
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

// The numeric search and marching can run for milliseconds: the routed
// call releases the interpreter lock around them, so other Python threads
// run meanwhile. Releasing it costs about 35 ns, a tenth of a closed-form
// call from Python, so the pairs that always have a closed form keep it.

/// Whether the curve is a B-spline, or a piece of one: a pair with it has
/// no closed form.
fn is_bspline(curve: &CurveArg) -> bool {
    match curve {
        CurveArg::Whole(curve) => curve.as_bspline().is_some(),
        CurveArg::Trimmed(curve) => curve.as_bspline().is_some(),
    }
}

/// Whether the routed curve-surface call may run the numeric search: a
/// B-spline on either side, or a curve other than a whole line against a
/// torus.
fn may_search(curve: &CurveArg, surface: &Surface) -> bool {
    let line = matches!(curve, CurveArg::Whole(Curve3D::Line(_)));
    is_bspline(curve)
        || matches!(surface, Surface::BSpline(_))
        || (!line && matches!(surface, Surface::Torus(_)))
}

/// Intersects `first` with `second` through the routed Rust `intersect`
/// and wraps the result with both operands.
pub(crate) fn intersect(
    first: &Bound<'_, PyAny>,
    second: &Bound<'_, PyAny>,
    tol: Tolerance,
) -> PyResult<Py<PyAny>> {
    let py = first.py();
    // The receiver is read curves first: most calls are on curves.
    let a = operand_of(first, false)?;
    let b = operand_of(second, matches!(a, Operand::Surface(_)))?;
    let (first, second) = (first.clone().unbind(), second.clone().unbind());
    let object = match (&a, &b) {
        (Operand::Curve(a), Operand::Curve(b)) => PyCurveCurveIntersection {
            result: if is_bspline(a) || is_bspline(b) {
                py.detach(|| routed_curve_curve(a, b, tol))
            } else {
                routed_curve_curve(a, b, tol)
            },
            first,
            second,
        }
        .into_pyobject(py)?
        .into_any(),
        (Operand::Curve(curve), Operand::Surface(surface))
        | (Operand::Surface(surface), Operand::Curve(curve)) => PyCurveSurfaceIntersection {
            result: if may_search(curve, surface) {
                py.detach(|| routed_curve_surface(curve, surface, tol))
            } else {
                routed_curve_surface(curve, surface, tol)
            },
            first,
            second,
        }
        .into_pyobject(py)?
        .into_any(),
        (Operand::Surface(a), Operand::Surface(b)) => PySurfaceSurfaceIntersection {
            // Marching runs for some pairs, and the closed forms build
            // curves: releasing the lock costs little next to either.
            result: py.detach(|| a.intersect(b, tol)),
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
