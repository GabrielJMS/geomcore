import pytest

import geomcore
from geomcore import curves, surfaces

SUBCLASS_PARENTS = {
    "ConstructionError": "GeomcoreError",
    "FrameConstructionError": "ConstructionError",
    "TrimmedCurveConstructionError": "ConstructionError",
    "KnotInsertionError": "GeomcoreError",
    "KnotRemovalError": "GeomcoreError",
    "DegreeElevationError": "GeomcoreError",
    "BSplineSplitError": "GeomcoreError",
    "BSplineConversionError": "GeomcoreError",
    "ParametrizeError": "GeomcoreError",
    "SurfaceIsoError": "GeomcoreError",
    "DiscretizationCriteriaError": "GeomcoreError",
}


def test_base_is_value_error():
    assert issubclass(geomcore.GeomcoreError, ValueError)


@pytest.mark.parametrize("name, parent", SUBCLASS_PARENTS.items())
def test_hierarchy(name, parent):
    cls = getattr(geomcore, name)
    assert issubclass(cls, getattr(geomcore, parent))
    assert issubclass(cls, geomcore.GeomcoreError)
    assert issubclass(cls, ValueError)
    assert cls.__doc__


@pytest.mark.parametrize(
    "catch", [geomcore.ConstructionError, geomcore.GeomcoreError, ValueError]
)
def test_negative_radius_circle(catch):
    with pytest.raises(catch):
        curves.Circle3D(geomcore.Point3D(0, 0, 0), geomcore.Vector3D(0, 0, 1), -1.0)


def test_negative_radius_sphere_is_construction_error():
    with pytest.raises(geomcore.ConstructionError):
        surfaces.Sphere(geomcore.Point3D(0, 0, 0), -1.0)


def test_discretization_criteria_error():
    circle = curves.Circle3D(geomcore.Point3D(0, 0, 0), geomcore.Vector3D(0, 0, 1), 1.0)
    with pytest.raises(geomcore.DiscretizationCriteriaError):
        circle.discretize(-1.0)


def test_frame_construction_error():
    with pytest.raises(geomcore.FrameConstructionError):
        geomcore.Frame3D(
            geomcore.Point3D(0, 0, 0),
            geomcore.Vector3D(0, 0, 0),
            geomcore.Vector3D(1, 0, 0),
        )


def test_trimmed_curve_construction_error():
    line = curves.Line3D.from_two_points(
        geomcore.Point3D(0, 0, 0), geomcore.Point3D(1, 0, 0)
    )
    with pytest.raises(geomcore.TrimmedCurveConstructionError):
        curves.TrimmedCurve3D(line, 1.0, 0.0)
    with pytest.raises(geomcore.ConstructionError):
        curves.TrimmedCurve3D(line, 1.0, 0.0)


def _straight_bspline():
    return curves.BSplineCurve3D.interpolate(
        [
            geomcore.Point3D(0, 0, 0),
            geomcore.Point3D(1, 1, 0),
            geomcore.Point3D(2, 0, 0),
        ],
        2,
    )


def test_knot_insertion_error():
    spline = _straight_bspline()
    with pytest.raises(geomcore.KnotInsertionError):
        spline.insert_knot(1e9, 1)


def test_bspline_split_error():
    spline = _straight_bspline()
    with pytest.raises(geomcore.BSplineSplitError):
        spline.split_at(1e9)


def _cubic_spline():
    return curves.BSplineCurve3D.interpolate(
        [
            geomcore.Point3D(0, 0, 0),
            geomcore.Point3D(1, 1, 0),
            geomcore.Point3D(2, 0, 0),
            geomcore.Point3D(3, 1, 0),
        ],
        3,
    )


def _bilinear_surface():
    poles = [
        [geomcore.Point3D(0, 0, 0), geomcore.Point3D(0, 1, 0)],
        [geomcore.Point3D(1, 0, 0), geomcore.Point3D(1, 1, 0)],
    ]
    return surfaces.BSplineSurface(
        1, 1, poles, [0.0, 1.0], [2, 2], [0.0, 1.0], [2, 2], False, False
    )


def test_knot_removal_error():
    spline = _cubic_spline()
    with pytest.raises(geomcore.KnotRemovalError):
        spline.remove_knot(0.123456, 1, 1e-6)
    with pytest.raises(geomcore.KnotRemovalError):
        spline.remove_knots([0.5], -1.0)


def test_degree_elevation_error():
    spline = _cubic_spline()
    with pytest.raises(geomcore.DegreeElevationError):
        spline.elevate_degree(100)
    with pytest.raises(geomcore.DegreeElevationError):
        _bilinear_surface().elevate_u_degree(100)


def test_bspline_conversion_error():
    hyperbola = curves.Hyperbola3D(
        geomcore.Point3D(0, 0, 0),
        geomcore.Vector3D(0, 0, 1),
        geomcore.Vector3D(1, 0, 0),
        1.0,
        1.0,
    )
    far = curves.TrimmedCurve3D(hyperbola, -800.0, 800.0)
    with pytest.raises(geomcore.BSplineConversionError):
        far.to_bspline()
    plane = surfaces.Plane(geomcore.Point3D(0, 0, 0), geomcore.Vector3D(0, 0, 1))
    with pytest.raises(geomcore.BSplineConversionError):
        plane.to_bspline_patch((1.0, 0.0), (0.0, 1.0))


def test_parametrize_error():
    sphere = surfaces.Sphere(geomcore.Point3D(0, 0, 0), 1.0)
    far_line = curves.Line3D.from_two_points(
        geomcore.Point3D(5, 0, 0), geomcore.Point3D(5, 1, 0)
    )
    with pytest.raises(geomcore.ParametrizeError):
        far_line.parametrize_on(sphere)


def test_surface_iso_error():
    surface = _bilinear_surface()
    with pytest.raises(geomcore.SurfaceIsoError):
        surface.u_iso(float("nan"))
    with pytest.raises(geomcore.SurfaceIsoError):
        surface.v_iso(5.0)
