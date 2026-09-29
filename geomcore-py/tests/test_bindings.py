"""Behavioral tests for the geomcore Python bindings."""

import math

import pytest

import geomcore
from geomcore import Axis3D, Frame3D, Point3D, Transform, Vector3D
from geomcore.curves import BSplineCurve3D, Circle3D, Line2D
from geomcore.surfaces import Cylinder, Sphere


def test_readme_snippet():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    point = circle.eval_point(math.pi / 4)
    assert point.x == pytest.approx(2.0 * math.cos(math.pi / 4))
    assert point.y == pytest.approx(2.0 * math.sin(math.pi / 4))
    assert point.z == pytest.approx(0.0)


def test_submodule_import_forms():
    # Both attribute access and from-import must work.
    assert geomcore.curves.Circle3D is Circle3D
    from geomcore.curves import Circle3D as C2

    assert C2 is Circle3D


def test_eval_points_bulk():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    params = [i / 100 * math.tau for i in range(100)]
    points = circle.eval_points(params)
    assert len(points) == 100
    for u, p in zip(params, points):
        assert p.x == pytest.approx(2.0 * math.cos(u))
        assert p.y == pytest.approx(2.0 * math.sin(u))


def test_constructor_error_raises_value_error():
    with pytest.raises(ValueError) as excinfo:
        Circle3D(Point3D.origin(), Vector3D.z(), -1.0)
    assert str(excinfo.value)


def test_init_constructs():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.radius() == 2.0
    # `new` is no longer a constructor: only __init__ builds objects.
    assert not hasattr(Circle3D, "new")


def test_parametrize_on_cylinder():
    # Coaxial circle on a cylinder of the same radius: the pcurve is the
    # horizontal line v = 0 in (u, v) space.
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    pcurve = circle.parametrize_on(cylinder)
    assert isinstance(pcurve, Line2D)
    origin = pcurve.origin()
    direction = pcurve.direction()
    assert origin.x == pytest.approx(0.0)
    assert origin.y == pytest.approx(0.0)
    assert direction.x == pytest.approx(1.0)
    assert direction.y == pytest.approx(0.0)


def test_parametrize_on_not_analytic_raises():
    sphere = Sphere(Point3D.origin(), 2.0)
    line = geomcore.curves.Line3D(Point3D(2.0, 0.0, 0.0), Vector3D.z())
    with pytest.raises(ValueError):
        line.parametrize_on(sphere)


def test_bspline_curve_degree_one_line():
    curve = BSplineCurve3D(
        1,
        [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [2, 2],
        False,
    )
    mid = curve.eval_point(0.5)
    assert mid.x == pytest.approx(1.0)
    assert mid.y == pytest.approx(0.0)
    assert curve.degree() == 1
    assert not curve.is_periodic()


def test_transform_rotation():
    axis = Axis3D(Point3D.origin(), Vector3D.z())
    rot = Transform.rotation(axis, math.pi / 2)
    p = rot.apply_point(Point3D(1.0, 0.0, 0.0))
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(1.0)


def test_transform_composition():
    t = Transform.translation(Vector3D(1.0, 0.0, 0.0))
    rot = Transform.rotation(Axis3D(Point3D.origin(), Vector3D.z()), math.pi / 2)
    composed = t.then(rot)
    p = composed.apply_point(Point3D.origin())
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(1.0)


def test_sphere_parameters_round_trip():
    sphere = Sphere(Point3D.origin(), 3.0)
    p = sphere.eval_point(0.7, 0.4)
    u, v = sphere.parameters_of(p)
    assert u == pytest.approx(0.7)
    assert v == pytest.approx(0.4)


def test_frame3_accessors():
    frame = Frame3D(Point3D.origin(), Vector3D.z(), Vector3D.x())
    assert frame.z_direction().components() == pytest.approx((0.0, 0.0, 1.0))
    assert frame.x_direction().components() == pytest.approx((1.0, 0.0, 0.0))


def test_docstrings_present():
    assert Circle3D.__doc__
    assert Circle3D.eval_point.__doc__
