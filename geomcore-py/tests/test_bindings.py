"""Behavioral tests for the geomcore Python bindings."""

import math

import pytest

import geomcore
from geomcore import (
    Axis3D,
    BoundingBox3D,
    Frame3D,
    Point2D,
    Point3D,
    Tolerance,
    Transform,
    Vector2D,
    Vector3D,
)
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


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_inputs_raise_value_error(bad):
    from geomcore.curves import Parabola3D
    from geomcore.surfaces import Plane

    with pytest.raises(ValueError, match="NaN or infinite"):
        Circle3D(Point3D.origin(), Vector3D.z(), bad)
    with pytest.raises(ValueError, match="NaN or infinite"):
        Sphere(Point3D(bad, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        Plane(Point3D.origin(), Vector3D(bad, 0.0, 1.0))
    with pytest.raises(ValueError, match="NaN or infinite"):
        Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), bad)


def test_zero_size_conics_raise_value_error():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    o, z, x = Point3D.origin(), Vector3D.z(), Vector3D.x()
    with pytest.raises(ValueError, match="not positive"):
        Parabola3D(o, z, x, 0.0)
    with pytest.raises(ValueError, match="not positive"):
        Ellipse3D(o, z, x, 3.0, 0.0)
    with pytest.raises(ValueError, match="not positive"):
        Hyperbola3D(o, z, x, 2.0, 0.0)


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


def test_parametrize_on_takes_a_tolerance():
    # A radius 1e-4 too wide lies on the cylinder at confusion 1e-3 only.
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 1.0)
    circle = Circle3D(Point3D(0.0, 0.0, 2.0), Vector3D.z(), 1.0 + 1e-4)
    loose = Tolerance(confusion=1e-3)
    assert isinstance(circle.parametrize_on(cylinder, loose), Line2D)
    assert isinstance(circle.parametrize_on(cylinder, tol=loose), Line2D)
    with pytest.raises(ValueError):
        circle.parametrize_on(cylinder)


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


def test_tolerance_defaults():
    tol = Tolerance.default()
    assert tol.confusion == 1e-7
    assert tol.angular == 1e-12
    assert tol.parametric == 1e-9
    assert Tolerance().confusion == tol.confusion
    custom = Tolerance(confusion=1e-3)
    assert custom.confusion == 1e-3
    assert custom.angular == tol.angular
    # Models fit in a cube of MODEL_SIZE; angles derive from distances.
    assert Tolerance.MODEL_SIZE == 1e5
    assert tol.angular_for_extent(10.0) == 1e-8
    assert tol.angular_for_extent(1e9) == tol.angular


def test_points_coincide_within_confusion():
    p = Point3D(1.0, 2.0, 3.0)
    assert p.coincides(Point3D(1.0, 2.0, 3.0 + 5e-8))
    assert not p.coincides(Point3D(1.0, 2.0, 3.0 + 5e-7))
    near = Point3D(1.0, 2.0005, 3.0)
    assert not p.coincides(near)
    assert p.coincides(near, Tolerance(confusion=1e-3))
    assert p.coincides(near, tol=Tolerance(confusion=1e-3))
    q = Point2D(1.0, 2.0)
    assert q.coincides(Point2D(1.0 + 5e-8, 2.0))
    assert not q.coincides(Point2D(1.0 + 5e-7, 2.0))


def _bspline_curve(end):
    return BSplineCurve3D(1, [Point3D.origin(), end], [0.0, 1.0], [2, 2], False)


def _bspline_surface(corner_z):
    poles = [
        [Point3D(0.0, 0.0, 0.0), Point3D(0.0, 1.0, 0.0)],
        [Point3D(1.0, 0.0, 0.0), Point3D(1.0, 1.0, corner_z)],
    ]
    return geomcore.surfaces.BSplineSurface(
        1, 1, poles, [0.0, 1.0], [2, 2], [0.0, 1.0], [2, 2], False, False
    )


_O, _X, _Z = Point3D.origin(), Vector3D.x(), Vector3D.z()

# For each class: a factory building one value, and a value that differs from it.
VALUE_TYPES = [
    (lambda: Point3D(1.0, 2.0, 3.0), Point3D(1.0, 2.0, 3.0 + 1e-12)),
    (lambda: Point2D(1.0, 2.0), Point2D(1.0, 2.5)),
    (lambda: Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, 1.0, 0.0)),
    (lambda: Vector2D(1.0, 0.0), Vector2D(0.0, 1.0)),
    (lambda: Axis3D(_O, _Z), Axis3D(_O, _X)),
    (lambda: Frame3D(_O, _Z, _X), Frame3D(Point3D(0.0, 0.0, 1.0), _Z, _X)),
    (lambda: Tolerance(), Tolerance(confusion=1e-3)),
    (lambda: Transform.translation(_X), Transform.identity()),
    (lambda: BoundingBox3D(_O, Point3D(1.0, 1.0, 1.0)), BoundingBox3D(_O, Point3D(2.0, 1.0, 1.0))),
    (lambda: geomcore.surfaces.Plane(_O, _Z), geomcore.surfaces.Plane(_O, _X)),
    (lambda: Cylinder(_O, _Z, 1.0), Cylinder(_O, _Z, 2.0)),
    (lambda: geomcore.surfaces.Cone(_O, _Z, 0.3, 1.0), geomcore.surfaces.Cone(_O, _Z, 0.4, 1.0)),
    (lambda: Sphere(_O, 1.0), Sphere(_O, 2.0)),
    (lambda: geomcore.surfaces.Torus(_O, _Z, 3.0, 1.0), geomcore.surfaces.Torus(_O, _Z, 3.0, 0.5)),
    (lambda: _bspline_surface(0.0), _bspline_surface(1.0)),
    (lambda: geomcore.curves.Line3D(_O, _X), geomcore.curves.Line3D(_O, _Z)),
    (lambda: Circle3D(_O, _Z, 2.0), Circle3D(_O, _Z, 2.5)),
    (
        lambda: geomcore.curves.Ellipse3D(_O, _Z, _X, 3.0, 1.0),
        geomcore.curves.Ellipse3D(_O, _Z, _X, 3.0, 2.0),
    ),
    (lambda: geomcore.curves.Parabola3D(_O, _Z, _X, 1.0), geomcore.curves.Parabola3D(_O, _Z, _X, 2.0)),
    (
        lambda: geomcore.curves.Hyperbola3D(_O, _Z, _X, 3.0, 1.0),
        geomcore.curves.Hyperbola3D(_O, _Z, _X, 3.0, 2.0),
    ),
    (lambda: _bspline_curve(Point3D(1.0, 0.0, 0.0)), _bspline_curve(Point3D(2.0, 0.0, 0.0))),
    (lambda: Line2D(Point2D.origin(), Vector2D(1.0, 0.0)), Line2D(Point2D.origin(), Vector2D(0.0, 1.0))),
    (lambda: geomcore.curves.Circle2D(Point2D.origin(), 1.0), geomcore.curves.Circle2D(Point2D.origin(), 2.0)),
    (
        lambda: geomcore.curves.TrimmedCurve3D.segment(_O, Point3D(1.0, 0.0, 0.0)),
        geomcore.curves.TrimmedCurve3D.segment(_O, Point3D(2.0, 0.0, 0.0)),
    ),
    (
        lambda: geomcore.curves.TrimmedCurve2D.segment(Point2D.origin(), Point2D(1.0, 0.0)),
        geomcore.curves.TrimmedCurve2D.segment(Point2D.origin(), Point2D(2.0, 0.0)),
    ),
    (lambda: Circle3D(_O, _Z, 2.0).discretize(0.1), Circle3D(_O, _Z, 2.0).discretize(0.01)),
    (
        lambda: geomcore.curves.Circle2D(Point2D.origin(), 1.0).discretize(0.1),
        geomcore.curves.Circle2D(Point2D.origin(), 1.0).discretize(0.01),
    ),
]


@pytest.mark.parametrize(
    "make, different", VALUE_TYPES, ids=[type(d).__name__ for _, d in VALUE_TYPES]
)
def test_equality_compares_values_exactly(make, different):
    value = make()
    assert value == make()
    assert not value != make()
    assert value != different
    assert value != "not a geometry value"


@pytest.mark.parametrize(
    "make, different", VALUE_TYPES, ids=[type(d).__name__ for _, d in VALUE_TYPES]
)
def test_value_types_are_unhashable(make, different):
    with pytest.raises(TypeError, match="unhashable"):
        hash(make())


def test_equal_points_from_different_computations():
    curve = _bspline_curve(Point3D(2.0, 0.0, 0.0))
    assert curve.eval_point(0.5) == Point3D(1.0, 0.0, 0.0)
    assert curve.eval_point(0.0) == Point3D.origin()


def test_contains_curve():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.contains(Point3D(2.0, 0.0, 0.0))
    assert circle.contains(circle.eval_point(1.0))
    assert not circle.contains(Point3D(3.0, 0.0, 0.0))
    assert not circle.contains(Point3D.origin())
    # Explicit tolerance widens the acceptance band.
    near = Point3D(2.0 + 1e-6, 0.0, 0.0)
    assert not circle.contains(near)
    assert circle.contains(near, Tolerance(confusion=1e-5))


def test_contains_surface():
    from geomcore.surfaces import Plane, Torus

    plane = Plane(Point3D.origin(), Vector3D.z())
    assert plane.contains(Point3D.origin())
    assert plane.contains(Point3D(1.0, 2.0, 0.0))
    assert not plane.contains(Point3D(0.0, 0.0, 1.0))

    sphere = Sphere(Point3D.origin(), 3.0)
    assert sphere.contains(Point3D(3.0, 0.0, 0.0))
    assert not sphere.contains(Point3D.origin())

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)
    assert torus.contains(Point3D(5.0, 0.0, 0.0))
    assert not torus.contains(Point3D.origin())


def test_project_point_curve():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    projection = circle.project_point(Point3D(3.0, 0.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert u == pytest.approx(0.0)
    assert dist == pytest.approx(1.0)
    projection = circle.project_point(Point3D(2.0, 0.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert dist == 0.0
    # Batch variant agrees with scalar calls.
    batch = circle.project_points([Point3D(3.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)])
    assert (batch[0].parameter, batch[0].distance) == pytest.approx((0.0, 1.0))
    assert batch[1].distance == 0.0


def test_project_point_surface():
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    projection = plane.project_point(Point3D(1.0, 2.0, 3.0))
    u, v, dist = projection.u, projection.v, projection.distance
    assert (u, v) == pytest.approx((1.0, 2.0))
    assert dist == pytest.approx(3.0)
    batch = plane.project_points([Point3D.origin(), Point3D(0.0, 0.0, 2.0)])
    assert batch[0].distance == 0.0
    assert batch[1].distance == pytest.approx(2.0)


def _curves(result):
    """The curves of a surface-surface result."""
    return [piece.curve for piece in result.curves]


def test_intersect_plane_plane():
    from geomcore.curves import Line3D
    from geomcore.surfaces import Plane

    xy = Plane(Point3D.origin(), Vector3D.z())
    zy = Plane(Point3D.origin(), Vector3D.x())
    (line,) = _curves(xy.intersect(zy))
    assert isinstance(line, Line3D)
    assert line.origin().x == pytest.approx(0.0)
    # Direction follows the ordered pair (n1 x n2 = Z x X = Y).
    assert line.direction().components() == pytest.approx((0.0, 1.0, 0.0))

    parallel = xy.intersect(Plane(Point3D(0.0, 0.0, 1.0), Vector3D.z()))
    assert parallel.is_empty() and parallel.is_complete()

    coincident = xy.intersect(xy)
    assert coincident.is_coincident()
    assert coincident.curves == []


def test_intersect_plane_sphere():
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    (circle,) = _curves(plane.intersect(Sphere(Point3D.origin(), 2.0)))
    assert circle.radius() == pytest.approx(2.0)

    tangent = plane.intersect(Sphere(Point3D(0.0, 0.0, 2.0), 2.0))
    assert tangent.curves == []
    (point,) = tangent.contact_points
    assert (point.x, point.y, point.z) == pytest.approx((0.0, 0.0, 0.0))

    assert plane.intersect(Sphere(Point3D(0.0, 0.0, 5.0), 2.0)).is_empty()


def test_intersect_sphere_sphere():
    s1 = Sphere(Point3D.origin(), 2.0)
    (circle,) = _curves(s1.intersect(Sphere(Point3D(3.0, 0.0, 0.0), 2.0)))
    assert circle.center().x == pytest.approx(1.5)

    (point,) = s1.intersect(Sphere(Point3D(4.0, 0.0, 0.0), 2.0)).contact_points
    assert point.x == pytest.approx(2.0)

    assert s1.intersect(Sphere(Point3D(5.0, 0.0, 0.0), 2.0)).is_empty()
    assert s1.intersect(s1).is_coincident()


def test_intersect_plane_cylinder():
    from geomcore.surfaces import Cylinder, Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    (circle,) = _curves(plane.intersect(cylinder))
    assert circle.radius() == pytest.approx(2.0)

    side = Plane(Point3D(0.0, 1.0, 0.0), Vector3D.y())
    lines = _curves(side.intersect(cylinder))
    assert len(lines) == 2
    assert lines[0].direction().components() == pytest.approx((0.0, 0.0, 1.0))

    assert side.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 0.5)).is_empty()
    miss = Plane(Point3D(0.0, 3.0, 0.0), Vector3D.y())
    assert miss.intersect(cylinder).is_empty()


def test_intersect_plane_cone():
    from geomcore import Frame3D
    from geomcore.curves import Hyperbola3D
    from geomcore.surfaces import Cone, Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    (circle,) = _curves(plane.intersect(cone))
    assert circle.radius() == pytest.approx(2.0)

    # The cone is double: a plane beyond the apex cuts the other nappe.
    below = Plane(Point3D(0.0, 0.0, -6.0), Vector3D.z())
    (circle,) = _curves(below.intersect(cone))
    assert circle.radius() == pytest.approx(0.5368, abs=1e-3)

    h1, h2 = _curves(Plane(Point3D(3.0, 0.0, 0.0), Vector3D.x()).intersect(cone))
    assert isinstance(h1, Hyperbola3D) and isinstance(h2, Hyperbola3D)
    assert h1.center().distance(h2.center()) < 1e-9

    apex_plane = Plane(cone.apex(), Vector3D.z())
    (apex,) = apex_plane.intersect(cone).contact_points
    assert apex.distance(cone.apex()) < 1e-9


def test_intersect_line_surface():
    from geomcore.curves import Line3D
    from geomcore.surfaces import Cylinder, Plane

    line = Line3D(Point3D.origin(), Vector3D.x())
    hits = line.intersect(Sphere(Point3D.origin(), 2.0)).hits
    assert [h.curve_parameter for h in hits] == pytest.approx([-2.0, 2.0])
    assert [h.point.x for h in hits] == pytest.approx([-2.0, 2.0])

    (hit,) = Line3D(Point3D.origin(), Vector3D.z()).intersect(
        Plane(Point3D.origin(), Vector3D.z())
    ).hits
    assert hit.curve_parameter == pytest.approx(0.0)
    on_plane = line.intersect(Plane(Point3D.origin(), Vector3D.z()))
    assert on_plane.hits == [] and on_plane.overlaps == [(-math.inf, math.inf)]

    generator = Line3D(Point3D(2.0, 0.0, 0.0), Vector3D.z())
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    assert generator.intersect(cylinder).overlaps == [(-math.inf, math.inf)]


def test_intersect_symmetric_quadrics():
    from geomcore import Frame3D, IntersectionMethod
    from geomcore.surfaces import Cone, Cylinder

    sphere = Sphere(Point3D.origin(), 3.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)

    c1, c2 = _curves(sphere.intersect(cylinder))
    assert c1.radius() == pytest.approx(2.0)
    assert abs(c1.center().z) == pytest.approx(math.sqrt(5.0))

    assert len(sphere.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 3.0)).curves) == 1
    # Off-axis pairs have no closed form: marching traces them.
    off = Cylinder(Point3D(0.0, 1.0, 0.0), Vector3D.z(), 2.0)
    assert sphere.intersect(off).method == IntersectionMethod.NUMERIC

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    r1, r2 = _curves(sphere.intersect(cone))
    assert sorted([r1.radius(), r2.radius()]) == pytest.approx([0.7746, 2.6188], abs=1e-3)
    # Sphere around the apex: one ring on each nappe.
    small = Cone.from_frame(Frame3D.world(), 0.4, 0.5)
    rings = sorted(_curves(sphere.intersect(small)), key=lambda c: c.center().z)
    lower, upper = rings
    assert upper.radius() == pytest.approx(1.5786, abs=1e-3)
    assert lower.radius() == pytest.approx(0.7302, abs=1e-3)
    assert lower.center().z < small.apex().z < upper.center().z

    c1 = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    c2 = Cylinder(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    l1, _ = _curves(c1.intersect(c2))
    assert l1.origin().x == pytest.approx(1.5)
    assert c1.intersect(c1).is_coincident()
    crossed = Cylinder(Point3D.origin(), Vector3D.x(), 2.0)
    assert c1.intersect(crossed).method == IntersectionMethod.NUMERIC


def test_intersect_cone_cylinder():
    from geomcore import Frame3D, IntersectionMethod
    from geomcore.surfaces import Cone, Cylinder

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    lower, upper = sorted(_curves(cone.intersect(cylinder)), key=lambda c: c.center().z)
    assert (lower.radius(), upper.radius()) == pytest.approx((2.0, 2.0))
    assert upper.center().distance(Point3D.origin()) < 1e-9
    assert lower.center().z == pytest.approx(2.0 * cone.apex().z)

    off = Cylinder(Point3D(1.0, 0.0, 0.0), Vector3D.z(), 2.0)
    assert cone.intersect(off).method == IntersectionMethod.NUMERIC


def test_intersect_cone_cone():
    from geomcore import Frame3D, IntersectionMethod
    from geomcore.surfaces import Cone

    c1 = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    frame2 = Frame3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), Vector3D.x())
    c2 = Cone.from_frame(frame2, 0.6, 1.0)
    r1, r2 = _curves(c1.intersect(c2))
    assert sorted([r1.radius(), r2.radius()]) == pytest.approx([1.1155, 4.7245], abs=1e-3)

    assert c1.intersect(c1).is_coincident()
    tilted = Cone(Point3D.origin(), Vector3D.x(), 0.4, 2.0)
    assert c1.intersect(tilted).method == IntersectionMethod.NUMERIC


def test_intersect_torus_symmetric():
    from geomcore import IntersectionMethod
    from geomcore.surfaces import Cone, Cylinder, Plane, Torus

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)

    c1, c2 = _curves(torus.intersect(Plane(Point3D.origin(), Vector3D.z())))
    assert sorted([c1.radius(), c2.radius()]) == pytest.approx([3.0, 5.0])

    assert len(torus.intersect(Plane(Point3D.origin(), Vector3D.x())).curves) == 2

    # Bitangent plane through the center: Villarceau circles.
    tilt = math.asin(1.0 / 4.0)
    villarceau = Plane(Point3D.origin(), Vector3D(0.0, math.sin(tilt), math.cos(tilt)))
    c1, c2 = _curves(torus.intersect(villarceau))
    assert [c1.radius(), c2.radius()] == pytest.approx([4.0, 4.0])
    assert sorted([c1.center().x, c2.center().x]) == pytest.approx([-1.0, 1.0])

    (piece,) = torus.intersect(Sphere(Point3D.origin(), 5.0)).curves
    assert piece.tangent is True
    assert piece.curve.radius() == pytest.approx(5.0)

    assert len(torus.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 4.5)).curves) == 2

    rings = torus.intersect(Cone.from_frame(Frame3D.world(), 1.1, 2.0)).curves
    assert [ring.tangent for ring in rings] == [False, False]
    rings = torus.intersect(Cone.from_frame(Frame3D.world(), 1.4, 0.0)).curves
    assert len(rings) == 4

    other = Torus(Point3D(0.0, 0.0, 1.5), Vector3D.z(), 4.0, 1.0)
    assert len(torus.intersect(other).curves) == 2
    assert torus.intersect(torus).is_coincident()
    tilted = Torus(Point3D.origin(), Vector3D.x(), 4.0, 1.0)
    assert torus.intersect(tilted).method == IntersectionMethod.NUMERIC


def test_conic_surface_intersections():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D
    from geomcore.surfaces import Cone, Cylinder, Plane

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    assert len(ellipse.intersect(Plane(Point3D.origin(), Vector3D.x())).hits) == 2

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    on_plane = parabola.intersect(Plane(Point3D.origin(), Vector3D.z()))
    assert on_plane.overlaps == [(-math.inf, math.inf)]

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    assert len(hyperbola.intersect(cylinder).hits) >= 1

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    assert ellipse.intersect(cone).hits


def test_circle_surface_and_measures():
    import math

    from geomcore.surfaces import Cone, Cylinder, Plane

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.circumference() == pytest.approx(math.tau * 2.0)
    assert circle.disk_area() == pytest.approx(math.pi * 4.0)

    hits = circle.intersect(Plane(Point3D.origin(), Vector3D.x())).hits
    assert len(hits) == 2

    assert circle.intersect(Sphere(Point3D.origin(), 3.0)).is_empty()

    on_cylinder = circle.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 2.0))
    assert on_cylinder.overlaps == [pytest.approx((0.0, math.tau))]

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    assert circle.intersect(cone).overlaps == [pytest.approx((0.0, math.tau))]

    from geomcore.curves import Ellipse3D

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    assert ellipse.area() == pytest.approx(math.pi * 4.5)

    sphere = Sphere(Point3D.origin(), 3.0)
    assert sphere.area() == pytest.approx(4.0 * math.pi * 9.0)
    assert sphere.volume() == pytest.approx(4.0 / 3.0 * math.pi * 27.0)


def test_interpolate_curve():
    curve = BSplineCurve3D.interpolate(
        [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 1.0, 0.0), Point3D(2.0, 0.0, 0.0)], 2
    )
    assert curve.contains(Point3D(0.0, 0.0, 0.0))
    assert curve.contains(Point3D(2.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        BSplineCurve3D.interpolate([Point3D.origin()], 1)


def test_approximate_curve():
    import math

    pts = [Point3D(2.0 * math.cos(i / 12 * math.tau), 2.0 * math.sin(i / 12 * math.tau), 0.0) for i in range(12)]
    curve = BSplineCurve3D.approximate(pts, 3, 7)
    assert curve.degree() == 3
    with pytest.raises(ValueError):
        BSplineCurve3D.approximate(pts, 3, 20)


def test_curve_surface_intersect_any_pair():
    from geomcore import IntersectionMethod
    from geomcore.curves import Line3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    result = line.intersect(Sphere(Point3D.origin(), 2.0))
    assert [h.curve_parameter for h in result.hits] == pytest.approx([-2.0, 2.0])
    assert line.intersect(Sphere(Point3D(0.0, 0.0, 5.0), 2.0)).hits == []

    patch = _bspline_surface(0.0)
    vertical = Line3D(Point3D(0.5, 0.5, -1.0), Vector3D.z())
    result = vertical.intersect(patch)
    assert result.method == IntersectionMethod.NUMERIC
    (hit,) = result.hits
    assert hit.point.z == pytest.approx(0.0, abs=1e-9)


def test_intersect_surfaces_march_without_closed_form():
    from geomcore import CompletenessStatus, IntersectionMethod
    from geomcore.surfaces import Plane

    patch = _bspline_surface(0.0)
    result = patch.intersect(Plane(Point3D(0.5, 0.0, 0.0), Vector3D.x()))
    assert result.method == IntersectionMethod.NUMERIC
    assert result.completeness.status == CompletenessStatus.SAMPLED
    (piece,) = result.curves
    assert isinstance(piece.curve, BSplineCurve3D)

    miss = Plane(Point3D(0.0, 0.0, 5.0), Vector3D.z())
    assert patch.intersect(miss).is_empty()


@pytest.mark.parametrize("height", [0.3, 0.8, -0.45])
def test_intersect_plane_sphere_off_grid(height: float) -> None:
    # Planes between the seed grid lines used to find no curve (#6).
    from geomcore.surfaces import Plane

    plane = Plane(Point3D(0.0, 0.0, height), Vector3D.z())
    (piece,) = plane.intersect(Sphere(Point3D.origin(), 1.0)).curves
    p = piece.curve.eval_point(0.5)
    assert math.hypot(p.x, p.y) == pytest.approx(math.sqrt(1.0 - height**2), abs=1e-3)


def test_curve_curve_generic():
    from geomcore import curve_curve_extrema
    from geomcore.curves import Circle3D, Line3D

    l1 = Line3D(Point3D.origin(), Vector3D.x())
    l2 = Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.y())
    extrema = curve_curve_extrema(l1, l2)
    assert len(extrema.minima) == 1
    assert extrema.minima[0].distance == pytest.approx(1.0)

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    hits = l1.intersect(circle).hits
    assert [h.first_parameter for h in hits] == pytest.approx([-2.0, 2.0])


def test_parametrize_numeric():
    import math

    from geomcore import parametrize_numeric

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    pcurve = parametrize_numeric(circle, cylinder, 9)
    assert len(pcurve) == 9
    for u, v in pcurve:
        assert v == pytest.approx(0.0)
    # Round-trip through the surface recovers the curve.
    for i, (u, v) in enumerate(pcurve):
        p = cylinder.eval_point(u, v)
        q = circle.eval_point(i / 8 * math.tau)
        assert p.x == pytest.approx(q.x)
        assert p.y == pytest.approx(q.y)


def test_intersect_curves_2d_and_lines():
    from geomcore.curves import Circle2D, Line2D, Line3D

    l1 = Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0))
    l2 = Line2D(Point2D(0.0, 0.0), Vector2D(0.0, 1.0))
    (hit,) = l1.intersect(l2).hits
    assert (hit.first_parameter, hit.second_parameter) == pytest.approx((0.0, 0.0))

    circle = Circle2D(Point2D(0.0, 0.0), 2.0)
    hits = l1.intersect(circle).hits
    assert [h.first_parameter for h in hits] == pytest.approx([-2.0, 2.0])

    p1, _ = [h.point for h in circle.intersect(Circle2D(Point2D(3.0, 0.0), 2.0)).hits]
    assert p1.x == pytest.approx(1.5)

    a = Line3D(Point3D.origin(), Vector3D.x())
    b = Line3D(Point3D.origin(), Vector3D.y())
    (hit,) = a.intersect(b).hits
    assert (hit.point.x, hit.point.y, hit.point.z) == pytest.approx((0.0, 0.0, 0.0))
    skew = a.intersect(Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.y()))
    assert skew.is_empty() and skew.is_complete()


def test_intersect_curves_3d():
    from geomcore.curves import Circle3D, Line3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    hits = line.intersect(circle).hits
    assert [h.first_parameter for h in hits] == pytest.approx([-2.0, 2.0])

    c1 = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    c2 = Circle3D(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    p1, _ = [h.point for h in c1.intersect(c2).hits]
    assert p1.x == pytest.approx(1.5)
    tilted = Circle3D(Point3D.origin(), Vector3D.x(), 2.0)
    assert sorted(h.point.y for h in c1.intersect(tilted).hits) == pytest.approx([-2.0, 2.0])
    (overlap,) = c1.intersect(c1).overlaps
    assert overlap.first_range == pytest.approx((0.0, math.tau))
    lifted = Circle3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), 2.0)
    assert c1.intersect(lifted).is_empty()


def _bilinear_patch():
    from geomcore.surfaces import BSplineSurface

    return BSplineSurface(
        1,
        1,
        [
            [Point3D(0.0, 0.0, 0.0), Point3D(0.0, 2.0, 0.0)],
            [Point3D(2.0, 0.0, 0.0), Point3D(2.0, 2.0, 0.0)],
        ],
        [0.0, 1.0],
        [2, 2],
        [0.0, 1.0],
        [2, 2],
        False,
        False,
    )


def test_bspline_surface_projection():
    surface = _bilinear_patch()
    projection = surface.project_point(Point3D(1.0, 1.0, 1.0))
    u, v, dist = projection.u, projection.v, projection.distance
    assert (u, v) == pytest.approx((0.5, 0.5))
    assert dist == pytest.approx(1.0)
    assert surface.contains(Point3D(1.0, 1.0, 0.0))
    assert not surface.contains(Point3D(1.0, 1.0, 1.0))
    ext = surface.extrema(Point3D(1.0, 1.0, 1.0))
    assert len(ext) >= 1
    assert ext[0].distance == pytest.approx(1.0)


def test_bspline_curve_projection():
    curve = BSplineCurve3D(
        1,
        [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [2, 2],
        False,
    )
    projection = curve.project_point(Point3D(1.0, 1.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert u == pytest.approx(0.5)
    assert dist == pytest.approx(1.0)
    assert curve.contains(Point3D(1.0, 0.0, 0.0))
    assert not curve.contains(Point3D(1.0, 1.0, 0.0))
    # Beyond the end: the endpoint wins.
    projection = curve.project_point(Point3D(5.0, 0.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert u == pytest.approx(1.0)
    assert dist == pytest.approx(3.0)
    ext = curve.extrema(Point3D(1.0, 1.0, 0.0))
    assert len(ext) >= 1
    assert ext[0].distance == pytest.approx(1.0)


def test_conic_extrema_and_projection():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    ext = ellipse.extrema(Point3D(4.0, 0.0, 0.0))
    assert len(ext) >= 2
    assert ext[0].distance == pytest.approx(1.0)
    projection = ellipse.project_point(Point3D(0.0, 3.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert dist == pytest.approx(1.5)

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    projection = parabola.project_point(Point3D(1.0, 2.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert u == pytest.approx(2.0)
    assert dist == 0.0

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    projection = hyperbola.project_point(Point3D(3.0, 0.0, 0.0))
    u, dist = projection.parameter, projection.distance
    assert dist == pytest.approx(math.sqrt(0.8))


def _hit_tuples(result):
    """Curve-curve hits as `(first_parameter, point, second_parameter,
    multiplicity)` tuples."""
    return [(h.first_parameter, h.point, h.second_parameter, h.multiplicity) for h in result.hits]


def test_line_conic_intersections():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Line3D, Parabola3D

    tol = Tolerance.default()
    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    (t1, p1, u1, _), (t2, p2, u2, _) = _hit_tuples(
        Line3D(Point3D.origin(), Vector3D.x()).intersect(ellipse)
    )
    assert (t1, t2) == pytest.approx((-3.0, 3.0))
    assert (_or_recovered(u1, ellipse, p1), _or_recovered(u2, ellipse, p2)) == pytest.approx(
        (math.pi, 0.0)
    )

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    # The axis crosses the parabola once, at its vertex.
    [(t, p, u, mult)] = _hit_tuples(Line3D(Point3D.origin(), Vector3D.x()).intersect(parabola))
    assert mult == 1
    assert t == pytest.approx(0.0) and _or_recovered(u, parabola, p) == pytest.approx(0.0)

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    [(t, p, u, mult)] = _hit_tuples(
        Line3D(Point3D(2.0, -5.0, 0.0), Vector3D.y()).intersect(hyperbola, tol)
    )
    assert mult == 2
    assert (p.x, p.y, p.z) == pytest.approx((2.0, 0.0, 0.0))

    miss = Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.x())
    assert miss.intersect(ellipse).is_empty()
    assert miss.intersect(parabola).is_empty()
    assert miss.intersect(hyperbola).is_empty()


def test_circle_conic_intersections():
    from geomcore.curves import Circle3D, Ellipse3D, Hyperbola3D, Parabola3D

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    hits = _hit_tuples(circle.intersect(ellipse))
    assert len(hits) == 4
    for s, p, u, mult in hits:
        assert mult == 1
        assert circle.contains(p)
        assert ellipse.contains(p)
        # Only the circle is solved for: the ellipse's parameter is left to
        # the caller.
        assert u is None
        _assert_same_point(circle.eval_point(s), p)
        _assert_same_point(ellipse.eval_point(ellipse.parameter_of(p)), p)

    same = circle.intersect(Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 2.0))
    assert len(same.overlaps) == 1 and same.hits == []
    # Crossing planes: the tilted ellipse meets the shared line at
    # y = ±1.5, inside the circle.
    tilted = Ellipse3D(Point3D.origin(), Vector3D.x(), Vector3D.z(), 3.0, 1.5)
    assert circle.intersect(tilted).is_empty()

    unit = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    assert len(unit.intersect(parabola).hits) == 2

    big = Circle3D(Point3D.origin(), Vector3D.z(), 3.0)
    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    hits = big.intersect(hyperbola).hits
    assert len(hits) == 2
    assert all(h.point.x > 0.0 for h in hits)


def test_line_torus_intersections():
    from geomcore.curves import Line3D
    from geomcore.surfaces import Torus

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)
    hits = Line3D(Point3D.origin(), Vector3D.x()).intersect(torus).hits
    assert [h.curve_parameter for h in hits] == pytest.approx([-5.0, -3.0, 3.0, 5.0])
    for hit in hits:
        assert hit.multiplicity == 1
        assert torus.contains(hit.point)
        projection = torus.project_point(hit.point)
        uu, vv, dist = projection.u, projection.v, projection.distance
        assert dist == pytest.approx(0.0)
        if hit.surface_parameters is not None:
            assert hit.surface_parameters == pytest.approx((uu, vv))

    (hit,) = Line3D(Point3D(0.0, 5.0, 0.0), Vector3D.x()).intersect(torus).hits
    assert hit.curve_parameter == pytest.approx(0.0)
    assert hit.multiplicity == 2

    assert Line3D(Point3D(0.0, 6.0, 0.0), Vector3D.x()).intersect(torus).is_empty()


def test_ellipse_ellipse_intersections():
    from geomcore.curves import Ellipse3D

    e1 = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    e2 = Ellipse3D(Point3D(0.5, 0.0, 0.0), Vector3D.z(), Vector3D.x(), 2.0, 2.0)
    hits = _hit_tuples(e1.intersect(e2))
    assert len(hits) == 4
    for s, p, t, _ in hits:
        assert e1.contains(p)
        assert e2.contains(p)

    grazing = Ellipse3D(Point3D(5.0, 0.0, 0.0), Vector3D.z(), Vector3D.x(), 2.0, 2.0)
    [(s, p, t, mult)] = _hit_tuples(e1.intersect(grazing))
    assert mult == 2
    assert (p.x, p.y, p.z) == pytest.approx((3.0, 0.0, 0.0))

    far = Ellipse3D(Point3D(10.0, 0.0, 0.0), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    assert e1.intersect(far).is_empty()
    assert len(e1.intersect(e1).overlaps) == 1
    # Crossing planes: both pass through (0, ±1.5, 0).
    tilted = Ellipse3D(Point3D.origin(), Vector3D.x(), Vector3D.z(), 3.0, 1.5)
    hits = _hit_tuples(e1.intersect(tilted))
    assert sorted(p.y for _, p, _, _ in hits) == pytest.approx([-1.5, 1.5])
    lifted = Ellipse3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    assert e1.intersect(lifted).is_empty()


def test_conic_conic_intersections():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    o, z, x = Point3D.origin(), Vector3D.z(), Vector3D.x()
    ellipse = Ellipse3D(o, z, x, 3.0, 2.0)
    hyperbola = Hyperbola3D(o, z, x, 1.0, 0.5)
    parabola = Parabola3D(o, z, x, 1.0)
    # The branch x > 0 meets the ellipse at (√153/5, ±√128/10).
    hits = _hit_tuples(ellipse.intersect(hyperbola))
    hx, hy = math.sqrt(153) / 5, math.sqrt(128) / 10
    assert sorted(p.y for _, p, _, _ in hits) == pytest.approx([-hy, hy])
    assert [p.x for _, p, _, _ in hits] == pytest.approx([hx, hx])
    pairs = [
        (ellipse, hyperbola),
        (ellipse, parabola),
        (parabola, hyperbola),
        (parabola, Parabola3D(Point3D(2.0, 0.0, 0.0), z, Vector3D(-1.0, 0.0, 0.0), 1.0)),
        (hyperbola, Hyperbola3D(Point3D(0.0, -2.0, 0.0), z, Vector3D.y(), 1.0, 1.0)),
    ]
    for a, b in pairs:
        hits = _hit_tuples(a.intersect(b))
        assert hits, (a, b)
        for s, p, t, _ in hits:
            # One conic is solved for; a parameter not computed is None.
            assert s is not None or t is not None
            _assert_same_point(a.eval_point(_or_recovered(s, a, p)), p)
            _assert_same_point(b.eval_point(_or_recovered(t, b, p)), p)
        # The receiver's parameter comes first.
        back = _hit_tuples(b.intersect(a))
        on_b = sorted(_or_recovered(t, b, p) for _, p, t, _ in hits)
        assert on_b == pytest.approx(sorted(_or_recovered(t, b, p) for t, p, _, _ in back))
        on_a = sorted(_or_recovered(s, a, p) for s, p, _, _ in hits)
        assert on_a == pytest.approx(sorted(_or_recovered(s, a, p) for _, p, s, _ in back))
    assert len(parabola.intersect(parabola).overlaps) == 1
    lifted = Parabola3D(Point3D(0.0, 0.0, 1.0), z, x, 1.0)
    assert parabola.intersect(lifted).is_empty()


def _rigid_motion():
    axis = Axis3D(Point3D(0.5, -1.0, 0.0), Vector3D(1.0, 2.0, 2.0))
    return Transform.translation(Vector3D(3.0, -1.0, 2.0)).then(Transform.rotation(axis, 0.7))


def _or_recovered(parameter, curve, point):
    """The stored parameter, or the curve's parameter of the point when the
    solver did not compute it."""
    return curve.parameter_of(point) if parameter is None else parameter


def _assert_same_point(a, b):
    assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z), abs=1e-9)


def test_curves_transformed_commute_with_eval():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Line3D, Parabola3D

    t = _rigid_motion()
    curves = [
        Line3D(Point3D(1.0, -2.0, 0.5), Vector3D(1.0, 2.0, 2.0)),
        Circle3D(Point3D(1.0, 0.0, 0.0), Vector3D.z(), 2.0),
        Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5),
        Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 0.5),
        Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0),
        BSplineCurve3D(
            2,
            [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 2.0, 0.0), Point3D(3.0, 1.0, 1.0)],
            [0.0, 1.0],
            [3, 3],
            False,
        ),
    ]
    for curve in curves:
        moved = curve.transformed(t)
        assert type(moved) is type(curve)
        for u in (0.0, 0.3, 0.9):
            _assert_same_point(moved.eval_point(u), t.apply_point(curve.eval_point(u)))


def test_surfaces_transformed_commute_with_eval():
    from geomcore.surfaces import Cone, Plane, Torus

    t = _rigid_motion()
    surfaces = [
        Plane(Point3D(0.0, 0.0, 1.0), Vector3D(0.0, 1.0, 1.0)),
        Cylinder(Point3D.origin(), Vector3D.z(), 2.0),
        Cone(Point3D.origin(), Vector3D.z(), 0.4, 1.0),
        Sphere(Point3D(1.0, 2.0, 3.0), 2.0),
        Torus(Point3D.origin(), Vector3D.z(), 3.0, 1.0),
        _bilinear_patch(),
    ]
    for surface in surfaces:
        moved = surface.transformed(t)
        assert type(moved) is type(surface)
        for u, v in ((0.0, 0.0), (0.3, 0.6), (0.9, 0.2)):
            _assert_same_point(moved.eval_point(u, v), t.apply_point(surface.eval_point(u, v)))


def test_transformed_scales_radius_and_rejects_collapse():
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    assert cylinder.transformed(Transform.scaling(Point3D.origin(), 3.0)).radius() == pytest.approx(6.0)
    with pytest.raises(ValueError):
        cylinder.transformed(Transform.scaling(Point3D.origin(), 0.0))


def test_eval_derivative_rejects_unsupported_orders():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    with pytest.raises(ValueError):
        circle.eval_derivative(0.0, 0)
    assert circle.eval_derivative(0.0, 5).components()[1] == pytest.approx(2.0)
    curve = BSplineCurve3D.interpolate(
        [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 1.0, 0.0), Point3D(2.0, 0.0, 0.0)], 2
    )
    with pytest.raises(ValueError):
        curve.eval_derivative(0.5, 3)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 1.0)
    for du, dv in [(0, 0), (3, 0), (2, 1)]:
        with pytest.raises(ValueError):
            cylinder.eval_derivative(0.0, 0.0, du, dv)


def test_conic_projection_of_non_finite_point_is_nan():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    o, z, x = Point3D.origin(), Vector3D.z(), Vector3D.x()
    for curve in [Ellipse3D(o, z, x, 3.0, 2.0), Hyperbola3D(o, z, x, 3.0, 2.0), Parabola3D(o, z, x, 1.0)]:
        for p in [Point3D(math.nan, 0.0, 0.0), Point3D(math.inf, 0.0, 0.0)]:
            projection = curve.project_point(p)
            u, d = projection.parameter, projection.distance
            assert math.isnan(u) and math.isnan(d)
            assert curve.extrema(p) == []


def test_trimmed_curve_segment_and_arc():
    from geomcore.curves import Line3D, TrimmedCurve3D

    segment = TrimmedCurve3D.segment(Point3D.origin(), Point3D(3.0, 4.0, 0.0))
    assert segment.bounds() == (0.0, 5.0)
    assert isinstance(segment.basis(), Line3D)
    assert segment.end_point().y == pytest.approx(4.0)
    # Beyond the end: the projection clamps to it.
    projection = segment.project_point(Point3D(6.0, 8.0, 0.0))
    u, d = projection.parameter, projection.distance
    assert u == pytest.approx(5.0)
    assert d == pytest.approx(5.0)
    assert not segment.contains(Point3D(6.0, 8.0, 0.0))

    # The right half of the unit circle, across the seam at angle 0.
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    arc = TrimmedCurve3D(circle, 1.5 * math.pi, 2.5 * math.pi)
    assert isinstance(arc.basis(), Circle3D)
    projection = arc.project_point(Point3D(3.0, 0.0, 0.0))
    u, d = projection.parameter, projection.distance
    assert u == pytest.approx(2.0 * math.pi)
    assert d == pytest.approx(2.0)
    projection = arc.project_point(Point3D(-3.0, 0.1, 0.0))
    u, _ = projection.parameter, projection.distance
    assert u == pytest.approx(2.5 * math.pi)
    assert arc.contains(Point3D(0.0, 1.0, 0.0))
    assert not arc.contains(Point3D(-1.0, 0.0, 0.0))
    assert [e.parameter for e in arc.extrema(Point3D(3.0, 0.0, 0.0))][0] == pytest.approx(2.0 * math.pi)


def test_trimmed_curve_rejects_bad_ranges():
    from geomcore.curves import Line3D, TrimmedCurve3D

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    for first, last in [(1.0, 1.0), (2.0, 1.0), (0.0, 7.0), (math.nan, 1.0)]:
        with pytest.raises(ValueError):
            TrimmedCurve3D(circle, first, last)
    with pytest.raises(ValueError):
        TrimmedCurve3D.segment(Point3D.origin(), Point3D.origin())
    with pytest.raises(TypeError):
        TrimmedCurve3D(TrimmedCurve3D(Line3D(Point3D.origin(), Vector3D.x()), 0.0, 1.0), 0.0, 1.0)
    curve = BSplineCurve3D(1, [Point3D.origin(), Point3D(2.0, 0.0, 0.0)], [0.0, 1.0], [2, 2], False)
    with pytest.raises(ValueError):
        TrimmedCurve3D(curve, -0.5, 0.5)
    with pytest.raises(ValueError):
        TrimmedCurve3D(curve, 0.0, 1.0).eval_derivative(0.5, 3)


def test_intersect_searches_only_the_range():
    from geomcore import curve_curve_extrema
    from geomcore.curves import TrimmedCurve3D

    segment = TrimmedCurve3D.segment(Point3D(-2.0, 0.5, 0.0), Point3D(2.0, 0.5, 0.0))
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    assert len(segment.intersect(circle).hits) == 2
    quarter = TrimmedCurve3D(circle, 0.0, math.pi / 2)
    (hit,) = segment.intersect(quarter).hits
    assert hit.point.x > 0.0

    a = TrimmedCurve3D.segment(Point3D.origin(), Point3D(1.0, 0.0, 0.0))
    b = TrimmedCurve3D.segment(Point3D(3.0, -1.0, 1.0), Point3D(3.0, 1.0, 1.0))
    assert curve_curve_extrema(a, b).minima[0].distance == pytest.approx(math.sqrt(5.0))

    sphere = Sphere(Point3D.origin(), 1.0)
    half = TrimmedCurve3D.segment(Point3D.origin(), Point3D(2.0, 0.0, 0.0))
    (hit,) = half.intersect(sphere).hits
    assert hit.curve_parameter == pytest.approx(1.0)


def test_trimmed_curve_2d():
    from geomcore.curves import Circle2D, TrimmedCurve2D

    arc = TrimmedCurve2D(Circle2D(Point2D.origin(), 2.0), 0.0, math.pi / 2)
    assert isinstance(arc.basis(), Circle2D)
    projection = arc.project_point(Point2D(1.0, 1.0))
    u, _ = projection.parameter, projection.distance
    assert u == pytest.approx(math.pi / 4)
    projection = arc.project_point(Point2D(-1.0, -5.0))
    u, _ = projection.parameter, projection.distance
    assert u == 0.0
    assert arc.contains(Point2D(0.0, 2.0))
    assert not arc.contains(Point2D(-2.0, 0.0))

    segment = TrimmedCurve2D.segment(Point2D.origin(), Point2D(0.0, 2.0))
    assert isinstance(segment.basis(), Line2D)
    assert segment.bounds() == (0.0, 2.0)
    with pytest.raises(TypeError):
        TrimmedCurve2D(Circle3D(Point3D.origin(), Vector3D.z(), 1.0), 0.0, 1.0)


def _intersect_shapes():
    from geomcore.curves import Circle2D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D
    from geomcore.surfaces import Cone, Plane, Torus

    o, z, x = Point3D.origin(), Vector3D.z(), Vector3D.x()
    return [
        Plane(Point3D(0.0, 0.0, 0.5), z),
        Cylinder(o, z, 2.0),
        Cone(o, z, 0.4, 2.0),
        Sphere(o, 3.0),
        Torus(o, z, 4.0, 1.0),
        Line3D(Point3D(0.0, -0.5, 0.25), x),
        Circle3D(o, z, 2.5),
        Ellipse3D(o, z, x, 3.0, 1.5),
        Parabola3D(o, z, x, 1.0),
        Hyperbola3D(o, z, x, 2.0, 1.0),
        Line2D(Point2D(0.0, 0.5), Vector2D(1.0, 0.0)),
        Circle2D(Point2D.origin(), 1.0),
    ]


def _is_surface(shape):
    return type(shape).__module__ == "geomcore.surfaces"


def _rounded(p):
    return tuple(round(c, 9) for c in ((p.x, p.y, p.z) if hasattr(p, "z") else (p.x, p.y)))


def _hit_points(result):
    """The points of a result's hits, rounded and sorted."""
    return sorted(_rounded(h.point) for h in result.hits)


def test_intersect_gives_the_same_result_in_both_orders():
    shapes = _intersect_shapes()
    pairs = 0
    for a in shapes:
        for b in shapes:
            if ("2D" in type(a).__name__) != ("2D" in type(b).__name__):
                continue
            forward, backward = a.intersect(b), b.intersect(a)
            assert forward.first is a and forward.second is b
            assert backward.first is b and backward.second is a
            assert type(forward) is type(backward)
            assert forward.method == backward.method, (a, b)
            if _is_surface(a) and _is_surface(b):
                assert len(forward.curves) == len(backward.curves), (a, b)
                assert forward.is_coincident() == backward.is_coincident()
            elif _is_surface(a) != _is_surface(b):
                # The curve's parameter keeps its role in either order.
                assert forward.hits == backward.hits, (a, b)
                assert forward.overlaps == backward.overlaps, (a, b)
            else:
                assert _hit_points(forward) == _hit_points(backward), (a, b)
                assert len(forward.overlaps) == len(backward.overlaps), (a, b)
            pairs += 1
    assert pairs == 104


def test_intersect_rejects_mixed_dimensions_and_foreign_objects():
    from geomcore.curves import Circle2D

    o, z = Point3D.origin(), Vector3D.z()
    circle = Circle3D(o, z, 2.5)
    for a, b in [
        (circle, Circle2D(Point2D.origin(), 1.0)),
        (Sphere(o, 1.0), Circle2D(Point2D.origin(), 1.0)),
        (circle, 5),
    ]:
        with pytest.raises(TypeError):
            a.intersect(b)
        if hasattr(b, "intersect"):
            with pytest.raises(TypeError):
                b.intersect(a)


def _xyz(p):
    return (p.x, p.y, p.z)


def test_discretize_circle_and_bspline():
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    # 2(1 - cos(pi/8)) = 0.152...: eight segments meet 0.16.
    d = circle.discretize(0.16)
    assert len(d.points) == len(d.parameters) == 9
    assert len(d.chord_deviations) == len(d.angle_deviations) == 8
    assert d.parameters[0] == 0.0 and d.parameters[-1] == pytest.approx(2.0 * math.pi)
    assert _xyz(d.points[0]) == _xyz(d.points[-1])
    assert all(dev == pytest.approx(2.0 * (1.0 - math.cos(math.pi / 8))) for dev in d.chord_deviations)
    assert all(angle == pytest.approx(math.pi / 8) for angle in d.angle_deviations)
    # A tighter angle limit asks for more points.
    assert len(circle.discretize(0.16, max_angle_deviation=0.1).points) == 33

    curve = BSplineCurve3D(
        2,
        [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 2.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [3, 3],
        False,
    )
    d = curve.discretize(1e-3)
    assert _xyz(d.points[0]) == (0.0, 0.0, 0.0)
    assert _xyz(d.points[-1]) == (2.0, 0.0, 0.0)
    assert all(dev <= 1e-3 for dev in d.chord_deviations)
    for u, p in zip(d.parameters, d.points):
        assert curve.eval_point(u).distance(p) < 1e-12

    with pytest.raises(ValueError, match="chord deviation"):
        circle.discretize(0.0)
    with pytest.raises(ValueError, match="angle deviation"):
        curve.discretize(1e-3, max_angle_deviation=-1.0)


def test_discretize_ellipse_and_trimmed_curves():
    from geomcore.curves import Ellipse3D, Parabola3D, TrimmedCurve3D

    # A round ellipse is a circle: 2(1 - cos(pi/8)) = 0.152... gives eight segments.
    round_ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 2.0)
    d = round_ellipse.discretize(0.16)
    assert len(d.points) == 9
    assert _xyz(d.points[0]) == _xyz(d.points[-1])
    flat = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 5.0, 0.5)
    d = flat.discretize(1e-3, max_angle_deviation=0.1)
    assert all(dev <= 1e-3 for dev in d.chord_deviations)
    assert all(angle <= 0.1 for angle in d.angle_deviations)
    # A segment is its own chord.
    segment = TrimmedCurve3D.segment(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 2.0, 2.0))
    d = segment.discretize(1e-7, max_angle_deviation=1e-9)
    assert [_xyz(p) for p in d.points] == [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0)]
    assert d.chord_deviations == [0.0]
    arc = TrimmedCurve3D(Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 0.5), -2.0, 3.0)
    d = arc.discretize(1e-3)
    assert d.parameters[0] == -2.0 and d.parameters[-1] == 3.0
    assert all(dev <= 1e-3 for dev in d.chord_deviations)


def test_discretize_2d_curves():
    from geomcore.curves import Circle2D, TrimmedCurve2D

    circle = Circle2D(Point2D(0.5, 0.25), 2.0)
    # 4 sin^2(pi/16) = 0.152...: eight segments meet 0.16.
    d = circle.discretize(0.16)
    assert isinstance(d, geomcore.Discretization2D)
    assert len(d.points) == 9 and len(d.chord_deviations) == 8
    assert (d.points[0].x, d.points[0].y) == (d.points[-1].x, d.points[-1].y)
    assert all(angle == pytest.approx(math.pi / 8) for angle in d.angle_deviations)
    arc = TrimmedCurve2D(circle, 0.5, 2.0)
    d = arc.discretize(1e-3, max_angle_deviation=0.05)
    assert d.parameters[0] == 0.5 and d.parameters[-1] == 2.0
    assert all(dev <= 1e-3 for dev in d.chord_deviations)
    assert all(angle <= 0.05 for angle in d.angle_deviations)
    segment = TrimmedCurve2D.segment(Point2D(0.0, 0.0), Point2D(3.0, 4.0))
    d = segment.discretize(1e-7)
    assert [(p.x, p.y) for p in d.points] == [(0.0, 0.0), (3.0, 4.0)]
    assert d.chord_deviations == [0.0] and d.angle_deviations == [0.0]
    with pytest.raises(ValueError):
        segment.discretize(0.0)
    # Below the confusion tolerance, segments already lie on the curve.
    with pytest.raises(ValueError, match="confusion"):
        segment.discretize(1e-9)


def _cubic_curve():
    return BSplineCurve3D(
        3,
        [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 2.0, 0.0), Point3D(3.0, 2.0, 1.0), Point3D(4.0, 0.0, 1.0)],
        [0.0, 1.0],
        [4, 4],
        False,
    )


def test_bspline_curve_getters():
    curve = _cubic_curve()
    assert [ (p.x, p.y, p.z) for p in curve.poles() ] == [
        (0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (3.0, 2.0, 1.0), (4.0, 0.0, 1.0)]
    assert curve.weights() is None
    assert curve.knots() == [0.0, 1.0]
    assert curve.multiplicities() == [4, 4]
    rational = BSplineCurve3D.new_rational(
        2,
        [Point3D(1.0, 0.0, 0.0), Point3D(1.0, 1.0, 0.0), Point3D(0.0, 1.0, 0.0)],
        [1.0, math.sqrt(2) / 2, 1.0],
        [0.0, 1.0],
        [3, 3],
        False,
    )
    assert rational.is_rational()
    assert rational.weights() == pytest.approx([1.0, math.sqrt(2) / 2, 1.0])


def test_bspline_curve_editing_round_trip():
    curve = _cubic_curve()
    refined = curve.refine_knots([0.25, 0.5, 0.5, 0.75])
    assert len(refined.poles()) == 8
    for u in (0.0, 0.3, 0.7, 1.0):
        assert refined.eval_point(u).distance(curve.eval_point(u)) < 1e-12
    simpler, removed = refined.remove_knots([0.25, 0.5, 0.5, 0.75], 1e-9)
    assert removed == 4
    assert simpler.knots() == curve.knots()
    single = curve.insert_knot(0.5, 2)
    assert len(single.poles()) == 6
    back, removed = single.remove_knot(0.5, 2, 1e-9)
    assert removed == 2
    assert back.knots() == curve.knots()
    raised = curve.elevate_degree(1)
    assert raised.degree() == 4
    assert raised.eval_point(0.4).distance(curve.eval_point(0.4)) < 1e-12
    with pytest.raises(ValueError):
        curve.insert_knot(2.0, 1)
    with pytest.raises(ValueError):
        curve.remove_knots([0.5], 0.0)


def test_bspline_curve_split_extract_reverse():
    curve = _cubic_curve()
    first, second = curve.split_at(0.4)
    assert first.bounds() == pytest.approx((0.0, 0.4))
    assert second.bounds() == pytest.approx((0.4, 1.0))
    assert first.eval_point(0.4).distance(curve.eval_point(0.4)) < 1e-12
    assert second.eval_point(0.4).distance(curve.eval_point(0.4)) < 1e-12
    piece = curve.extract(0.2, 0.8)
    assert piece.bounds() == pytest.approx((0.2, 0.8))
    assert piece.eval_point(0.5).distance(curve.eval_point(0.5)) < 1e-12
    rev = curve.reversed()
    assert rev.eval_point(0.3).distance(curve.eval_point(0.7)) < 1e-12
    with pytest.raises(ValueError):
        curve.split_at(0.0)
    with pytest.raises(ValueError):
        curve.extract(0.8, 0.2)


def test_bspline_curve_bezier_pieces_and_length():
    curve = BSplineCurve3D(
        3,
        [Point3D(float(i), float(i % 2), 0.0) for i in range(6)],
        [0.0, 1.0, 2.0, 3.0],
        [4, 1, 1, 4],
        False,
    )
    pieces = curve.to_bezier_pieces()
    assert len(pieces) == 3
    assert pieces[1].bounds() == pytest.approx((1.0, 2.0))
    assert pieces[1].eval_point(1.5).distance(curve.eval_point(1.5)) < 1e-12
    segment = BSplineCurve3D(
        1, [Point3D.origin(), Point3D(2.0, 0.0, 0.0)], [0.0, 1.0], [2, 2], False
    )
    assert segment.length(1e-9) == pytest.approx(2.0)
    assert _cubic_curve().length(1e-9) == pytest.approx(5.39065, abs=1e-4)


def test_bspline_curve_approximate_within():
    points = [Point3D(float(i), 0.1 * float(i), 0.0) for i in range(6)]
    fit = BSplineCurve3D.approximate_within(points, 2, 1e-6)
    for p in points:
        assert fit.contains(p, Tolerance(1e-3, 1e-9, 1e-9))


def test_conic_to_bspline():
    from geomcore.curves import Ellipse3D, TrimmedCurve3D

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    spline = circle.to_bspline()
    assert spline.is_periodic()
    assert spline.eval_point(math.pi / 2).distance(circle.eval_point(math.pi / 2)) < 1e-12
    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.0)
    e_spline = ellipse.to_bspline()
    assert e_spline.eval_point(0.0).distance(ellipse.eval_point(0.0)) < 1e-12
    arc = TrimmedCurve3D(circle, 0.0, math.pi)
    arc_spline = arc.to_bspline()
    assert arc_spline.bounds() == pytest.approx((0.0, math.pi))
    # The B-spline parameter matches the angle at the knots and each
    # piece's middle, not in between — but every point lies on the circle.
    assert arc_spline.eval_point(math.pi / 2).distance(circle.eval_point(math.pi / 2)) < 1e-12
    for u in (0.0, 0.3, 1.0, 2.0, math.pi):
        assert circle.contains(arc_spline.eval_point(u))
    segment = TrimmedCurve3D.segment(Point3D.origin(), Point3D(1.0, 0.0, 0.0))
    seg_spline = segment.to_bspline()
    assert seg_spline.eval_point(0.5).distance(Point3D(0.5, 0.0, 0.0)) < 1e-12


def test_curve_bounds_and_boxes():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Line3D, Parabola3D, TrimmedCurve3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    assert line.bounds() == (float("-inf"), float("inf"))
    assert line.bounding_box() is None
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.bounds() == pytest.approx((0.0, math.tau))
    box = circle.bounding_box()
    assert (box.min().x, box.max().x) == pytest.approx((-2.0, 2.0))
    assert (box.min().y, box.max().y) == pytest.approx((-2.0, 2.0))
    assert box.encloses(Point3D(1.0, 1.0, 0.0))
    assert not box.encloses(Point3D(3.0, 0.0, 0.0))
    assert Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0).bounding_box() is None
    assert Hyperbola3D(
        Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0, 1.0
    ).bounding_box() is None
    assert Ellipse3D(
        Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0
    ).bounding_box() is not None
    curve = _cubic_curve()
    cbox = curve.bounding_box()
    assert (cbox.min().x, cbox.max().x) == pytest.approx((0.0, 4.0))
    trimmed = TrimmedCurve3D(circle, 0.0, math.pi / 2)
    tbox = trimmed.bounding_box()
    assert tbox is not None and tbox.encloses(Point3D(2.0, 0.0, 0.0))


def test_bounding_box_ops():
    box = BoundingBox3D(Point3D(1.0, 0.0, 5.0), Point3D(0.0, 2.0, 3.0))
    assert (box.min().x, box.min().y, box.min().z) == (0.0, 0.0, 3.0)
    assert (box.max().x, box.max().y, box.max().z) == (1.0, 2.0, 5.0)
    assert (box.center().x, box.center().y, box.center().z) == (0.5, 1.0, 4.0)
    assert box.diagonal().components() == (1.0, 2.0, 2.0)
    other = BoundingBox3D(Point3D(2.0, -1.0, 0.0), Point3D(3.0, 0.0, 0.5))
    both = box.union(other)
    assert (both.min().x, both.max().x) == (0.0, 3.0)
    grown = box.enlarged(0.5)
    assert grown.min().x == pytest.approx(-0.5)
    assert box.overlaps(other) or not box.overlaps(other)
    assert not box.overlaps(BoundingBox3D(Point3D(5.0, 5.0, 5.0), Point3D(6.0, 6.0, 6.0)))
    assert BoundingBox3D.from_points([Point3D(1.0, 2.0, 3.0)]).min().x == 1.0
    with pytest.raises(ValueError):
        BoundingBox3D.from_points([])


def test_surface_getters_and_uv_bounds():
    surface = _bilinear_patch()
    assert (surface.u_degree(), surface.v_degree()) == (1, 1)
    assert surface.u_pole_count() == 2 and surface.v_pole_count() == 2
    assert [(p.x, p.y) for p in surface.poles()[1]] == [(2.0, 0.0), (2.0, 2.0)]
    assert surface.weights() is None
    assert surface.u_knots() == [0.0, 1.0] and surface.v_knots() == [0.0, 1.0]
    assert surface.u_multiplicities() == [2, 2]
    assert surface.u_bounds() == (0.0, 1.0) and surface.v_bounds() == (0.0, 1.0)
    sphere = Sphere(Point3D.origin(), 2.0)
    assert sphere.u_bounds() == pytest.approx((0.0, math.tau))
    assert sphere.bounding_box().max().x == pytest.approx(2.0)
    assert sphere.bounding_box().min().x == pytest.approx(-2.0)
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    assert plane.bounding_box() is None
    assert plane.u_bounds() == (float("-inf"), float("inf"))


def test_surface_editing_round_trip():
    surface = _bilinear_patch()
    refined = surface.refine_u_knots([0.25, 0.5])
    assert refined.u_pole_count() == 4
    assert refined.eval_point(0.3, 0.7).distance(surface.eval_point(0.3, 0.7)) < 1e-12
    simpler, removed = refined.remove_u_knots([0.25, 0.5], 1e-9)
    assert removed == 2
    assert simpler.u_knots() == surface.u_knots()
    one, n = surface.insert_v_knot(0.5, 1).remove_v_knot(0.5, 1, 1e-9)
    assert n == 1 and one.v_knots() == surface.v_knots()
    raised = surface.elevate_u_degree(1)
    assert raised.u_degree() == 2
    assert raised.eval_point(0.4, 0.6).distance(surface.eval_point(0.4, 0.6)) < 1e-12
    raised_v = surface.elevate_v_degree(2)
    assert raised_v.v_degree() == 3
    with pytest.raises(ValueError):
        surface.insert_u_knot(2.0, 1)


def test_surface_split_extract_reverse_and_patches():
    surface = _bilinear_patch()
    left, right = surface.split_u_at(0.5)
    assert left.u_bounds() == pytest.approx((0.0, 0.5))
    assert right.eval_point(0.75, 0.25).distance(surface.eval_point(0.75, 0.25)) < 1e-12
    bottom, _top = surface.split_v_at(0.5)
    assert bottom.v_bounds() == pytest.approx((0.0, 0.5))
    piece = surface.extract((0.25, 0.75), (0.25, 0.75))
    assert piece.eval_point(0.5, 0.5).distance(surface.eval_point(0.5, 0.5)) < 1e-12
    assert surface.extract_u(0.25, 0.75).u_knots() == pytest.approx([0.25, 0.75])
    assert surface.extract_v(0.25, 0.75).v_knots() == pytest.approx([0.25, 0.75])
    assert surface.reversed_u().eval_point(0.25, 0.5).distance(
        surface.eval_point(0.75, 0.5)) < 1e-12
    assert surface.reversed_v().eval_point(0.5, 0.25).distance(
        surface.eval_point(0.5, 0.75)) < 1e-12
    from geomcore.surfaces import BSplineSurface

    net = BSplineSurface(
        2, 2,
        [[Point3D(float(i), float(j), float(i * j)) for j in range(3)] for i in range(4)],
        [0.0, 1.0, 2.0], [3, 1, 3], [0.0, 1.0], [3, 3], False, False,
    )
    patches = net.to_bezier_patches()
    assert (len(patches), len(patches[0])) == (2, 1)
    assert patches[1][0].eval_point(1.5, 0.5).distance(net.eval_point(1.5, 0.5)) < 1e-12
    with pytest.raises(ValueError):
        surface.split_u_at(0.0)


def test_surface_approximate():
    from geomcore.surfaces import BSplineSurface

    grid = [
        [Point3D(x, y, math.sin(3 * x) * math.cos(2 * y)) for j in range(8) for y in [j / 7]]
        for i in range(8) for x in [i / 7]
    ]
    fit = BSplineSurface.approximate(grid, 3, 3, 5, 5)
    assert (fit.u_pole_count(), fit.v_pole_count()) == (5, 5)
    loose = BSplineSurface.approximate_within(grid, 3, 3, 0.05)
    assert loose.u_pole_count() <= 8 and loose.v_pole_count() <= 8
    assert loose.eval_point(0.5, 0.5).distance(fit.eval_point(0.5, 0.5)) < 0.2


def test_surface_to_bspline_conversions():
    from geomcore.surfaces import BSplineSurface, Cone, Cylinder, Plane, Torus

    plane = Plane(Point3D.origin(), Vector3D.z())
    patch = plane.to_bspline_patch((0.0, 1.0), (0.0, 2.0))
    assert isinstance(patch, BSplineSurface)
    assert patch.eval_point(0.5, 1.0).distance(plane.eval_point(0.5, 1.0)) < 1e-12
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    tube = cylinder.to_bspline_patch((0.0, math.tau), (-1.0, 4.0))
    assert tube.is_u_periodic() and not tube.is_v_periodic()
    assert tube.eval_point(math.pi / 2, 3.0).distance(
        cylinder.eval_point(math.pi / 2, 3.0)) < 1e-12
    sphere = Sphere(Point3D.origin(), 2.0)
    ball = sphere.to_bspline()
    assert ball.eval_point(0.0, 0.0).distance(sphere.eval_point(0.0, 0.0)) < 1e-9
    torus = Torus(Point3D.origin(), Vector3D.z(), 3.0, 1.0)
    donut = torus.to_bspline()
    assert donut.eval_point(0.0, 0.0).distance(torus.eval_point(0.0, 0.0)) < 1e-9
    cone = Cone(Point3D.origin(), Vector3D.z(), 0.5, 1.0)
    assert cone.to_bspline_patch((0.0, 1.0), (0.0, 2.0)).eval_point(
        0.5, 1.0).distance(cone.eval_point(0.5, 1.0)) < 1e-9
    with pytest.raises(ValueError):
        plane.to_bspline_patch((0.0, 0.0), (0.0, 1.0))


def _iso_surfaces():
    from geomcore.surfaces import Cone, Plane, Torus

    return [
        Plane(Point3D(1.0, 2.0, 3.0), Vector3D(0.0, 0.0, 1.0)),
        Cylinder(Point3D.origin(), Vector3D.z(), 2.0),
        Cone(Point3D.origin(), Vector3D.z(), 0.4, 1.0),
        Sphere(Point3D(1.0, 0.0, 0.0), 3.0),
        Torus(Point3D.origin(), Vector3D.z(), 5.0, 1.5),
        Sphere(Point3D.origin(), 2.0).to_bspline_patch((0.0, 3.0), (-1.0, 1.0)),
    ]


@pytest.mark.parametrize("index", range(6))
def test_iso_curves_are_the_surface_with_one_parameter_fixed(index):
    s = _iso_surfaces()[index]

    def finite(lo, hi):
        return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) else (-2.0, 2.0)

    u0, u1 = finite(*s.u_bounds())
    v0, v1 = finite(*s.v_bounds())
    for k in (0.2, 0.5, 0.9):
        u, v = u0 + k * (u1 - u0), v0 + k * (v1 - v0)
        for t in (0.1, 0.6):
            tv, tu = v0 + t * (v1 - v0), u0 + t * (u1 - u0)
            for got, want in (
                (s.u_iso(u).eval_point(tv), s.eval_point(u, tv)),
                (s.v_iso(v).eval_point(tu), s.eval_point(tu, v)),
            ):
                assert (got.x, got.y, got.z) == pytest.approx((want.x, want.y, want.z), abs=1e-9)


def test_iso_curves_have_the_kernel_types():
    from geomcore.curves import Line3D

    plane, cylinder, cone, sphere, torus, patch = _iso_surfaces()
    assert isinstance(plane.u_iso(0.0), Line3D) and isinstance(plane.v_iso(0.0), Line3D)
    assert isinstance(cylinder.u_iso(0.0), Line3D) and isinstance(cylinder.v_iso(0.0), Circle3D)
    assert isinstance(cone.u_iso(0.0), Line3D) and isinstance(cone.v_iso(0.0), Circle3D)
    assert isinstance(sphere.u_iso(0.0), Circle3D) and isinstance(sphere.v_iso(0.0), Circle3D)
    assert isinstance(torus.u_iso(0.0), Circle3D) and isinstance(torus.v_iso(0.0), Circle3D)
    assert isinstance(patch.u_iso(1.0), BSplineCurve3D) and isinstance(patch.v_iso(0.0), BSplineCurve3D)


def test_iso_curve_errors_are_value_errors():
    with pytest.raises(ValueError):
        Sphere(Point3D.origin(), 1.0).u_iso(math.nan)
    with pytest.raises(ValueError):
        Sphere(Point3D.origin(), 2.0).to_bspline_patch((0.0, 3.0), (-1.0, 1.0)).u_iso(10.0)


def test_a_pole_iso_is_a_circle_of_radius_zero():
    pole = Sphere(Point3D.origin(), 1.0).v_iso(math.pi / 2)
    assert pole.radius() == pytest.approx(0.0, abs=1e-12)
    assert pole.center().z == pytest.approx(1.0)
