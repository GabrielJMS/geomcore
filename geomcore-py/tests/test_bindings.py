"""Behavioral tests for the geomcore Python bindings."""

import math

import pytest

import geomcore
from geomcore import Axis3D, Frame3D, Point2D, Point3D, Tolerance, Transform, Vector2D, Vector3D
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
    u, dist = circle.project_point(Point3D(3.0, 0.0, 0.0))
    assert u == pytest.approx(0.0)
    assert dist == pytest.approx(1.0)
    u, dist = circle.project_point(Point3D(2.0, 0.0, 0.0))
    assert dist == 0.0
    # Batch variant agrees with scalar calls.
    batch = circle.project_points([Point3D(3.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)])
    assert batch[0] == pytest.approx((0.0, 1.0))
    assert batch[1][1] == 0.0


def test_project_point_surface():
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    u, v, dist = plane.project_point(Point3D(1.0, 2.0, 3.0))
    assert (u, v) == pytest.approx((1.0, 2.0))
    assert dist == pytest.approx(3.0)
    batch = plane.project_points([Point3D.origin(), Point3D(0.0, 0.0, 2.0)])
    assert batch[0][2] == 0.0
    assert batch[1][2] == pytest.approx(2.0)


def test_intersect_plane_plane():
    from geomcore.surfaces import Plane

    xy = Plane(Point3D.origin(), Vector3D.z())
    zy = Plane(Point3D.origin(), Vector3D.x())
    kind, line = xy.intersect(zy)
    assert kind == "line"
    assert line.origin().x == pytest.approx(0.0)
    # Direction follows the ordered pair (n1 x n2 = Z x X = Y).
    assert line.direction().components() == pytest.approx((0.0, 1.0, 0.0))

    kind, payload = xy.intersect(Plane(Point3D(0.0, 0.0, 1.0), Vector3D.z()))
    assert (kind, payload) == ("parallel", None)

    kind, payload = xy.intersect(xy)
    assert (kind, payload) == ("coincident", None)


def test_intersect_plane_sphere():
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    kind, circle = plane.intersect(Sphere(Point3D.origin(), 2.0))
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)

    kind, point = plane.intersect(Sphere(Point3D(0.0, 0.0, 2.0), 2.0))
    assert kind == "tangent_point"
    assert (point.x, point.y, point.z) == pytest.approx((0.0, 0.0, 0.0))

    assert plane.intersect(Sphere(Point3D(0.0, 0.0, 5.0), 2.0)) == ("empty", None)


def test_intersect_sphere_sphere():
    s1 = Sphere(Point3D.origin(), 2.0)
    kind, circle = s1.intersect(Sphere(Point3D(3.0, 0.0, 0.0), 2.0))
    assert kind == "circle"
    assert circle.center().x == pytest.approx(1.5)

    kind, point = s1.intersect(Sphere(Point3D(4.0, 0.0, 0.0), 2.0))
    assert kind == "tangent_point"
    assert point.x == pytest.approx(2.0)

    assert s1.intersect(Sphere(Point3D(5.0, 0.0, 0.0), 2.0)) == ("empty", None)
    assert s1.intersect(s1)[0] == "coincident"


def test_intersect_plane_cylinder():
    from geomcore.surfaces import Cylinder, Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    kind, circle = plane.intersect(cylinder)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)

    side = Plane(Point3D(0.0, 1.0, 0.0), Vector3D.y())
    kind, lines = side.intersect(cylinder)
    assert kind == "two_lines"
    assert lines[0].direction().components() == pytest.approx((0.0, 0.0, 1.0))

    assert side.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 0.5)) == (
        "empty",
        None,
    )
    miss = Plane(Point3D(0.0, 3.0, 0.0), Vector3D.y())
    assert miss.intersect(cylinder) == ("empty", None)


def test_intersect_plane_cone():
    from geomcore import Frame3D
    from geomcore.surfaces import Cone, Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    kind, circle = plane.intersect(cone)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(2.0)

    # The cone is double: a plane beyond the apex cuts the other nappe.
    below = Plane(Point3D(0.0, 0.0, -6.0), Vector3D.z())
    kind, circle = below.intersect(cone)
    assert kind == "circle"
    assert circle.radius() == pytest.approx(0.5368, abs=1e-3)

    kind, (h1, h2) = Plane(Point3D(3.0, 0.0, 0.0), Vector3D.x()).intersect(cone)
    assert kind == "hyperbola"
    assert h1.center().distance(h2.center()) < 1e-9

    apex_plane = Plane(cone.apex(), Vector3D.z())
    kind, payload = apex_plane.intersect(cone)
    assert kind == "apex_point"


def test_intersect_line_surface():
    from geomcore.curves import Line3D
    from geomcore.surfaces import Cylinder, Plane

    line = Line3D(Point3D.origin(), Vector3D.x())
    kind, ((t1, p1), (t2, p2)) = line.intersect(Sphere(Point3D.origin(), 2.0))
    assert kind == "two_points"
    assert (t1, t2) == pytest.approx((-2.0, 2.0))
    assert (p1.x, p2.x) == pytest.approx((-2.0, 2.0))

    kind, (t, p) = Line3D(Point3D.origin(), Vector3D.z()).intersect(
        Plane(Point3D.origin(), Vector3D.z())
    )
    assert kind == "point"
    assert t == pytest.approx(0.0)
    assert line.intersect(Plane(Point3D.origin(), Vector3D.z())) == (
        "coincident",
        None,
    )

    gen = Line3D(Point3D(2.0, 0.0, 0.0), Vector3D.z())
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    assert gen.intersect(cylinder) == ("coincident", None)


def test_intersect_symmetric_quadrics():
    import math

    from geomcore import Frame3D
    from geomcore.surfaces import Cone, Cylinder

    sphere = Sphere(Point3D.origin(), 3.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)

    kind, (c1, c2) = sphere.intersect(cylinder)
    assert kind == "two_circles"
    assert c1.radius() == pytest.approx(2.0)
    assert c1.center().z == pytest.approx(math.sqrt(5.0))

    kind, _ = sphere.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 3.0))
    assert kind == "circle"
    off = Cylinder(Point3D(0.0, 1.0, 0.0), Vector3D.z(), 2.0)
    assert sphere.intersect(off) == ("not_analytic", None)

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    kind, (r1, r2) = sphere.intersect(cone)
    assert kind == "two_circles"
    assert r1.radius() == pytest.approx(2.6188, abs=1e-3)
    assert r2.radius() == pytest.approx(0.7746, abs=1e-3)
    # Sphere around the apex: one ring on each nappe.
    small = Cone.from_frame(Frame3D.world(), 0.4, 0.5)
    kind, (upper, lower) = sphere.intersect(small)
    assert kind == "two_circles"
    assert upper.radius() == pytest.approx(1.5786, abs=1e-3)
    assert lower.radius() == pytest.approx(0.7302, abs=1e-3)
    assert lower.center().z < small.apex().z < upper.center().z

    c1 = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    c2 = Cylinder(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    kind, (l1, l2) = c1.intersect(c2)
    assert kind == "two_lines"
    assert l1.origin().x == pytest.approx(1.5)
    assert c1.intersect(c1)[0] == "coincident"
    crossed = Cylinder(Point3D.origin(), Vector3D.x(), 2.0)
    assert c1.intersect(crossed) == ("not_analytic", None)


def test_intersect_cone_cylinder():
    from geomcore import Frame3D
    from geomcore.surfaces import Cone, Cylinder

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    kind, (lower, upper) = cone.intersect(cylinder)
    assert kind == "two_circles"
    assert (lower.radius(), upper.radius()) == pytest.approx((2.0, 2.0))
    assert upper.center().distance(Point3D.origin()) < 1e-9
    assert lower.center().z == pytest.approx(2.0 * cone.apex().z)

    off = Cylinder(Point3D(1.0, 0.0, 0.0), Vector3D.z(), 2.0)
    assert cone.intersect(off) == ("not_analytic", None)


def test_intersect_cone_cone():
    from geomcore import Frame3D
    from geomcore.surfaces import Cone

    c1 = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    frame2 = Frame3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), Vector3D.x())
    c2 = Cone.from_frame(frame2, 0.6, 1.0)
    kind, (r1, r2) = c1.intersect(c2)
    assert kind == "two_circles"
    assert r1.radius() == pytest.approx(1.1155, abs=1e-3)
    assert r2.radius() == pytest.approx(4.7245, abs=1e-3)

    assert c1.intersect(c1)[0] == "coincident"
    tilted = Cone(Point3D.origin(), Vector3D.x(), 0.4, 2.0)
    assert c1.intersect(tilted) == ("not_analytic", None)


def test_intersect_torus_symmetric():
    from geomcore.surfaces import Cone, Cylinder, Plane, Torus

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)

    kind, (c1, c2) = torus.intersect(Plane(Point3D.origin(), Vector3D.z()))
    assert kind == "two_circles"
    assert sorted([c1.radius(), c2.radius()]) == pytest.approx([3.0, 5.0])

    kind, _ = torus.intersect(Plane(Point3D.origin(), Vector3D.x()))
    assert kind == "two_circles"

    # Bitangent plane through the center: Villarceau circles.
    tilt = math.asin(1.0 / 4.0)
    villarceau = Plane(Point3D.origin(), Vector3D(0.0, math.sin(tilt), math.cos(tilt)))
    kind, (c1, c2) = torus.intersect(villarceau)
    assert kind == "two_circles"
    assert [c1.radius(), c2.radius()] == pytest.approx([4.0, 4.0])
    assert sorted([c1.center().x, c2.center().x]) == pytest.approx([-1.0, 1.0])

    kind, c = torus.intersect(Sphere(Point3D.origin(), 5.0))
    assert kind == "tangent_circle"
    assert c.radius() == pytest.approx(5.0)

    kind, _ = torus.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 4.5))
    assert kind == "two_circles"

    kind, rings = torus.intersect(Cone.from_frame(Frame3D.world(), 1.1, 2.0))
    assert kind == "rings"
    assert [tangent for _, tangent in rings] == [False, False]
    kind, rings = torus.intersect(Cone.from_frame(Frame3D.world(), 1.4, 0.0))
    assert kind == "rings"
    assert len(rings) == 4

    other = Torus(Point3D(0.0, 0.0, 1.5), Vector3D.z(), 4.0, 1.0)
    kind, _ = torus.intersect(other)
    assert kind == "two_circles"
    assert torus.intersect(torus)[0] == "coincident"
    tilted = Torus(Point3D.origin(), Vector3D.x(), 4.0, 1.0)
    assert torus.intersect(tilted) == ("not_analytic", None)


def test_conic_surface_intersections():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D
    from geomcore.surfaces import Cone, Cylinder, Plane

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    kind, hits = ellipse.intersect(Plane(Point3D.origin(), Vector3D.x()))
    assert kind == "hits"
    assert len(hits) == 2

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    kind, _ = parabola.intersect(Plane(Point3D.origin(), Vector3D.z()))
    assert kind == "coincident"

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    cylinder = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    kind, hits = hyperbola.intersect(cylinder)
    assert kind == "hits"
    assert len(hits) >= 1

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    kind, _ = ellipse.intersect(cone)
    assert kind == "hits"


def test_circle_surface_and_measures():
    import math

    from geomcore.surfaces import Cone, Cylinder, Plane

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert circle.circumference() == pytest.approx(math.tau * 2.0)
    assert circle.disk_area() == pytest.approx(math.pi * 4.0)

    kind, hits = circle.intersect(Plane(Point3D.origin(), Vector3D.x()))
    assert kind == "hits"
    assert len(hits) == 2

    kind, _ = circle.intersect(Sphere(Point3D.origin(), 3.0))
    assert kind == "empty"

    kind, _ = circle.intersect(Cylinder(Point3D.origin(), Vector3D.z(), 2.0))
    assert kind == "circle"

    cone = Cone.from_frame(Frame3D.world(), 0.4, 2.0)
    kind, _ = circle.intersect(cone)
    assert kind == "circle"

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


def test_intersect_curve_surface():
    from geomcore import intersect_curve_surface
    from geomcore.curves import Line3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    hits = intersect_curve_surface(line, Sphere(Point3D.origin(), 2.0))
    assert len(hits) == 2
    assert hits[0][0] == pytest.approx(-2.0)
    assert hits[1][0] == pytest.approx(2.0)

    assert intersect_curve_surface(line, Sphere(Point3D(0.0, 0.0, 5.0), 2.0)) == []


def test_intersect_marching():
    from geomcore import intersect_marching
    from geomcore.surfaces import Plane

    xy = Plane(Point3D.origin(), Vector3D.z())
    zy = Plane(Point3D.origin(), Vector3D.x())
    curves = intersect_marching(xy, zy)
    assert len(curves) == 1

    sphere = Sphere(Point3D.origin(), 2.0)
    loops = intersect_marching(xy, sphere)
    assert len(loops) == 1

    miss = Plane(Point3D(0.0, 0.0, 5.0), Vector3D.z())
    assert intersect_marching(miss, sphere) == []


@pytest.mark.parametrize("height", [0.3, 0.8, -0.45])
def test_intersect_marching_off_grid_plane(height: float) -> None:
    # Planes between the seed grid lines used to find no curve (#6).
    from geomcore import intersect_marching
    from geomcore.surfaces import Plane

    plane = Plane(Point3D(0.0, 0.0, height), Vector3D.z())
    curves = intersect_marching(plane, Sphere(Point3D.origin(), 1.0))
    assert len(curves) == 1
    p = curves[0].eval_point(0.5)
    assert math.hypot(p.x, p.y) == pytest.approx(math.sqrt(1.0 - height**2), abs=1e-3)


def test_curve_curve_generic():
    from geomcore import curve_curve_extrema, intersect_curve_curve
    from geomcore.curves import Circle3D, Line3D

    l1 = Line3D(Point3D.origin(), Vector3D.x())
    l2 = Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.y())
    ext = curve_curve_extrema(l1, l2)
    assert len(ext) == 1
    assert ext[0][2] == pytest.approx(1.0)

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    hits = intersect_curve_curve(l1, circle)
    assert len(hits) == 2
    assert hits[0][0] == pytest.approx(-2.0)
    assert hits[1][0] == pytest.approx(2.0)


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
    kind, (s, p, t) = l1.intersect(l2)
    assert kind == "point"
    assert (s, t) == pytest.approx((0.0, 0.0))

    circle = Circle2D(Point2D(0.0, 0.0), 2.0)
    kind, ((t1, _), (t2, _)) = l1.intersect(circle)
    assert kind == "points"
    assert (t1, t2) == pytest.approx((-2.0, 2.0))

    kind, (p1, p2) = circle.intersect(Circle2D(Point2D(3.0, 0.0), 2.0))
    assert kind == "points"
    assert p1.x == pytest.approx(1.5)

    a = Line3D(Point3D.origin(), Vector3D.x())
    b = Line3D(Point3D.origin(), Vector3D.y())
    kind, (_, p, _) = a.intersect(b)
    assert kind == "point"
    assert (p.x, p.y, p.z) == pytest.approx((0.0, 0.0, 0.0))
    kind, payload = a.intersect(Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.y()))
    assert kind == "skew"
    assert payload[2] == pytest.approx(1.0)


def test_intersect_curves_3d():
    from geomcore.curves import Circle3D, Line3D

    line = Line3D(Point3D.origin(), Vector3D.x())
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    kind, ((t1, _), (t2, _)) = line.intersect(circle)
    assert kind == "points"
    assert (t1, t2) == pytest.approx((-2.0, 2.0))

    c1 = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    c2 = Circle3D(Point3D(3.0, 0.0, 0.0), Vector3D.z(), 2.0)
    kind, (p1, p2) = c1.intersect(c2)
    assert kind == "points"
    assert p1.x == pytest.approx(1.5)
    tilted = Circle3D(Point3D.origin(), Vector3D.x(), 2.0)
    assert c1.intersect(tilted) == ("not_analytic", None)
    assert c1.intersect(c1)[0] == "coincident"
    lifted = Circle3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), 2.0)
    assert c1.intersect(lifted) == ("empty", None)


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
    u, v, dist = surface.project_point(Point3D(1.0, 1.0, 1.0))
    assert (u, v) == pytest.approx((0.5, 0.5))
    assert dist == pytest.approx(1.0)
    assert surface.contains(Point3D(1.0, 1.0, 0.0))
    assert not surface.contains(Point3D(1.0, 1.0, 1.0))
    ext = surface.extrema(Point3D(1.0, 1.0, 1.0))
    assert len(ext) >= 1
    assert ext[0][2] == pytest.approx(1.0)


def test_bspline_curve_projection():
    curve = BSplineCurve3D(
        1,
        [Point3D(0.0, 0.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [2, 2],
        False,
    )
    u, dist = curve.project_point(Point3D(1.0, 1.0, 0.0))
    assert u == pytest.approx(0.5)
    assert dist == pytest.approx(1.0)
    assert curve.contains(Point3D(1.0, 0.0, 0.0))
    assert not curve.contains(Point3D(1.0, 1.0, 0.0))
    # Beyond the end: the endpoint wins.
    u, dist = curve.project_point(Point3D(5.0, 0.0, 0.0))
    assert u == pytest.approx(1.0)
    assert dist == pytest.approx(3.0)
    ext = curve.extrema(Point3D(1.0, 1.0, 0.0))
    assert len(ext) >= 1
    assert ext[0][1] == pytest.approx(1.0)


def test_conic_extrema_and_projection():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    ext = ellipse.extrema(Point3D(4.0, 0.0, 0.0))
    assert len(ext) >= 2
    assert ext[0][1] == pytest.approx(1.0)
    u, dist = ellipse.project_point(Point3D(0.0, 3.0, 0.0))
    assert dist == pytest.approx(1.5)

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    u, dist = parabola.project_point(Point3D(1.0, 2.0, 0.0))
    assert u == pytest.approx(2.0)
    assert dist == 0.0

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    u, dist = hyperbola.project_point(Point3D(3.0, 0.0, 0.0))
    assert dist == pytest.approx(math.sqrt(0.8))


def test_line_conic_intersections():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Line3D, Parabola3D

    tol = Tolerance.default()
    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    kind, ((t1, p1, u1), (t2, p2, u2)) = Line3D(Point3D.origin(), Vector3D.x()).intersect(
        ellipse
    )
    assert kind == "points"
    assert (t1, t2) == pytest.approx((-3.0, 3.0))
    assert (u1, u2) == pytest.approx((math.pi, 0.0))

    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    kind, (t, p, u) = Line3D(Point3D.origin(), Vector3D.x()).intersect(parabola)
    assert kind == "tangent"
    assert t == pytest.approx(0.0) and u == pytest.approx(0.0)

    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    kind, (t, p, u) = Line3D(Point3D(2.0, -5.0, 0.0), Vector3D.y()).intersect(
        hyperbola, tol
    )
    assert kind == "tangent"
    assert (p.x, p.y, p.z) == pytest.approx((2.0, 0.0, 0.0))

    miss = Line3D(Point3D(0.0, 0.0, 1.0), Vector3D.x())
    assert miss.intersect(ellipse) == ("empty", None)
    assert miss.intersect(parabola) == ("empty", None)
    assert miss.intersect(hyperbola) == ("empty", None)


def test_circle_conic_intersections():
    from geomcore.curves import Circle3D, Ellipse3D, Hyperbola3D, Parabola3D

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    kind, hits = circle.intersect(ellipse)
    assert kind == "hits"
    assert len(hits) == 4
    for s, p, u, mult in hits:
        assert mult == 1
        assert circle.contains(p)
        assert ellipse.contains(p)
        _assert_same_point(circle.eval_point(s), p)
        _assert_same_point(ellipse.eval_point(u), p)

    assert circle.intersect(
        Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 2.0)
    ) == ("coincident", None)
    # Crossing planes: the tilted ellipse meets the shared line at
    # y = ±1.5, inside the circle.
    tilted = Ellipse3D(Point3D.origin(), Vector3D.x(), Vector3D.z(), 3.0, 1.5)
    assert circle.intersect(tilted) == ("empty", None)

    unit = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    parabola = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.0)
    kind, hits = unit.intersect(parabola)
    assert kind == "hits"
    assert len(hits) == 2

    big = Circle3D(Point3D.origin(), Vector3D.z(), 3.0)
    hyperbola = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    kind, hits = big.intersect(hyperbola)
    assert kind == "hits"
    assert len(hits) == 2
    assert all(p.x > 0.0 for _, p, _, _ in hits)


def test_line_torus_intersections():
    from geomcore.curves import Line3D
    from geomcore.surfaces import Torus

    torus = Torus(Point3D.origin(), Vector3D.z(), 4.0, 1.0)
    kind, hits = Line3D(Point3D.origin(), Vector3D.x()).intersect(torus)
    assert kind == "hits"
    assert [t for t, _, _, _ in hits] == pytest.approx([-5.0, -3.0, 3.0, 5.0])
    for t, p, (u, v), mult in hits:
        assert mult == 1
        assert torus.contains(p)
        uu, vv, dist = torus.project_point(p)
        assert dist == pytest.approx(0.0)
        assert (u, v) == pytest.approx((uu, vv))

    kind, hits = Line3D(Point3D(0.0, 5.0, 0.0), Vector3D.x()).intersect(torus)
    assert kind == "hits"
    assert len(hits) == 1
    assert hits[0][0] == pytest.approx(0.0)
    assert hits[0][3] == 2

    assert Line3D(Point3D(0.0, 6.0, 0.0), Vector3D.x()).intersect(torus) == (
        "empty",
        None,
    )


def test_ellipse_ellipse_intersections():
    from geomcore.curves import Ellipse3D

    e1 = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    e2 = Ellipse3D(Point3D(0.5, 0.0, 0.0), Vector3D.z(), Vector3D.x(), 2.0, 2.0)
    kind, hits = e1.intersect(e2)
    assert kind == "hits"
    assert len(hits) == 4
    for s, p, t, _ in hits:
        assert e1.contains(p)
        assert e2.contains(p)

    grazing = Ellipse3D(Point3D(5.0, 0.0, 0.0), Vector3D.z(), Vector3D.x(), 2.0, 2.0)
    kind, [(s, p, t, mult)] = e1.intersect(grazing)
    assert kind == "hits"
    assert mult == 2
    assert (p.x, p.y, p.z) == pytest.approx((3.0, 0.0, 0.0))

    far = Ellipse3D(Point3D(10.0, 0.0, 0.0), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    assert e1.intersect(far) == ("empty", None)
    assert e1.intersect(e1) == ("coincident", None)
    # Crossing planes: both pass through (0, ±1.5, 0).
    tilted = Ellipse3D(Point3D.origin(), Vector3D.x(), Vector3D.z(), 3.0, 1.5)
    kind, hits = e1.intersect(tilted)
    assert kind == "hits"
    assert sorted(p.y for _, p, _, _ in hits) == pytest.approx([-1.5, 1.5])
    lifted = Ellipse3D(Point3D(0.0, 0.0, 1.0), Vector3D.z(), Vector3D.x(), 3.0, 1.5)
    assert e1.intersect(lifted) == ("empty", None)


def test_conic_conic_intersections():
    from geomcore.curves import Ellipse3D, Hyperbola3D, Parabola3D

    o, z, x = Point3D.origin(), Vector3D.z(), Vector3D.x()
    ellipse = Ellipse3D(o, z, x, 3.0, 2.0)
    hyperbola = Hyperbola3D(o, z, x, 1.0, 0.5)
    parabola = Parabola3D(o, z, x, 1.0)
    # The branch x > 0 meets the ellipse at (√153/5, ±√128/10).
    kind, hits = ellipse.intersect(hyperbola)
    assert kind == "hits"
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
        kind, hits = a.intersect(b)
        assert kind == "hits" and hits, (a, b)
        for s, p, t, _ in hits:
            _assert_same_point(a.eval_point(s), p)
            _assert_same_point(b.eval_point(t), p)
        # The receiver's parameter comes first.
        kind, back = b.intersect(a)
        assert sorted(t for _, _, t, _ in hits) == pytest.approx(sorted(t for t, _, _, _ in back))
        assert sorted(s for s, _, _, _ in hits) == pytest.approx(sorted(s for _, _, s, _ in back))
    assert parabola.intersect(parabola) == ("coincident", None)
    lifted = Parabola3D(Point3D(0.0, 0.0, 1.0), z, x, 1.0)
    assert parabola.intersect(lifted) == ("empty", None)


def _rigid_motion():
    axis = Axis3D(Point3D(0.5, -1.0, 0.0), Vector3D(1.0, 2.0, 2.0))
    return Transform.translation(Vector3D(3.0, -1.0, 2.0)).then(Transform.rotation(axis, 0.7))


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
            u, d = curve.project_point(p)
            assert math.isnan(u) and math.isnan(d)
            assert curve.extrema(p) == []


def test_trimmed_curve_segment_and_arc():
    from geomcore.curves import Line3D, TrimmedCurve3D

    segment = TrimmedCurve3D.segment(Point3D.origin(), Point3D(3.0, 4.0, 0.0))
    assert segment.bounds() == (0.0, 5.0)
    assert isinstance(segment.basis(), Line3D)
    assert segment.end_point().y == pytest.approx(4.0)
    # Beyond the end: the projection clamps to it.
    u, d = segment.project_point(Point3D(6.0, 8.0, 0.0))
    assert u == pytest.approx(5.0)
    assert d == pytest.approx(5.0)
    assert not segment.contains(Point3D(6.0, 8.0, 0.0))

    # The right half of the unit circle, across the seam at angle 0.
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    arc = TrimmedCurve3D(circle, 1.5 * math.pi, 2.5 * math.pi)
    assert isinstance(arc.basis(), Circle3D)
    u, d = arc.project_point(Point3D(3.0, 0.0, 0.0))
    assert u == pytest.approx(2.0 * math.pi)
    assert d == pytest.approx(2.0)
    u, _ = arc.project_point(Point3D(-3.0, 0.1, 0.0))
    assert u == pytest.approx(2.5 * math.pi)
    assert arc.contains(Point3D(0.0, 1.0, 0.0))
    assert not arc.contains(Point3D(-1.0, 0.0, 0.0))
    assert [e[0] for e in arc.extrema(Point3D(3.0, 0.0, 0.0))][0] == pytest.approx(2.0 * math.pi)


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


def test_generic_functions_search_only_the_range():
    from geomcore import curve_curve_extrema, intersect_curve_curve, intersect_curve_surface
    from geomcore.curves import TrimmedCurve3D

    segment = TrimmedCurve3D.segment(Point3D(-2.0, 0.5, 0.0), Point3D(2.0, 0.5, 0.0))
    circle = Circle3D(Point3D.origin(), Vector3D.z(), 1.0)
    assert len(intersect_curve_curve(segment, circle)) == 2
    quarter = TrimmedCurve3D(circle, 0.0, math.pi / 2)
    hits = intersect_curve_curve(segment, quarter)
    assert len(hits) == 1
    assert hits[0][1].x > 0.0

    a = TrimmedCurve3D.segment(Point3D.origin(), Point3D(1.0, 0.0, 0.0))
    b = TrimmedCurve3D.segment(Point3D(3.0, -1.0, 1.0), Point3D(3.0, 1.0, 1.0))
    assert curve_curve_extrema(a, b)[0][2] == pytest.approx(math.sqrt(5.0))

    sphere = Sphere(Point3D.origin(), 1.0)
    half = TrimmedCurve3D.segment(Point3D.origin(), Point3D(2.0, 0.0, 0.0))
    hits = intersect_curve_surface(half, sphere)
    assert len(hits) == 1
    assert hits[0][0] == pytest.approx(1.0)


def test_trimmed_curve_2d():
    from geomcore.curves import Circle2D, TrimmedCurve2D

    arc = TrimmedCurve2D(Circle2D(Point2D.origin(), 2.0), 0.0, math.pi / 2)
    assert isinstance(arc.basis(), Circle2D)
    u, _ = arc.project_point(Point2D(1.0, 1.0))
    assert u == pytest.approx(math.pi / 4)
    u, _ = arc.project_point(Point2D(-1.0, -5.0))
    assert u == 0.0
    assert arc.contains(Point2D(0.0, 2.0))
    assert not arc.contains(Point2D(-2.0, 0.0))

    segment = TrimmedCurve2D.segment(Point2D.origin(), Point2D(0.0, 2.0))
    assert isinstance(segment.basis(), Line2D)
    assert segment.bounds() == (0.0, 2.0)
    with pytest.raises(TypeError):
        TrimmedCurve2D(Circle3D(Point3D.origin(), Vector3D.z(), 1.0), 0.0, 1.0)


# Per-pair method suffix for each shape class.
PAIR_NOUN = {
    "Plane": "plane",
    "Cylinder": "cylinder",
    "Cone": "cone",
    "Sphere": "sphere",
    "Torus": "torus",
    "Line3D": "line",
    "Circle3D": "circle",
    "Ellipse3D": "ellipse",
    "Parabola3D": "parabola",
    "Hyperbola3D": "hyperbola",
    "Line2D": "line",
    "Circle2D": "circle",
}


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


def _comparable(value):
    """Turns a result payload into plain data; curves become sample points."""
    if isinstance(value, (tuple, list)):
        return tuple(_comparable(v) for v in value)
    if hasattr(value, "eval_point"):
        return (type(value).__name__, repr(value.eval_point(0.3)))
    if isinstance(value, (Point2D, Point3D)):
        return repr(value)
    return value


def _per_pair(a, b):
    """The deprecated per-pair method on `a` that takes `b`, if any."""
    return getattr(a, "intersect_" + PAIR_NOUN[type(b).__name__], None)


CONICS = {"Circle3D", "Ellipse3D", "Parabola3D", "Hyperbola3D"}


def _is_conic_pair(a, b):
    """Whether the pair returns conic-conic hits, receiver's parameter first."""
    names = {type(a).__name__, type(b).__name__}
    return names <= CONICS and names != {"Circle3D"}


def _rounded(p):
    return (round(p.x, 9), round(p.y, 9), round(p.z, 9))


def _receiver_swapped(result, swap=True):
    """Conic-conic hits as `(other, point, receiver, mult)`, sorted."""
    kind, hits = result
    if kind != "hits":
        return result
    return (
        kind,
        sorted(
            (round(t if swap else s, 9), _rounded(p), round(s if swap else t, 9), m)
            for s, p, t, m in hits
        ),
    )


def test_intersect_matches_per_pair_methods_in_both_orders():
    shapes = _intersect_shapes()
    pairs = 0
    for a in shapes:
        for b in shapes:
            if ("2D" in type(a).__name__) != ("2D" in type(b).__name__):
                continue
            method = _per_pair(a, b)
            if method is None:
                continue
            with pytest.warns(DeprecationWarning):
                expected = _comparable(method(b))
            assert _comparable(a.intersect(b)) == expected, (a, b)
            if _is_conic_pair(a, b):
                # The receiver's parameter comes first.
                assert _receiver_swapped(b.intersect(a)) == _receiver_swapped(
                    a.intersect(b), swap=False
                ), (b, a)
            else:
                assert _comparable(b.intersect(a)) == expected, (b, a)
            pairs += 1
    assert pairs == 49


def test_intersect_rejects_pairs_without_exact_solver():
    from geomcore.curves import Circle2D, Ellipse3D
    from geomcore.surfaces import Torus

    o, z, x = Point3D.origin(), Vector3D.z(), Vector3D.x()
    torus = Torus(o, z, 4.0, 1.0)
    circle = Circle3D(o, z, 2.5)
    for a, b in [
        (circle, torus),
        (Ellipse3D(o, z, x, 3.0, 1.5), torus),
        (circle, Circle2D(Point2D.origin(), 1.0)),
        (circle, BSplineCurve3D.interpolate([o, Point3D(1.0, 1.0, 0.0), Point3D(2.0, 0.0, 0.0)], 2)),
        (circle, 5),
    ]:
        with pytest.raises(TypeError, match=r"\.intersect supports"):
            a.intersect(b)
        if hasattr(b, "intersect"):
            with pytest.raises(TypeError):
                b.intersect(a)


def test_per_pair_intersect_methods_warn():
    with pytest.warns(DeprecationWarning, match="Sphere.intersect_sphere is deprecated"):
        Sphere(Point3D.origin(), 1.0).intersect_sphere(Sphere(Point3D(1.0, 0.0, 0.0), 1.0))


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
