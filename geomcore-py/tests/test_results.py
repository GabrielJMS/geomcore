"""Intersection results: the routed `intersect` and the classes it returns."""

import math

import pytest

from geomcore import (
    Completeness,
    CompletenessStatus,
    CurveCurveExtrema,
    CurveCurveHit,
    CurveCurveHit2D,
    CurveCurveIntersection,
    CurveCurveIntersection2D,
    CurveCurveOverlap,
    CurveCurveRegion,
    CurveSurfaceHit,
    CurveSurfaceIntersection,
    CurveSurfaceRegion,
    Frame3D,
    IncompleteReason,
    IntersectionCurve,
    IntersectionMethod,
    Point2D,
    Point3D,
    SurfaceSurfaceIntersection,
    Vector2D,
    Vector3D,
    curve_curve_extrema,
)
from geomcore.curves import (
    BSplineCurve3D,
    Circle2D,
    Circle3D,
    Line2D,
    Line3D,
    TrimmedCurve2D,
    TrimmedCurve3D,
)
from geomcore.surfaces import BSplineSurface, Cylinder, Plane, Sphere

TAU = 2.0 * math.pi


def _square():
    """The flat square [0, 2] x [0, 2] in z = 0, as a bilinear B-spline."""
    poles = [
        [Point3D(0.0, 0.0, 0.0), Point3D(0.0, 2.0, 0.0)],
        [Point3D(2.0, 0.0, 0.0), Point3D(2.0, 2.0, 0.0)],
    ]
    return BSplineSurface(1, 1, poles, [0.0, 1.0], [2, 2], [0.0, 1.0], [2, 2], False, False)


def _touching_parabola():
    """A parabola touching the square at (1, 1, 0), at t = 0.5."""
    poles = [Point3D(0.5, 1.0, 1.0), Point3D(1.0, 1.0, -1.0), Point3D(1.5, 1.0, 1.0)]
    return BSplineCurve3D(2, poles, [0.0, 1.0], [3, 3], False)


def _segment(start, end):
    return BSplineCurve3D(1, [start, end], [0.0, 1.0], [2, 2], False)


# ------------------------------------------------------------------ method


def test_closed_form_pair_is_exact_and_complete():
    line = Line3D(Point3D(-5.0, 0.0, 0.0), Vector3D.unit_x())
    result = line.intersect(Sphere(Point3D.origin(), 2.0))
    assert isinstance(result, CurveSurfaceIntersection)
    assert result.method == IntersectionMethod.EXACT
    assert result.is_complete()
    assert result.completeness.status == CompletenessStatus.COMPLETE
    assert result.completeness.reason is None
    assert result.completeness.regions == []
    assert [hit.point.x for hit in result.hits] == pytest.approx([-2.0, 2.0])


def test_bspline_pair_is_numeric():
    curve = _segment(Point3D(0.5, 0.5, -1.0), Point3D(0.5, 0.5, 1.0))
    result = curve.intersect(Plane(Point3D.origin(), Vector3D.unit_z()))
    assert result.method == IntersectionMethod.NUMERIC
    (hit,) = result.hits
    assert isinstance(hit, CurveSurfaceHit)
    assert hit.curve_parameter == pytest.approx(0.5)
    assert hit.point.z == pytest.approx(0.0, abs=1e-9)
    assert hit.multiplicity == 1


def test_bspline_surfaces_march_and_are_sampled():
    square = _square()
    plane = Plane(Point3D(1.0, 0.0, 0.0), Vector3D.unit_x())
    result = square.intersect(plane)
    assert isinstance(result, SurfaceSurfaceIntersection)
    assert result.method == IntersectionMethod.NUMERIC
    assert result.completeness.status == CompletenessStatus.SAMPLED
    assert not result.is_complete()
    assert len(result.curves) == 1
    piece = result.curves[0]
    assert isinstance(piece, IntersectionCurve)
    assert piece.tangent is False
    assert piece.curve.eval_point(sum(piece.curve.bounds()) / 2).x == pytest.approx(1.0)


# ------------------------------------------------------------ completeness


def test_tangent_curve_surface_contact_is_incomplete_with_a_region():
    result = _touching_parabola().intersect(_square())
    completeness = result.completeness
    assert isinstance(completeness, Completeness)
    assert completeness.status == CompletenessStatus.INCOMPLETE
    assert not completeness.is_complete()
    assert completeness.reason == IncompleteReason.UNSEPARATED
    (region,) = completeness.regions
    assert isinstance(region, CurveSurfaceRegion)
    t0, t1 = region.curve_range
    assert t0 <= 0.5 <= t1 and t1 - t0 < 1e-2
    assert len(region.u_range) == 2 and len(region.v_range) == 2
    assert region.min_corner.x <= 1.0 <= region.max_corner.x


def test_tangent_curve_curve_contact_is_incomplete_with_a_region():
    parabola = BSplineCurve3D(
        2,
        [Point3D(0.0, 1.0, 0.0), Point3D(2.0, -1.0, 0.0), Point3D(4.0, 1.0, 0.0)],
        [0.0, 1.0],
        [3, 3],
        False,
    )
    segment = _segment(Point3D.origin(), Point3D(4.0, 0.0, 0.0))
    result = parabola.intersect(segment)
    assert isinstance(result, CurveCurveIntersection)
    assert result.completeness.status == CompletenessStatus.INCOMPLETE
    (region,) = result.completeness.regions
    assert isinstance(region, CurveCurveRegion)
    s0, s1 = region.first_range
    assert s0 <= 0.5 <= s1 and s1 - s0 < 1e-2
    assert len(region.second_range) == 2
    assert region.min_corner.x <= 2.0 <= region.max_corner.x
    # The sampled search, run as well, finds the contact.
    assert len(result.hits) == 1


# -------------------------------------------------------------- operands


def test_results_hold_the_operands_passed():
    a = Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0)
    b = Plane(Point3D.origin(), Vector3D.unit_x())
    result = a.intersect(b)
    assert result.first is a
    assert result.second is b
    surfaces = Sphere(Point3D.origin(), 2.0).intersect(b)
    assert surfaces.second is b


def test_swapped_curve_surface_order_gives_the_same_hits():
    line = Line3D(Point3D(-5.0, 0.0, 0.0), Vector3D.unit_x())
    sphere = Sphere(Point3D.origin(), 2.0)
    forward, backward = line.intersect(sphere), sphere.intersect(line)
    assert backward.first is sphere and backward.second is line
    assert backward.hits == forward.hits
    assert backward.hits[0].curve_parameter == pytest.approx(3.0)
    # `curve` and `surface` name the operands by role, in either order.
    assert forward.curve is line and forward.surface is sphere
    assert backward.curve is line and backward.surface is sphere


def test_swapped_curve_curve_order_swaps_the_parameters():
    line = Line3D(Point3D(-5.0, 0.0, 0.0), Vector3D.unit_x())
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
    forward, backward = line.intersect(circle), circle.intersect(line)
    assert len(forward.hits) == len(backward.hits) == 2
    for hit in forward.hits:
        assert isinstance(hit, CurveCurveHit)
        match = [h for h in backward.hits if h.point.distance(hit.point) < 1e-9]
        assert len(match) == 1
        assert match[0].first_parameter == pytest.approx(hit.second_parameter)
        assert match[0].second_parameter == pytest.approx(hit.first_parameter)


# ------------------------------------------------------------- trimmed


def test_trimmed_arc_across_the_seam_keeps_both_hits():
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0)
    arc = TrimmedCurve3D(circle, 1.5 * math.pi, 2.5 * math.pi)
    result = arc.intersect(Plane(Point3D(0.5, 0.0, 0.0), Vector3D.unit_x()))
    assert result.first is arc
    assert len(result.hits) == 2
    for hit in result.hits:
        assert hit.point.x == pytest.approx(0.5)
    parameters = [hit.curve_parameter for hit in result.hits if hit.curve_parameter is not None]
    for t in parameters:
        assert 1.5 * math.pi <= t <= 2.5 * math.pi
    # Half the circle is outside the arc: a plane there misses it.
    miss = arc.intersect(Plane(Point3D(-0.5, 0.0, 0.0), Vector3D.unit_x()))
    assert miss.is_empty()


def test_trimmed_2d_curves_intersect():
    circle = Circle2D(Point2D(0.0, 0.0), 1.0)
    upper = TrimmedCurve2D(circle, 0.0, math.pi)
    result = upper.intersect(Line2D(Point2D(-2.0, 0.5), Vector2D(1.0, 0.0)))
    assert isinstance(result, CurveCurveIntersection2D)
    assert result.method == IntersectionMethod.EXACT
    assert len(result.hits) == 2
    for hit in result.hits:
        assert isinstance(hit, CurveCurveHit2D)
        assert isinstance(hit.point, Point2D)
        assert hit.point.y == pytest.approx(0.5)


# ------------------------------------------------------------- overlaps


def test_curve_lying_on_a_surface_is_an_overlap():
    sphere = Sphere(Point3D.origin(), 2.0)
    equator = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
    result = equator.intersect(sphere)
    assert result.hits == []
    assert result.overlaps == [pytest.approx((0.0, TAU))]
    assert not result.is_empty()


def test_coincident_curves_are_an_overlap():
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0)
    result = circle.intersect(Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0))
    (overlap,) = result.overlaps
    assert isinstance(overlap, CurveCurveOverlap)
    assert overlap.first_range == pytest.approx((0.0, TAU))
    assert overlap.second_range == pytest.approx((0.0, TAU))
    assert overlap.same_direction is True
    assert result.hits == []


def test_coincident_surfaces():
    plane = Plane(Point3D.origin(), Vector3D.unit_z())
    result = plane.intersect(Plane(Point3D.origin(), Vector3D.unit_z()))
    assert result.is_coincident()
    assert not result.is_empty()
    assert result.curves == []


# ------------------------------------------------------------- is_empty


def test_missing_pairs_are_empty_and_complete():
    sphere = Sphere(Point3D.origin(), 1.0)
    far = Plane(Point3D(0.0, 0.0, 5.0), Vector3D.unit_z())
    for result in (sphere.intersect(far), Line3D(Point3D(0.0, 0.0, 5.0), Vector3D.unit_x()).intersect(sphere)):
        assert result.is_empty()
        assert result.is_complete()


def test_tangent_plane_gives_a_contact_point():
    sphere = Sphere(Point3D.origin(), 2.0)
    result = Plane(Point3D(0.0, 0.0, 2.0), Vector3D.unit_z()).intersect(sphere)
    assert result.curves == []
    (point,) = result.contact_points
    assert point.distance(Point3D(0.0, 0.0, 2.0)) < 1e-9
    assert not result.is_empty()


def test_surface_surface_curves_are_geomcore_curves():
    result = Plane(Point3D.origin(), Vector3D.unit_z()).intersect(Sphere(Point3D.origin(), 2.0))
    assert result.method == IntersectionMethod.EXACT
    (piece,) = result.curves
    assert isinstance(piece.curve, Circle3D)
    assert piece.curve.radius() == pytest.approx(2.0)


# ---------------------------------------------------------------- types


def test_every_curve_and_surface_class_accepts_every_other():
    operands = [
        Line3D(Point3D(0.0, 0.0, -3.0), Vector3D.unit_z()),
        Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0),
        _segment(Point3D(-2.0, 0.1, 0.1), Point3D(2.0, 0.1, 0.1)),
        TrimmedCurve3D.segment(Point3D(0.0, -2.0, 0.0), Point3D(0.0, 2.0, 0.0)),
        Plane(Point3D.origin(), Vector3D.unit_x()),
        Cylinder(Point3D.origin(), Vector3D.unit_y(), 0.5),
        _square(),
    ]
    for a in operands:
        for b in operands:
            result = a.intersect(b)
            assert result.first is a and result.second is b


def test_mixed_dimensions_and_foreign_objects_raise_type_error():
    line2 = Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0))
    line3 = Line3D(Point3D.origin(), Vector3D.unit_x())
    with pytest.raises(TypeError):
        line2.intersect(line3)
    with pytest.raises(TypeError):
        line3.intersect(line2)
    with pytest.raises(TypeError):
        line3.intersect("a string")


def test_removed_apis_are_gone():
    import geomcore

    for name in ("intersect_marching", "intersect_curve_surface", "intersect_curve_curve"):
        assert not hasattr(geomcore, name)
    assert not hasattr(Sphere, "intersect_sphere")
    assert not hasattr(Line3D, "intersect_plane")


# --------------------------------------------------------------- extrema


def test_curve_curve_extrema_is_an_object():
    arch = BSplineCurve3D(
        2,
        [Point3D(0.0, 0.0, 0.0), Point3D(1.0, 2.0, 0.0), Point3D(2.0, 0.0, 0.0)],
        [0.0, 1.0],
        [3, 3],
        False,
    )
    rail = _segment(Point3D(1.0, 0.0, 1.0), Point3D(1.0, 2.0, 1.0))
    extrema = curve_curve_extrema(arch, rail)
    assert isinstance(extrema, CurveCurveExtrema)
    assert extrema.first is arch and extrema.second is rail
    assert extrema.is_complete()
    assert extrema.completeness.status == CompletenessStatus.COMPLETE
    (minimum,) = extrema.minima
    assert minimum.distance == pytest.approx(1.0)
    assert minimum.first_parameter == pytest.approx(0.5)
    assert 0.0 <= minimum.second_parameter <= 1.0
    assert curve_curve_extrema(first=arch, second=rail) == extrema


# ------------------------------------------------------- repr and equality


def test_repr_names_the_class_and_its_fields():
    line = Line3D(Point3D(-5.0, 0.0, 0.0), Vector3D.unit_x())
    result = line.intersect(Sphere(Point3D.origin(), 2.0))
    text = repr(result)
    assert text.startswith("CurveSurfaceIntersection(")
    assert "IntersectionMethod.EXACT" in text
    assert "CompletenessStatus.COMPLETE" in text
    hit = result.hits[0]
    assert repr(hit).startswith("CurveSurfaceHit(curve_parameter=")
    assert repr(hit.point) in repr(hit)
    assert "multiplicity=1" in repr(hit)
    assert repr(result.completeness) == "Completeness(status=CompletenessStatus.COMPLETE)"

    overlap = Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0).intersect(
        Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0)
    ).overlaps[0]
    assert repr(overlap).startswith("CurveCurveOverlap(first_range=(")
    assert "same_direction=True" in repr(overlap)

    region = _touching_parabola().intersect(_square()).completeness.regions[0]
    assert repr(region).startswith("CurveSurfaceRegion(curve_range=(")


def test_results_compare_by_value():
    line = Line3D(Point3D(-5.0, 0.0, 0.0), Vector3D.unit_x())
    sphere = Sphere(Point3D.origin(), 2.0)
    assert line.intersect(sphere) == line.intersect(sphere)
    assert line.intersect(sphere).hits[0] == line.intersect(sphere).hits[0]
    assert line.intersect(sphere) != line.intersect(Sphere(Point3D.origin(), 3.0))
    frame = Frame3D.world()
    assert line.intersect(sphere) != frame


# ------------------------------------------------------- swapped, numeric


def test_swapped_numeric_curve_curve_order_swaps_both_parameters():
    spline = BSplineCurve3D(
        2,
        [Point3D(0.0, -1.0, 0.0), Point3D(1.0, 2.0, 0.0), Point3D(2.0, -1.0, 0.0)],
        [0.0, 1.0],
        [3, 3],
        False,
    )
    line = Line3D(Point3D(-1.0, 0.0, 0.0), Vector3D.unit_x())
    forward, backward = spline.intersect(line), line.intersect(spline)
    assert forward.method == backward.method == IntersectionMethod.NUMERIC
    assert len(forward.hits) == len(backward.hits) == 2
    for hit in forward.hits:
        assert hit.first_parameter is not None and hit.second_parameter is not None
        match = [h for h in backward.hits if h.point.distance(hit.point) < 1e-9]
        assert len(match) == 1
        assert match[0].first_parameter == pytest.approx(hit.second_parameter)
        assert match[0].second_parameter == pytest.approx(hit.first_parameter)


# ------------------------------------------------------------------- 2D


def test_swapped_2d_order_gives_the_same_points():
    circle = Circle2D(Point2D(0.0, 0.0), 1.0)
    line = Line2D(Point2D(-2.0, 0.5), Vector2D(1.0, 0.0))
    forward, backward = circle.intersect(line), line.intersect(circle)
    assert backward.first is line and backward.second is circle
    assert len(forward.hits) == len(backward.hits) == 2
    for hit in forward.hits:
        assert any(h.point.distance(hit.point) < 1e-9 for h in backward.hits)


def test_coincident_2d_circles_are_an_overlap():
    circle = Circle2D(Point2D(1.0, 2.0), 1.0)
    result = circle.intersect(Circle2D(Point2D(1.0, 2.0), 1.0))
    (overlap,) = result.overlaps
    assert overlap.first_range == pytest.approx((0.0, TAU))
    assert overlap.same_direction is True
    assert result.hits == []


def test_collinear_opposite_2d_lines_overlap_in_opposite_directions():
    line = Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0))
    opposite = Line2D(Point2D(3.0, 0.0), Vector2D(-1.0, 0.0))
    (overlap,) = line.intersect(opposite).overlaps
    assert overlap.first_range == (-math.inf, math.inf)
    assert overlap.same_direction is False


# ---------------------------------------------------- python conventions


def test_enums_are_hashable_and_named():
    assert IntersectionMethod.EXACT.name == "EXACT"
    assert CompletenessStatus.SAMPLED.name == "SAMPLED"
    assert IncompleteReason.UNSEPARATED.name == "UNSEPARATED"
    counts = {IntersectionMethod.EXACT: 1, IntersectionMethod.NUMERIC: 2}
    assert counts[IntersectionMethod.NUMERIC] == 2
    assert len({CompletenessStatus.COMPLETE, CompletenessStatus.COMPLETE}) == 1
    assert hash(IncompleteReason.BUDGET_EXHAUSTED) == hash(IncompleteReason.BUDGET_EXHAUSTED)


def test_empty_results_are_falsy():
    sphere = Sphere(Point3D.origin(), 1.0)
    assert Line3D(Point3D.origin(), Vector3D.unit_x()).intersect(sphere)
    assert not Line3D(Point3D(0.0, 0.0, 5.0), Vector3D.unit_x()).intersect(sphere)
    assert sphere.intersect(Plane(Point3D.origin(), Vector3D.unit_z()))
    assert not sphere.intersect(Plane(Point3D(0.0, 0.0, 5.0), Vector3D.unit_z()))
    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 1.0)
    assert circle.intersect(Line3D(Point3D.origin(), Vector3D.unit_x()))
    assert not circle.intersect(Line3D(Point3D(0.0, 0.0, 5.0), Vector3D.unit_x()))
    circle_2d = Circle2D(Point2D(0.0, 0.0), 1.0)
    assert circle_2d.intersect(Line2D(Point2D(0.0, 0.0), Vector2D(1.0, 0.0)))
    assert not circle_2d.intersect(Line2D(Point2D(0.0, 5.0), Vector2D(1.0, 0.0)))


def test_extrema_length_is_the_number_of_minima():
    a = _segment(Point3D.origin(), Point3D(1.0, 0.0, 0.0))
    b = _segment(Point3D(0.0, 1.0, 0.0), Point3D(1.0, 2.0, 0.0))
    extrema = curve_curve_extrema(a, b)
    assert len(extrema) == len(extrema.minima) >= 1
    assert extrema


def test_curves_and_surfaces_have_deterministic_reprs():
    circle = Circle3D(Point3D(1.0, 0.0, 0.0), Vector3D.unit_z(), 2.5)
    assert repr(circle) == "Circle3D(center=(1.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), radius=2.5)"
    assert repr(Sphere(Point3D.origin(), 3.0)) == "Sphere(center=(0.0, 0.0, 0.0), radius=3.0)"
    assert repr(Plane(Point3D.origin(), Vector3D.unit_z())) == "Plane(origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))"
    result = Plane(Point3D.origin(), Vector3D.unit_z()).intersect(Sphere(Point3D.origin(), 3.0))
    assert "Circle3D(center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), radius=3.0)" in repr(result)
    assert " at 0x" not in repr(result)


# ------------------------------------------------- numeric overlaps, threads


def test_numeric_curve_lying_on_a_surface_is_an_overlap():
    # A straight B-spline lying on the square: one overlap, no string of
    # hits, and the result stays incomplete over it.
    segment = _segment(Point3D(0.5, 0.5, 0.0), Point3D(1.5, 1.0, 0.0))
    result = segment.intersect(_square())
    assert result.method == IntersectionMethod.NUMERIC
    assert result.hits == []
    assert result.overlaps == [(0.0, 1.0)]
    assert result.completeness.status == CompletenessStatus.INCOMPLETE
    assert result.completeness.reason == IncompleteReason.UNSEPARATED


def test_numeric_collinear_curves_are_an_overlap():
    first = _segment(Point3D(2.0, 0.0, 0.0), Point3D(4.0, 0.0, 0.0))
    line = Line3D(Point3D.origin(), Vector3D.unit_x())
    for result, (first_range, second_range) in [
        (first.intersect(line), ((0.0, 1.0), (2.0, 4.0))),
        (line.intersect(first), ((2.0, 4.0), (0.0, 1.0))),
    ]:
        assert result.hits == []
        (overlap,) = result.overlaps
        assert overlap.first_range == pytest.approx(first_range, abs=1e-6)
        assert overlap.second_range == pytest.approx(second_range, abs=1e-6)
        assert overlap.same_direction


def test_intersections_run_from_several_threads():
    from concurrent.futures import ThreadPoolExecutor

    square = _square()
    curves = [_touching_parabola(), _segment(Point3D(0.0, 1.0, -1.0), Point3D(2.0, 1.0, 1.0))]
    expected = [curve.intersect(square) for curve in curves]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda k: curves[k % 2].intersect(square), range(16)))
    assert results == [expected[k % 2] for k in range(16)]
