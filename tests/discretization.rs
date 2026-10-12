//! Discretization through its public interface: every reported deviation
//! is checked against an independent measurement of the true one.

use geomcore::curves::{ParametricCurve2D, ParametricCurve3D};
use geomcore::discretization::{
    Discretization2D, Discretization3D, DiscretizationCriteria, DiscretizationCriteriaError,
    Discretize2D, Discretize3D,
};
use geomcore::{
    BSplineCurve3D, Circle2D, Circle3D, Curve2D, Curve3D, Ellipse3D, Frame2D, Hyperbola3D, Line2D,
    Line3D, Parabola3D, Point2D, Point3D, TrimmedCurve2D, TrimmedCurve3D, Vector2D, Vector3D,
};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

// ---------------------------------------------------------------- checker

/// The distance from `p` to the segment `[a, b]`.
fn distance_to_segment(p: Point3D, a: Point3D, b: Point3D) -> f64 {
    let ab = b - a;
    let length2 = ab.dot(ab);
    let t = if length2 > 0.0 {
        ((p - a).dot(ab) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    p.distance(a + ab * t)
}

/// The angle between the curve's tangent at `u` and the segment `[a, b]`.
fn angle_to_segment<C: ParametricCurve3D>(curve: &C, u: f64, a: Point3D, b: Point3D) -> f64 {
    let tangent = curve.eval_derivative(u, 1);
    let chord = b - a;
    tangent.cross(chord).magnitude().atan2(tangent.dot(chord))
}

/// The largest value of `f` over `[a, b]`: dense samples, then a
/// golden-section refinement around the best one.
fn maximum(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    const SAMPLES: usize = 64;
    let step = (b - a) / SAMPLES as f64;
    let mut best = (f(a), a);
    for i in 1..=SAMPLES {
        let u = if i == SAMPLES { b } else { a + step * i as f64 };
        let value = f(u);
        if value > best.0 {
            best = (value, u);
        }
    }
    let (mut lo, mut hi) = ((best.1 - step).max(a), (best.1 + step).min(b));
    let ratio = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..60 {
        let (x1, x2) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
        if f(x1) > f(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    best.0.max(f(0.5 * (lo + hi)))
}

/// The true chord and angle deviation of each segment, measured on the
/// curve itself.
fn true_deviations<C: ParametricCurve3D>(curve: &C, d: &Discretization3D) -> Vec<(f64, f64)> {
    (0..d.points.len() - 1)
        .map(|i| {
            let (u0, u1) = (d.parameters[i], d.parameters[i + 1]);
            let (a, b) = (d.points[i], d.points[i + 1]);
            let chord = maximum(u0, u1, |u| distance_to_segment(curve.eval_point(u), a, b));
            let angle = maximum(u0, u1, |u| angle_to_segment(curve, u, a, b));
            (chord, angle)
        })
        .collect()
}

/// Checks the shape of a discretization and that every reported deviation
/// is proven (at least the true one) and meets the criteria. `scale` is
/// the curve's size, for the rounding slack of the measurement.
fn assert_proven<C: ParametricCurve3D>(
    curve: &C,
    d: &Discretization3D,
    criteria: DiscretizationCriteria,
    scale: f64,
) {
    let n = d.points.len();
    assert!(n >= 2);
    assert_eq!(d.parameters.len(), n);
    assert_eq!(d.chord_deviations.len(), n - 1);
    assert_eq!(d.angle_deviations.len(), n - 1);
    let (first, last) = curve.bounds();
    assert_eq!(d.parameters[0], first);
    assert_eq!(d.parameters[n - 1], last);
    assert!(d.parameters.windows(2).all(|w| w[0] < w[1]));
    for (u, p) in d.parameters.iter().zip(&d.points) {
        assert!(curve.eval_point(*u).distance(*p) <= 1e-12 * scale);
    }
    let max_angle = criteria.max_angle_deviation().unwrap_or(PI);
    for (i, (chord, angle)) in true_deviations(curve, d).into_iter().enumerate() {
        let (reported_chord, reported_angle) = (d.chord_deviations[i], d.angle_deviations[i]);
        assert!(
            reported_chord >= chord - 1e-12 * scale,
            "segment {i}: chord deviation {reported_chord} below the true {chord}"
        );
        assert!(reported_chord <= criteria.max_chord_deviation());
        assert!(
            reported_angle >= angle - 1e-9,
            "segment {i}: angle deviation {reported_angle} below the true {angle}"
        );
        assert!(reported_angle <= max_angle || reported_angle == PI);
    }
}

fn chord(max: f64) -> DiscretizationCriteria {
    DiscretizationCriteria::new(max).unwrap()
}

fn chord_and_angle(max: f64, angle: f64) -> DiscretizationCriteria {
    chord(max).with_max_angle_deviation(angle).unwrap()
}

// ---------------------------------------------------------------- criteria

#[test]
fn test_criteria_reject_non_positive_and_non_finite_values() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            DiscretizationCriteria::new(bad).err(),
            Some(DiscretizationCriteriaError::InvalidChordDeviation)
        );
        assert_eq!(
            chord(1.0).with_max_angle_deviation(bad).err(),
            Some(DiscretizationCriteriaError::InvalidAngleDeviation)
        );
    }
    let criteria = chord_and_angle(0.1, 0.2);
    assert_eq!(criteria.max_chord_deviation(), 0.1);
    assert_eq!(criteria.max_angle_deviation(), Some(0.2));
    assert_eq!(chord(0.1).max_angle_deviation(), None);
}

// ---------------------------------------------------------------- circles

fn circle(radius: f64) -> Circle3D {
    Circle3D::new(
        Point3D::new(1.0, -2.0, 0.5),
        Vector3D::new(1.0, 1.0, 1.0),
        radius,
    )
    .unwrap()
}

#[test]
fn test_circle_matches_the_closed_forms() {
    // A segment of arc angle θ deviates by r(1 − cos θ/2) and θ/2.
    let c = circle(2.0);
    // 2(1 − cos(π/8)) = 0.152…: 8 segments meet 0.16, 7 do not.
    let d = c.discretize(chord(0.16));
    assert_eq!(d.points.len(), 9);
    for (&dev, &angle) in d.chord_deviations.iter().zip(&d.angle_deviations) {
        assert!((dev - 2.0 * (1.0 - (PI / 8.0).cos())).abs() < 1e-15);
        assert!((angle - PI / 8.0).abs() < 1e-15);
    }
    // The angle criterion wins when it asks for more points: θ/2 ≤ 0.1.
    let d = c.discretize(chord_and_angle(0.16, 0.1));
    assert_eq!(d.points.len(), (TAU / 0.2).ceil() as usize + 1);
    assert_proven(&c, &d, chord_and_angle(0.16, 0.1), 2.0);
}

#[test]
fn test_full_circle_is_closed_with_at_least_three_segments() {
    let c = circle(1.0);
    let d = c.discretize(chord(1e-3));
    assert_eq!(d.points[0], d.points[d.points.len() - 1]);
    assert_proven(&c, &d, chord(1e-3), 1.0);
    // A deviation larger than the diameter still gives a triangle.
    let d = c.discretize(chord(10.0));
    assert_eq!(d.points.len(), 4);
    assert_eq!(d.points[0], d.points[3]);
}

#[test]
fn test_circle_arcs_whole_and_across_the_seam() {
    let c = circle(3.0);
    for (first, last) in [(0.0, FRAC_PI_2), (1.0, 4.0), (-1.0, 1.0), (5.0, 5.0 + TAU)] {
        let arc = TrimmedCurve3D::new(c, first, last).unwrap();
        for criteria in [chord(1e-2), chord(1e-4), chord_and_angle(1e-2, 0.05)] {
            let d = arc.discretize(criteria);
            assert_eq!(d.points[0], c.eval_point(first));
            assert_proven(&arc, &d, criteria, 3.0);
        }
    }
    // An arc shorter than the requested deviation is one segment.
    let tiny = TrimmedCurve3D::new(c, 0.0, 1e-4).unwrap();
    assert_eq!(tiny.discretize(chord(1e-3)).points.len(), 2);
}

// ---------------------------------------------------------------- B-splines

/// A wavy cubic over three spans of uneven length.
fn wavy() -> BSplineCurve3D {
    let poles = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 2.0, 0.5),
        Point3D::new(2.0, -1.0, 0.0),
        Point3D::new(3.0, 1.5, -0.5),
        Point3D::new(4.0, 0.0, 1.0),
        Point3D::new(5.0, 1.0, 0.0),
    ];
    BSplineCurve3D::new(3, poles, vec![0.0, 0.2, 1.0, 1.5], vec![4, 1, 1, 4], false).unwrap()
}

/// A rational cubic with weights from 0.2 to 8.
fn uneven_weights() -> BSplineCurve3D {
    let poles = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 3.0, 0.0),
        Point3D::new(3.0, 3.0, 1.0),
        Point3D::new(4.0, -1.0, 0.0),
        Point3D::new(6.0, 0.0, -1.0),
        Point3D::new(7.0, 2.0, 0.0),
    ];
    let weights = vec![1.0, 0.2, 8.0, 0.5, 3.0, 1.0];
    BSplineCurve3D::new_rational(3, poles, weights, vec![0.0, 1.0, 2.0], vec![4, 2, 4], false)
        .unwrap()
}

/// A rational quadratic whose weights make the centroid term of the
/// rational bound necessary.
fn skewed_weights() -> BSplineCurve3D {
    let poles = vec![
        Point3D::new(-0.7348247574079321, -0.4659857917540662, 0.0),
        Point3D::new(-0.8715469097018922, -0.6487383729544853, 0.0),
        Point3D::new(0.26508886772773677, 0.9783988954513023, 0.0),
    ];
    let weights = vec![1.656624523186388, 0.27100620195828673, 0.3570884064073742];
    BSplineCurve3D::new_rational(2, poles, weights, vec![0.0, 1.0], vec![3, 3], false).unwrap()
}

/// A unit circle as four rational quadratic arcs.
fn rational_circle() -> BSplineCurve3D {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let poles = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
    ]
    .map(|(x, y)| Point3D::new(x, y, 0.0))
    .to_vec();
    let weights = vec![1.0, h, 1.0, h, 1.0, h, 1.0, h, 1.0];
    BSplineCurve3D::new_rational(
        2,
        poles,
        weights,
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![3, 2, 2, 2, 3],
        false,
    )
    .unwrap()
}

/// A closed cubic over six spans.
fn periodic_ring() -> BSplineCurve3D {
    let poles = (0..6)
        .map(|i| {
            let a = i as f64 * TAU / 6.0;
            Point3D::new(
                2.0 * a.cos(),
                2.0 * a.sin(),
                if i % 2 == 0 { 0.3 } else { -0.3 },
            )
        })
        .collect();
    BSplineCurve3D::new(3, poles, (0..7).map(f64::from).collect(), vec![1; 7], true).unwrap()
}

#[test]
fn test_bspline_chord_deviations_are_proven() {
    for curve in [
        wavy(),
        uneven_weights(),
        skewed_weights(),
        rational_circle(),
        periodic_ring(),
    ] {
        for max in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-5] {
            let criteria = chord(max);
            let d = curve.discretize(criteria);
            assert_proven(&curve, &d, criteria, 8.0);
            assert_eq!(d.points[0], curve.eval_point(curve.bounds().0));
        }
    }
}

#[test]
fn test_trimmed_bsplines_including_across_the_seam() {
    let ring = periodic_ring();
    for (first, last) in [(0.5, 2.7), (-1.3, 0.4), (5.5, 7.0), (2.0, 8.0)] {
        let piece = TrimmedCurve3D::new(&ring, first, last).unwrap();
        let d = piece.discretize(chord(1e-4));
        assert_proven(&piece, &d, chord(1e-4), 4.0);
        assert_eq!(d.points[0], ring.eval_point(first));
        assert_eq!(
            d.points[d.points.len() - 1] == d.points[0],
            last - first == 6.0
        );
    }
    let curve = uneven_weights();
    let piece = TrimmedCurve3D::new(curve.clone(), 0.3, 1.9).unwrap();
    assert_proven(&piece, &piece.discretize(chord(1e-4)), chord(1e-4), 8.0);
}

#[test]
fn test_closed_bsplines_are_closed() {
    let d = periodic_ring().discretize(chord(1e-3));
    assert_eq!(d.points[0], d.points[d.points.len() - 1]);
    // Six spans give six segments, even for a huge deviation.
    assert_eq!(periodic_ring().discretize(chord(100.0)).points.len(), 7);
}

#[test]
fn test_degree_one_is_one_segment_per_span() {
    let poles = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1.0, 1.0, 0.0),
        Point3D::new(2.0, 0.0, 1.0),
        Point3D::new(4.0, 0.0, 0.0),
    ];
    let polyline = BSplineCurve3D::new(
        1,
        poles.clone(),
        vec![0.0, 1.0, 3.0, 4.0],
        vec![2, 1, 1, 2],
        false,
    )
    .unwrap();
    let d = polyline.discretize(chord(1e-6));
    assert_eq!(d.parameters, vec![0.0, 1.0, 3.0, 4.0]);
    assert_eq!(d.points, poles);
    assert_eq!(d.chord_deviations, vec![0.0; 3]);
}

#[test]
fn test_a_span_shorter_than_the_deviation_is_one_segment() {
    let poles = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(1e-4, 1e-4, 0.0),
        Point3D::new(2e-4, 0.0, 0.0),
    ];
    let small = BSplineCurve3D::new(2, poles, vec![0.0, 1.0], vec![3, 3], false).unwrap();
    assert_eq!(small.discretize(chord(1e-3)).points.len(), 2);
}

#[test]
fn test_borrowed_and_owned_trimmed_curves_agree() {
    let curve = uneven_weights();
    let owned = TrimmedCurve3D::new(curve.clone(), 0.2, 1.7).unwrap();
    let borrowed = TrimmedCurve3D::new(&curve, 0.2, 1.7).unwrap();
    let criteria = chord(1e-4);
    assert_eq!(owned.discretize(criteria), borrowed.discretize(criteria));
    let c = circle(2.0);
    let owned = TrimmedCurve3D::new(c, 1.0, 3.0).unwrap();
    let borrowed = TrimmedCurve3D::new(&c, 1.0, 3.0).unwrap();
    assert_eq!(owned.discretize(criteria), borrowed.discretize(criteria));
}

// ---------------------------------------------------------------- count and fill

#[test]
fn test_count_then_fill_into_part_of_a_larger_buffer() {
    let curve = wavy();
    let criteria = chord(1e-4);
    let n = curve.point_count(criteria);
    let expected = curve.discretize(criteria);
    assert_eq!(expected.points.len(), n);
    let (mut parameters, mut points) = (vec![-1.0; n + 10], vec![Point3D::ORIGIN; n + 10]);
    let (mut chords, mut angles) = (vec![-1.0; n + 10], vec![-1.0; n + 10]);
    curve.discretize_into(
        criteria,
        &mut parameters[5..5 + n],
        &mut points[5..5 + n],
        &mut chords[5..4 + n],
        &mut angles[5..4 + n],
    );
    assert_eq!(&parameters[5..5 + n], expected.parameters.as_slice());
    assert_eq!(&points[5..5 + n], expected.points.as_slice());
    assert_eq!(&chords[5..4 + n], expected.chord_deviations.as_slice());
    assert_eq!(&angles[5..4 + n], expected.angle_deviations.as_slice());
    assert_eq!(parameters[4], -1.0);
    assert_eq!(parameters[5 + n], -1.0);
}

#[test]
#[should_panic(expected = "wrong length")]
fn test_fill_panics_on_a_wrong_length() {
    let curve = wavy();
    let criteria = chord(1e-4);
    let n = curve.point_count(criteria);
    let (mut parameters, mut points) = (vec![0.0; n], vec![Point3D::ORIGIN; n + 1]);
    let (mut chords, mut angles) = (vec![0.0; n - 1], vec![0.0; n - 1]);
    curve.discretize_into(
        criteria,
        &mut parameters,
        &mut points,
        &mut chords,
        &mut angles,
    );
}

#[test]
fn test_same_input_gives_bit_identical_output() {
    for criteria in [chord(1e-5), chord_and_angle(1e-3, 0.1)] {
        let a = uneven_weights().discretize(criteria);
        let b = uneven_weights().discretize(criteria);
        assert!(
            a.parameters
                .iter()
                .zip(&b.parameters)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        assert_eq!(a, b);
    }
}

// ---------------------------------------------------------------- angle deviation

/// A cubic over one span from four planar poles.
fn planar_cubic(poles: [(f64, f64); 4]) -> BSplineCurve3D {
    let poles = poles.map(|(x, y)| Point3D::new(x, y, 0.0)).to_vec();
    BSplineCurve3D::new(3, poles, vec![0.0, 1.0], vec![4, 4], false).unwrap()
}

/// The segments reporting an angle deviation of π, as parameter ranges.
fn unbounded_segments(d: &Discretization3D) -> Vec<(f64, f64)> {
    (0..d.angle_deviations.len())
        .filter(|&i| d.angle_deviations[i] == PI)
        .map(|i| (d.parameters[i], d.parameters[i + 1]))
        .collect()
}

#[test]
fn test_nonrational_angle_deviations_are_proven() {
    for curve in [wavy(), periodic_ring()] {
        for criteria in [
            chord(1e-3),
            chord_and_angle(1e-2, 0.2),
            chord_and_angle(1e-4, 0.02),
        ] {
            let d = curve.discretize(criteria);
            assert_proven(&curve, &d, criteria, 8.0);
            assert!(unbounded_segments(&d).is_empty());
        }
    }
}

#[test]
fn test_a_span_turning_past_half_a_turn_meets_the_angle_limit() {
    // The tangent turns by about 270° within the one span.
    let loop_ = planar_cubic([(0.0, 0.0), (4.0, 3.0), (-4.0, 3.0), (0.5, 0.0)]);
    let criteria = chord_and_angle(1e-3, 0.1);
    let d = loop_.discretize(criteria);
    assert_proven(&loop_, &d, criteria, 4.0);
    assert!(unbounded_segments(&d).is_empty());
}

#[test]
fn test_a_cusp_reports_pi_only_next_to_it() {
    // The speed vanishes at u = 0.5.
    let cusp = planar_cubic([(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]);
    let criteria = chord_and_angle(1e-4, 0.1);
    let d = cusp.discretize(criteria);
    assert_proven(&cusp, &d, criteria, 1.0);
    let unbounded = unbounded_segments(&d);
    assert!(!unbounded.is_empty());
    assert!(
        unbounded.iter().all(|&(u0, u1)| u0 > 0.45 && u1 < 0.55),
        "{unbounded:?}"
    );
}

#[test]
fn test_coincident_poles_report_pi_only_at_that_end() {
    // The first two poles coincide: the speed vanishes at u = 0.
    let curve = planar_cubic([(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]);
    let criteria = chord_and_angle(1e-4, 0.1);
    let d = curve.discretize(criteria);
    assert_proven(&curve, &d, criteria, 2.0);
    let unbounded = unbounded_segments(&d);
    assert!(!unbounded.is_empty());
    assert!(unbounded.iter().all(|&(_, u1)| u1 < 0.05), "{unbounded:?}");
}

#[test]
fn test_degree_one_segments_have_no_angle_deviation() {
    let poles = vec![Point3D::new(0.0, 0.0, 0.0), Point3D::new(1.0, 1.0, 0.0)];
    let line = BSplineCurve3D::new(1, poles, vec![0.0, 1.0], vec![2, 2], false).unwrap();
    assert_eq!(line.discretize(chord(1e-3)).angle_deviations, vec![0.0]);
}

#[test]
fn test_rational_angle_deviations_are_proven() {
    for curve in [uneven_weights(), skewed_weights(), rational_circle()] {
        for criteria in [
            chord(1e-3),
            chord_and_angle(1e-2, 0.2),
            chord_and_angle(1e-4, 0.02),
        ] {
            let d = curve.discretize(criteria);
            assert_proven(&curve, &d, criteria, 8.0);
            // Only an angle limit halves a piece to find its bound: the
            // tight bend of the skewed weights needs it.
            if criteria.max_angle_deviation().is_some() {
                assert!(unbounded_segments(&d).is_empty());
            }
        }
    }
}

#[test]
fn test_a_rational_circle_needs_few_more_points_than_the_exact_one() {
    let exact = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 1.0).unwrap();
    let ratio = |criteria| {
        rational_circle().point_count(criteria) as f64 / exact.point_count(criteria) as f64
    };
    for max in [1e-2, 1e-4, 1e-6] {
        assert!(ratio(chord(max)) <= 1.4, "{max}: {}", ratio(chord(max)));
    }
    // A step's tangent and chord may differ by its whole turning, where a
    // circle's differ by half of it: an angle limit costs about twice the
    // exact circle's points.
    for (max, angle) in [(1e-2, 0.2), (1e-2, 0.05), (1e-5, 0.01)] {
        let r = ratio(chord_and_angle(max, angle));
        assert!(r <= 3.0, "{max}, {angle}: {r}");
    }
}

// ---------------------------------------------------------------- conics and lines

fn ellipse(major: f64, minor: f64) -> Ellipse3D {
    let normal = Vector3D::new(0.2, -0.3, 1.0);
    let x = Vector3D::new(1.0, 0.0, 0.0).cross(normal);
    Ellipse3D::new(Point3D::new(1.0, 2.0, 3.0), normal, x, major, minor).unwrap()
}

fn parabola(focal: f64) -> Parabola3D {
    let normal = Vector3D::new(0.0, 1.0, 1.0);
    Parabola3D::new(Point3D::new(-1.0, 0.5, 0.0), normal, Vector3D::X, focal).unwrap()
}

fn hyperbola(major: f64, minor: f64) -> Hyperbola3D {
    let normal = Vector3D::new(1.0, 1.0, 0.0);
    let x = Vector3D::new(-1.0, 1.0, 0.5);
    Hyperbola3D::new(Point3D::new(0.0, 0.0, 2.0), normal, x, major, minor).unwrap()
}

const CRITERIA: [(f64, Option<f64>); 4] = [
    (1e-1, None),
    (1e-3, None),
    (1e-2, Some(0.2)),
    (1e-4, Some(0.02)),
];

fn criteria_list() -> impl Iterator<Item = DiscretizationCriteria> {
    CRITERIA.into_iter().map(|(max, angle)| match angle {
        Some(angle) => chord_and_angle(max, angle),
        None => chord(max),
    })
}

#[test]
fn test_ellipses_whole_and_arcs() {
    for (major, minor) in [(2.0, 2.0), (3.0, 1.0), (10.0, 0.1)] {
        let e = ellipse(major, minor);
        for criteria in criteria_list() {
            let d = e.discretize(criteria);
            assert_proven(&e, &d, criteria, major);
            assert_eq!(d.points[0], d.points[d.points.len() - 1]);
            for (first, last) in [(0.3, 1.2), (2.0, 5.0), (-1.0, 0.5)] {
                let arc = TrimmedCurve3D::new(e, first, last).unwrap();
                assert_proven(&arc, &arc.discretize(criteria), criteria, major);
            }
        }
    }
}

#[test]
fn test_trimmed_parabolas_and_hyperbolas_including_far_along() {
    for (first, last) in [(-3.0, 3.0), (0.0, 1.0), (20.0, 40.0)] {
        let p = TrimmedCurve3D::new(parabola(0.5), first, last).unwrap();
        for criteria in criteria_list() {
            assert_proven(&p, &p.discretize(criteria), criteria, 400.0);
        }
    }
    for (major, minor) in [(1.0, 1.0), (3.0, 0.5), (0.5, 3.0)] {
        for (first, last) in [(-2.0, 2.0), (0.0, 0.5), (4.0, 6.0)] {
            let h = TrimmedCurve3D::new(hyperbola(major, minor), first, last).unwrap();
            for criteria in criteria_list() {
                assert_proven(&h, &h.discretize(criteria), criteria, 1000.0);
            }
        }
    }
}

#[test]
fn test_far_along_a_hyperbola_needs_few_points() {
    // Nearly straight beyond the vertex: the count follows the bend at the
    // vertex, not the parameter speed, which grows like e^u.
    let h = TrimmedCurve3D::new(hyperbola(1.0, 1.0), 0.0, 10.0).unwrap();
    assert!(
        h.point_count(chord(1e-3)) < 300,
        "{}",
        h.point_count(chord(1e-3))
    );
    let far = TrimmedCurve3D::new(hyperbola(1.0, 1.0), 8.0, 10.0).unwrap();
    assert!(
        far.point_count(chord(1e-3)) < 20,
        "{}",
        far.point_count(chord(1e-3))
    );
    let p = TrimmedCurve3D::new(parabola(1.0), 100.0, 120.0).unwrap();
    assert!(
        p.point_count(chord(1e-3)) < 100,
        "{}",
        p.point_count(chord(1e-3))
    );
}

#[test]
fn test_a_trimmed_line_is_one_segment() {
    let line = Line3D::new(Point3D::new(1.0, 2.0, 3.0), Vector3D::new(0.0, 3.0, 4.0)).unwrap();
    let segment = TrimmedCurve3D::new(line, -1.0, 2.0).unwrap();
    let d = segment.discretize(chord_and_angle(1e-9, 1e-9));
    assert_eq!(d.parameters, vec![-1.0, 2.0]);
    assert_eq!(d.points, vec![line.eval_point(-1.0), line.eval_point(2.0)]);
    assert_eq!(d.chord_deviations, vec![0.0]);
    assert_eq!(d.angle_deviations, vec![0.0]);
}

#[test]
fn test_run_time_curves_match_the_concrete_types() {
    fn same<C>(basis: C, first: f64, last: f64)
    where
        C: Clone + Into<Curve3D> + ParametricCurve3D,
        TrimmedCurve3D<C>: Discretize3D,
    {
        let concrete = TrimmedCurve3D::new(basis.clone(), first, last).unwrap();
        let run_time = TrimmedCurve3D::new(basis.into(), first, last).unwrap();
        for criteria in criteria_list() {
            assert_eq!(concrete.discretize(criteria), run_time.discretize(criteria));
        }
    }
    let line = Line3D::new(Point3D::ORIGIN, Vector3D::new(1.0, 1.0, 0.0)).unwrap();
    same(line, 0.0, 2.0);
    same(circle(2.0), 1.0, 4.0);
    same(ellipse(3.0, 1.0), -1.0, 2.0);
    same(parabola(0.5), -2.0, 3.0);
    same(hyperbola(2.0, 1.0), -1.0, 1.5);
    same(uneven_weights(), 0.2, 1.8);
}

#[test]
fn test_coarse_tolerances_keep_points_over_their_chords() {
    // Steps turning past a right angle can lie near the chord's line yet
    // far beyond the chord's ends.
    let flat = ellipse(10.0, 1.0);
    let narrow = parabola(0.05);
    for max in [0.5, 2.0, 5.0] {
        let criteria = chord(max);
        for (first, last) in [(0.0, 6.0), (-1.0, 2.5), (0.5, 3.0), (2.0, 4.0)] {
            let arc = TrimmedCurve3D::new(flat, first, last).unwrap();
            assert_proven(&arc, &arc.discretize(criteria), criteria, 10.0);
        }
        for (first, last) in [(-1.0, 3.0), (-0.2, 2.0), (-3.0, 0.5)] {
            let arc = TrimmedCurve3D::new(narrow, first, last).unwrap();
            assert_proven(&arc, &arc.discretize(criteria), criteria, 50.0);
        }
        let arc = TrimmedCurve3D::new(hyperbola(0.2, 3.0), -0.5, 3.0).unwrap();
        assert_proven(&arc, &arc.discretize(criteria), criteria, 100.0);
    }
}

// ---------------------------------------------------------------- 2D

fn distance_to_segment_2d(p: Point2D, a: Point2D, b: Point2D) -> f64 {
    let ab = b - a;
    let length2 = ab.dot(ab);
    let t = if length2 > 0.0 {
        ((p - a).dot(ab) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    p.distance(a + ab * t)
}

fn angle_to_segment_2d<C: ParametricCurve2D>(curve: &C, u: f64, a: Point2D, b: Point2D) -> f64 {
    let tangent = curve.eval_derivative(u, 1);
    let chord = b - a;
    tangent.cross(chord).abs().atan2(tangent.dot(chord))
}

/// [`assert_proven`] for a 2D curve.
fn assert_proven_2d<C: ParametricCurve2D>(
    curve: &C,
    d: &Discretization2D,
    criteria: DiscretizationCriteria,
    scale: f64,
) {
    let n = d.points.len();
    assert!(n >= 2);
    assert_eq!(d.parameters.len(), n);
    assert_eq!(d.chord_deviations.len(), n - 1);
    assert_eq!(d.angle_deviations.len(), n - 1);
    let (first, last) = curve.bounds();
    assert_eq!(d.parameters[0], first);
    assert_eq!(d.parameters[n - 1], last);
    assert!(d.parameters.windows(2).all(|w| w[0] < w[1]));
    for (u, p) in d.parameters.iter().zip(&d.points) {
        assert!(curve.eval_point(*u).distance(*p) <= 1e-12 * scale);
    }
    let max_angle = criteria.max_angle_deviation().unwrap_or(PI);
    for i in 0..n - 1 {
        let (u0, u1) = (d.parameters[i], d.parameters[i + 1]);
        let (a, b) = (d.points[i], d.points[i + 1]);
        let chord = maximum(u0, u1, |u| {
            distance_to_segment_2d(curve.eval_point(u), a, b)
        });
        let angle = maximum(u0, u1, |u| angle_to_segment_2d(curve, u, a, b));
        let (reported_chord, reported_angle) = (d.chord_deviations[i], d.angle_deviations[i]);
        assert!(
            reported_chord >= chord - 1e-12 * scale,
            "segment {i}: chord deviation {reported_chord} below the true {chord}"
        );
        assert!(reported_chord <= criteria.max_chord_deviation());
        assert!(
            reported_angle >= angle - 1e-9,
            "segment {i}: angle deviation {reported_angle} below the true {angle}"
        );
        assert!(reported_angle <= max_angle);
    }
}

/// A circle of radius 2 turning clockwise in the plane.
fn clockwise_circle_2d() -> Circle2D {
    let frame = Frame2D::new(
        Point2D::new(1.0, -3.0),
        Vector2D::new(1.0, 1.0),
        Vector2D::new(1.0, -1.0),
    );
    Circle2D::from_frame(frame.unwrap(), 2.0).unwrap()
}

#[test]
fn test_2d_circle_closed_forms() {
    let circle = Circle2D::new(Point2D::new(0.5, 0.25), 2.0).unwrap();
    // 2r sin²(θ/4) = 4 sin²(π/16) = 0.152...: eight segments meet 0.16.
    let d = circle.discretize(chord(0.16));
    assert_eq!(d.points.len(), 9);
    assert_eq!(d.points[0], d.points[8]);
    for (&chord_deviation, &angle) in d.chord_deviations.iter().zip(&d.angle_deviations) {
        assert!((chord_deviation - 4.0 * (PI / 16.0).sin().powi(2)).abs() < 1e-15);
        assert!((angle - PI / 8.0).abs() < 1e-15);
    }
    // At most π/10 between the tangent and a segment: ten segments.
    assert_eq!(circle.point_count(chord_and_angle(1.0, PI / 10.0)), 11);
    // A full circle has at least three segments, however coarse.
    assert_eq!(circle.point_count(chord(100.0)), 4);
    // So does an arc spanning a whole turn, ending on its first point.
    let turn = TrimmedCurve2D::new(circle, 1.0, 1.0 + TAU).unwrap();
    let d = turn.discretize(chord(100.0));
    assert_eq!(d.points.len(), 4);
    assert_eq!(d.points[0], d.points[3]);
}

#[test]
fn test_2d_circles_and_arcs_hold_their_deviations() {
    for circle in [
        Circle2D::new(Point2D::ORIGIN, 0.5).unwrap(),
        clockwise_circle_2d(),
    ] {
        for criteria in criteria_list() {
            assert_proven_2d(&circle, &circle.discretize(criteria), criteria, 2.0);
            for (first, last) in [(0.3, 1.2), (2.0, 7.0), (-1.0, 0.5)] {
                let arc = TrimmedCurve2D::new(circle, first, last).unwrap();
                assert_proven_2d(&arc, &arc.discretize(criteria), criteria, 2.0);
            }
        }
    }
}

#[test]
fn test_2d_trimmed_line_is_one_segment() {
    let line = Line2D::new(Point2D::new(1.0, 2.0), Vector2D::new(3.0, 4.0)).unwrap();
    let segment = TrimmedCurve2D::new(line, -1.0, 2.0).unwrap();
    let d = segment.discretize(chord_and_angle(1e-9, 1e-9));
    assert_eq!(d.parameters, vec![-1.0, 2.0]);
    assert_eq!(d.points, vec![line.eval_point(-1.0), line.eval_point(2.0)]);
    assert_eq!(d.chord_deviations, vec![0.0]);
    assert_eq!(d.angle_deviations, vec![0.0]);
}

#[test]
fn test_2d_borrowed_and_run_time_curves_match_concrete_ones() {
    fn same<C>(basis: C, first: f64, last: f64)
    where
        C: Clone + Into<Curve2D> + ParametricCurve2D,
        TrimmedCurve2D<C>: Discretize2D,
        for<'a> TrimmedCurve2D<&'a C>: Discretize2D,
    {
        let concrete = TrimmedCurve2D::new(basis.clone(), first, last).unwrap();
        let borrowed = TrimmedCurve2D::new(&basis, first, last).unwrap();
        let run_time = TrimmedCurve2D::new(basis.clone().into(), first, last).unwrap();
        for criteria in criteria_list() {
            let d = concrete.discretize(criteria);
            assert_eq!(borrowed.discretize(criteria), d);
            assert_eq!(run_time.discretize(criteria), d);
        }
    }
    let line = Line2D::new(Point2D::ORIGIN, Vector2D::new(1.0, 1.0)).unwrap();
    same(line, 0.0, 2.0);
    same(clockwise_circle_2d(), 1.0, 4.0);
    same(clockwise_circle_2d(), 1.0, 1.0 + TAU);
}
