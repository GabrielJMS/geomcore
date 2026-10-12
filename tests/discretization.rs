//! Discretization through its public interface: every reported deviation
//! is checked against an independent measurement of the true one.

use geomcore::curves::ParametricCurve3D;
use geomcore::discretization::{
    Discretization3D, DiscretizationCriteria, DiscretizationCriteriaError, Discretize3D,
};
use geomcore::{BSplineCurve3D, Circle3D, Point3D, TrimmedCurve3D, Vector3D};
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
