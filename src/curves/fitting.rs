//! Fitting B-spline curves to points: interpolation and least-squares
//! approximation.
//!
//! Both build their linear system one row per data point from the
//! `degree + 1` basis functions non-zero there, found by a span search
//! resumed from the previous (increasing) parameter, and solve all three
//! coordinates in one pass over a band stored flat. Interpolation solves
//! its band directly; least squares solves the normal equations, whose
//! matrix has half-bandwidth `degree`, by banded Cholesky.

use crate::Point3D;
use crate::curve_math::bspline as math;
use crate::curves::{BSplineConstructionError, BSplineCurve3D};

/// How fitting assigns a parameter to each data point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointParametrization {
    /// Cumulative chord lengths (simplest; overshoots at sharp turns).
    Chordal,
    /// Square roots of chord lengths (tamer at sharp turns; default choice).
    Centripetal,
}

impl PointParametrization {
    fn exponent(self) -> f64 {
        match self {
            PointParametrization::Chordal => 1.0,
            PointParametrization::Centripetal => 0.5,
        }
    }
}

/// Increasing parameters of `points` in `[0, 1]` (`exponent` 1 chordal, 0.5
/// centripetal), or `ConfusedPoints` for coincident consecutive points.
fn point_parameters(
    points: &[Point3D],
    parametrization: PointParametrization,
) -> Result<Vec<f64>, BSplineConstructionError> {
    let exponent = parametrization.exponent();
    let mut params = Vec::with_capacity(points.len());
    params.push(0.0);
    let mut total = 0.0;
    for pair in points.windows(2) {
        let d = pair[1].distance(pair[0]);
        if d <= crate::tol::CONFUSION {
            return Err(BSplineConstructionError::ConfusedPoints);
        }
        total += if exponent == 1.0 { d } else { d.sqrt() };
        params.push(total);
    }
    let inverse = 1.0 / total;
    for t in &mut params[1..] {
        *t *= inverse;
    }
    // Exactly 1, whatever the rounding of the division: the curve's end.
    *params.last_mut().expect("at least two points") = 1.0;
    Ok(params)
}

/// The basis functions non-zero at increasing parameters, one row at a
/// time: the span search resumes where the previous parameter left it.
struct BasisRows<'a> {
    flat: &'a [f64],
    degree: usize,
    span: usize,
    left: [f64; math::LARGE],
    right: [f64; math::LARGE],
    values: [f64; math::LARGE],
}

impl<'a> BasisRows<'a> {
    fn new(flat: &'a [f64], degree: usize) -> Self {
        BasisRows {
            flat,
            degree,
            span: degree,
            left: [0.0; math::LARGE],
            right: [0.0; math::LARGE],
            values: [0.0; math::LARGE],
        }
    }

    /// The values at `t` (at least the previous parameter) of the basis
    /// functions of poles `first..=first + degree`; returns `first`.
    fn at(&mut self, t: f64) -> usize {
        let (flat, p) = (self.flat, self.degree);
        let last_span = flat.len() - p - 2;
        while self.span < last_span && flat[self.span + 1] <= t {
            self.span += 1;
        }
        math::basis_at(
            t,
            p,
            self.span,
            flat,
            &mut self.left,
            &mut self.right,
            &mut self.values,
        );
        self.span - p
    }
}

/// Clamped flat knots of `degree` over `[0, 1]` for `params.len()` poles,
/// interior knots averaging `degree` consecutive parameters.
fn averaged_knots(params: &[f64], degree: usize) -> Vec<f64> {
    let (n, p) = (params.len(), degree);
    let mut flat = Vec::with_capacity(n + p + 1);
    flat.resize(p + 1, 0.0);
    // Running sum of params[j..j + p].
    let mut sum: f64 = params[1..p].iter().sum();
    for j in 1..n - p {
        sum += params[j + p - 1];
        flat.push(sum / p as f64);
        sum -= params[j];
    }
    flat.resize(n + p + 1, 1.0);
    flat
}

/// Clamped flat knots of `degree` over `[0, 1]` for `pole_count` poles,
/// interior knots evenly spaced.
fn uniform_knots(pole_count: usize, degree: usize) -> Vec<f64> {
    let (m, p) = (pole_count, degree);
    let mut flat = Vec::with_capacity(m + p + 1);
    flat.resize(p + 1, 0.0);
    flat.extend((1..m - p).map(|j| j as f64 / (m - p) as f64));
    flat.resize(m + p + 1, 1.0);
    flat
}

/// Distinct values and multiplicities of flat knots.
fn compress_knots(flat: &[f64]) -> (Vec<f64>, Vec<u32>) {
    let distinct = 1 + flat.windows(2).filter(|w| w[1] != w[0]).count();
    let mut knots = Vec::with_capacity(distinct);
    let mut mults: Vec<u32> = Vec::with_capacity(distinct);
    for &u in flat {
        if knots.last() == Some(&u) {
            *mults.last_mut().expect("parallel to knots") += 1;
        } else {
            knots.push(u);
            mults.push(1);
        }
    }
    (knots, mults)
}

fn check_degree(degree: usize) -> Result<(), BSplineConstructionError> {
    if (1..=math::MAX_DEGREE).contains(&degree) {
        Ok(())
    } else {
        Err(BSplineConstructionError::InvalidDegree)
    }
}

/// Solves `A x = b` in place for a band matrix stored row by row,
/// `band[i * width + (j + bandwidth - i)]` holding `A[i][j]` for
/// `|i - j| <= bandwidth` (`width = 2 * bandwidth + 1`), by LU without
/// pivoting; `rhs` holds `b` and receives `x`.
fn solve_band(band: &mut [f64], bandwidth: usize, rhs: &mut [Point3D]) {
    let (n, h) = (rhs.len(), bandwidth);
    let w = 2 * h + 1;
    for k in 0..n {
        let end = (k + h + 1).min(n);
        let (done, rest) = band.split_at_mut((k + 1) * w);
        // Row k from its diagonal on: columns k..end.
        let pivot_row = &done[k * w + h..k * w + h + (end - k)];
        let inverse = 1.0 / pivot_row[0];
        let bk = rhs[k];
        for (d, row) in rest.chunks_exact_mut(w).take(end - k - 1).enumerate() {
            // Row i = k + 1 + d holds column k at offset h - d - 1.
            let start = h - d - 1;
            let factor = row[start] * inverse;
            if factor == 0.0 {
                continue;
            }
            for (a, &b) in row[start + 1..start + end - k]
                .iter_mut()
                .zip(&pivot_row[1..])
            {
                *a -= factor * b;
            }
            let bi = &mut rhs[k + 1 + d];
            bi.x -= factor * bk.x;
            bi.y -= factor * bk.y;
            bi.z -= factor * bk.z;
        }
    }
    for i in (0..n).rev() {
        let row = &band[i * w + h..i * w + h + (n - i).min(h + 1)];
        let mut x = rhs[i];
        for (&a, r) in row[1..].iter().zip(&rhs[i + 1..]) {
            x.x -= a * r.x;
            x.y -= a * r.y;
            x.z -= a * r.z;
        }
        let inverse = 1.0 / row[0];
        rhs[i] = Point3D::new(x.x * inverse, x.y * inverse, x.z * inverse);
    }
}

/// Solves `A x = b` in place for a symmetric positive definite band matrix
/// whose lower band is stored row by row, `lower[i * (h + 1) + (j + h - i)]`
/// holding `A[i][j]` for `i - h <= j <= i`, by Cholesky (`A = L L'`, `L`
/// overwriting `lower`). `rhs` holds `b` and receives `x`. Returns false
/// when a pivot falls to `relative_floor` times the largest diagonal or
/// below: a pole the data leaves (nearly) undetermined.
fn solve_symmetric_band(
    lower: &mut [f64],
    h: usize,
    rhs: &mut [Point3D],
    relative_floor: f64,
) -> bool {
    let n = rhs.len();
    let w = h + 1;
    let at = |i: usize, j: usize| i * w + j + h - i;
    let largest = (0..n).map(|i| lower[at(i, i)]).fold(0.0f64, f64::max);
    let floor = relative_floor * largest;
    for i in 0..n {
        let start = i.saturating_sub(h);
        for j in start..=i {
            let mut sum = lower[at(i, j)];
            for k in start.max(j.saturating_sub(h))..j {
                sum -= lower[at(i, k)] * lower[at(j, k)];
            }
            if j == i {
                // NaN too: a pole the data cannot determine.
                if sum.is_nan() || sum <= floor {
                    return false;
                }
                lower[at(i, i)] = sum.sqrt();
            } else {
                lower[at(i, j)] = sum / lower[at(j, j)];
            }
        }
    }
    // L y = b, then L' x = y.
    for i in 0..n {
        let mut y = rhs[i];
        for k in i.saturating_sub(h)..i {
            let l = lower[at(i, k)];
            y.x -= l * rhs[k].x;
            y.y -= l * rhs[k].y;
            y.z -= l * rhs[k].z;
        }
        let inverse = 1.0 / lower[at(i, i)];
        rhs[i] = Point3D::new(y.x * inverse, y.y * inverse, y.z * inverse);
    }
    for i in (0..n).rev() {
        let mut x = rhs[i];
        for k in i + 1..(i + h + 1).min(n) {
            let l = lower[at(k, i)];
            x.x -= l * rhs[k].x;
            x.y -= l * rhs[k].y;
            x.z -= l * rhs[k].z;
        }
        let inverse = 1.0 / lower[at(i, i)];
        rhs[i] = Point3D::new(x.x * inverse, x.y * inverse, x.z * inverse);
    }
    true
}

impl BSplineCurve3D {
    /// Interpolates a clamped non-rational B-spline through `points`.
    ///
    /// Parameters come from chord lengths (`Chordal`) or their square
    /// roots (`Centripetal`, better behaved at sharp turns), normalized
    /// to `[0, 1]`; interior knots are parameter averages, and the
    /// poles solve the banded interpolation system by LU without
    /// pivoting (safe: averaged knots satisfy Schoenberg-Whitney, so the
    /// matrix is totally positive). End poles equal the end points.
    ///
    /// # Errors
    ///
    /// Returns [`BSplineConstructionError::TooFewPoints`] for fewer than
    /// two points, [`BSplineConstructionError::InvalidDegree`] outside
    /// `1..=25`, [`BSplineConstructionError::DegreeTooHigh`] when
    /// `degree` exceeds points minus one, or
    /// [`BSplineConstructionError::ConfusedPoints`] for coincident
    /// consecutive points.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, PointParametrization, Tolerance};
    /// let points = vec![
    ///     Point3D::new(0.0, 0.0, 0.0),
    ///     Point3D::new(1.0, 1.0, 0.0),
    ///     Point3D::new(2.0, 0.0, 0.0),
    /// ];
    /// let curve =
    ///     BSplineCurve3D::interpolate(&points, 2, PointParametrization::Centripetal).unwrap();
    /// let tol = Tolerance::DEFAULT;
    /// for p in &points {
    ///     assert!(curve.contains(*p, tol));
    /// }
    /// ```
    pub fn interpolate(
        points: &[Point3D],
        degree: usize,
        parametrization: PointParametrization,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        let n = points.len();
        if n < 2 {
            return Err(BSplineConstructionError::TooFewPoints);
        }
        check_degree(degree)?;
        if degree > n - 1 {
            return Err(BSplineConstructionError::DegreeTooHigh);
        }
        let params = point_parameters(points, parametrization)?;
        let p = degree;
        let flat = averaged_knots(&params, p);
        // Row j holds the basis functions at params[j], over poles
        // `first..=first + p`. With averaged knots `A[j][i] = 0` whenever
        // `|i - j| >= p` (each such basis function's support ends at or
        // starts after params[j]), so the band is `p - 1` wide each side.
        let h = p - 1;
        let width = 2 * h + 1;
        let mut band = vec![0.0; n * width];
        let mut rows = BasisRows::new(&flat, p);
        for (j, &t) in params.iter().enumerate() {
            let first = rows.at(t);
            let row = &mut band[j * width..(j + 1) * width];
            for (a, &value) in rows.values[..=p].iter().enumerate() {
                let column = first + a;
                if column + h >= j && column <= j + h {
                    row[column + h - j] = value;
                } else {
                    debug_assert_eq!(value, 0.0, "row {j}, column {column}");
                }
            }
        }
        let mut poles = points.to_vec();
        solve_band(&mut band, h, &mut poles);
        // The end rows are unit rows: the end poles are the end points exactly.
        poles[0] = points[0];
        poles[n - 1] = points[n - 1];
        let (knots, mults) = compress_knots(&flat);
        Ok(BSplineCurve3D::from_validated_parts(
            degree, poles, None, knots, mults, flat, false,
        ))
    }

    /// Least-squares B-spline fit through `points` with `num_poles` poles.
    ///
    /// Same chord-like parametrization as
    /// [`BSplineCurve3D::interpolate`], but uniform interior knots and an
    /// overdetermined system: the normal equations `A'A x = A'b`, whose
    /// matrix has half-bandwidth `degree`, solve by banded Cholesky. The
    /// fit approximates rather than hits the data; poles the data leaves
    /// undetermined fail explicitly instead of producing NaNs.
    ///
    /// # Errors
    ///
    /// [`BSplineConstructionError::TooFewPoints`] for fewer than two
    /// points, [`BSplineConstructionError::InvalidDegree`] outside
    /// `1..=25`, [`BSplineConstructionError::DegreeTooHigh`] when
    /// `degree` exceeds poles minus one,
    /// [`BSplineConstructionError::TooManyPoles`] for more poles than
    /// points, [`BSplineConstructionError::ConfusedPoints`] for
    /// coincident consecutive points, or
    /// [`BSplineConstructionError::PoleCountMismatch`] when the data
    /// leaves a pole undetermined.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, PointParametrization};
    /// let points: Vec<Point3D> = (0..9)
    ///     .map(|i| {
    ///         let u = i as f64 / 8.0 * std::f64::consts::TAU;
    ///         Point3D::new(2.0 * u.cos(), 2.0 * u.sin(), 0.0)
    ///     })
    ///     .collect();
    /// let curve =
    ///     BSplineCurve3D::approximate(&points, 3, 6, PointParametrization::Centripetal)
    ///         .unwrap();
    /// assert_eq!(curve.degree(), 3);
    /// // Near-circular fit: radius stays within a few percent of 2.
    /// for i in 0..=40 {
    ///     let p = curve.eval_point(i as f64 / 40.0);
    ///     let r = (p.x * p.x + p.y * p.y).sqrt();
    ///     assert!((r - 2.0).abs() < 0.1, "{p:?}");
    /// }
    /// ```
    pub fn approximate(
        points: &[Point3D],
        degree: usize,
        num_poles: usize,
        parametrization: PointParametrization,
    ) -> Result<BSplineCurve3D, BSplineConstructionError> {
        let n = points.len();
        if n < 2 {
            return Err(BSplineConstructionError::TooFewPoints);
        }
        check_degree(degree)?;
        if degree > num_poles.saturating_sub(1) {
            return Err(BSplineConstructionError::DegreeTooHigh);
        }
        if num_poles > n {
            return Err(BSplineConstructionError::TooManyPoles);
        }
        let params = point_parameters(points, parametrization)?;
        let (p, m) = (degree, num_poles);
        let flat = uniform_knots(m, p);
        let width = p + 1;
        let mut lower = vec![0.0; m * width];
        let mut poles = vec![Point3D::ORIGIN; m];
        let mut rows = BasisRows::new(&flat, p);
        for (point, &t) in points.iter().zip(&params) {
            let first = rows.at(t);
            let values = &rows.values[..=p];
            for (a, &va) in values.iter().enumerate() {
                let i = first + a;
                let row = &mut lower[i * width..(i + 1) * width];
                // Columns first..=i sit at offsets first + p - i..=p.
                for (b, &vb) in values[..=a].iter().enumerate() {
                    row[b + p - a] += va * vb;
                }
                let pole = &mut poles[i];
                pole.x += va * point.x;
                pole.y += va * point.y;
                pole.z += va * point.z;
            }
        }
        if !solve_symmetric_band(&mut lower, p, &mut poles, 1e-12)
            || poles.iter().any(|q| !q.is_finite())
        {
            return Err(BSplineConstructionError::PoleCountMismatch);
        }
        let (knots, mults) = compress_knots(&flat);
        Ok(BSplineCurve3D::from_validated_parts(
            degree, poles, None, knots, mults, flat, false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense Gaussian elimination with partial pivoting, the reference.
    fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<Point3D>) -> Vec<Point3D> {
        let n = b.len();
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
                .unwrap();
            a.swap(k, pivot);
            b.swap(k, pivot);
            for i in k + 1..n {
                let f = a[i][k] / a[k][k];
                let pivot_row = a[k].clone();
                for (x, y) in a[i][k..].iter_mut().zip(&pivot_row[k..]) {
                    *x -= f * y;
                }
                let bk = b[k];
                b[i] = Point3D::new(b[i].x - f * bk.x, b[i].y - f * bk.y, b[i].z - f * bk.z);
            }
        }
        for i in (0..n).rev() {
            let mut x = b[i];
            for j in i + 1..n {
                x = Point3D::new(
                    x.x - a[i][j] * b[j].x,
                    x.y - a[i][j] * b[j].y,
                    x.z - a[i][j] * b[j].z,
                );
            }
            b[i] = Point3D::new(x.x / a[i][i], x.y / a[i][i], x.z / a[i][i]);
        }
        b
    }

    /// The full basis matrix row at `t`, by the Cox-de Boor recurrence.
    fn dense_row(flat: &[f64], p: usize, t: f64) -> Vec<f64> {
        let n = flat.len() - p - 1;
        let mut rows = BasisRows::new(flat, p);
        let first = rows.at(t);
        let mut row = vec![0.0; n];
        row[first..=first + p].copy_from_slice(&rows.values[..=p]);
        row
    }

    fn wavy_points(count: usize) -> Vec<Point3D> {
        (0..count)
            .map(|i| {
                let s = i as f64 / (count - 1) as f64;
                Point3D::new(
                    3.0 * s,
                    (7.0 * s).sin() + 0.3 * s * s,
                    (3.0 * s).cos() * 0.5,
                )
            })
            .collect()
    }

    fn assert_points_close(a: &[Point3D], b: &[Point3D], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!(x.distance(*y) <= tol, "{x:?} vs {y:?}");
        }
    }

    #[test]
    fn test_basis_rows_sum_to_one_and_match_full_rows() {
        let points = wavy_points(17);
        let params = point_parameters(&points, PointParametrization::Chordal).unwrap();
        for p in 1..=5 {
            let flat = averaged_knots(&params, p);
            let mut rows = BasisRows::new(&flat, p);
            for &t in &params {
                rows.at(t);
                let sum: f64 = rows.values[..=p].iter().sum();
                assert!((sum - 1.0).abs() < 1e-14, "degree {p} at {t}: {sum}");
            }
        }
    }

    #[test]
    fn test_averaged_knots_match_definition() {
        let params = [0.0, 0.1, 0.25, 0.3, 0.6, 0.8, 1.0];
        let p = 3;
        let flat = averaged_knots(&params, p);
        assert_eq!(flat.len(), params.len() + p + 1);
        for j in 1..params.len() - p {
            let mean = params[j..j + p].iter().sum::<f64>() / p as f64;
            assert!((flat[j + p] - mean).abs() < 1e-15);
        }
        assert_eq!(&flat[..=p], &[0.0; 4]);
        assert_eq!(&flat[flat.len() - p - 1..], &[1.0; 4]);
    }

    #[test]
    fn test_interpolation_matches_dense_solve() {
        for (count, p) in [(2, 1), (5, 2), (9, 3), (40, 3), (23, 5)] {
            let points = wavy_points(count);
            let curve =
                BSplineCurve3D::interpolate(&points, p, PointParametrization::Chordal).unwrap();
            let params = point_parameters(&points, PointParametrization::Chordal).unwrap();
            let flat = averaged_knots(&params, p);
            let a = params.iter().map(|&t| dense_row(&flat, p, t)).collect();
            let expected = dense_solve(a, points.clone());
            assert_points_close(curve.poles(), &expected, 1e-11);
            for (q, &t) in points.iter().zip(&params) {
                assert!(curve.eval_point(t).distance(*q) < 1e-11);
            }
        }
    }

    #[test]
    fn test_approximation_matches_dense_normal_equations() {
        for (count, p, m) in [(4, 1, 2), (12, 3, 7), (60, 3, 20), (31, 4, 9), (8, 2, 8)] {
            let points = wavy_points(count);
            let curve =
                BSplineCurve3D::approximate(&points, p, m, PointParametrization::Centripetal)
                    .unwrap();
            let params = point_parameters(&points, PointParametrization::Centripetal).unwrap();
            let flat = uniform_knots(m, p);
            let rows: Vec<Vec<f64>> = params.iter().map(|&t| dense_row(&flat, p, t)).collect();
            let mut normal = vec![vec![0.0; m]; m];
            let mut rhs = vec![Point3D::ORIGIN; m];
            for (row, q) in rows.iter().zip(&points) {
                for i in 0..m {
                    for j in 0..m {
                        normal[i][j] += row[i] * row[j];
                    }
                    rhs[i] = Point3D::new(
                        rhs[i].x + row[i] * q.x,
                        rhs[i].y + row[i] * q.y,
                        rhs[i].z + row[i] * q.z,
                    );
                }
            }
            assert_points_close(curve.poles(), &dense_solve(normal, rhs), 1e-10);
        }
    }

    #[test]
    fn test_approximation_rejects_poles_without_data() {
        // Two clusters of points leave the middle poles of a 12-pole fit
        // without data support.
        let mut points: Vec<Point3D> = (0..6)
            .map(|i| Point3D::new(i as f64 * 1e-3, 0.0, 0.0))
            .collect();
        points.extend((0..6).map(|i| Point3D::new(10.0 + i as f64 * 1e-3, 1.0, 0.0)));
        assert_eq!(
            BSplineCurve3D::approximate(&points, 3, 12, PointParametrization::Chordal),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
    }

    #[test]
    fn test_interpolation_of_two_points_is_their_segment() {
        let points = [Point3D::new(1.0, 2.0, 3.0), Point3D::new(4.0, 6.0, 3.0)];
        let curve = BSplineCurve3D::interpolate(&points, 1, PointParametrization::Chordal).unwrap();
        assert_eq!(curve.poles(), &points);
        assert_eq!(curve.knots(), &[0.0, 1.0]);
        assert_eq!(curve.multiplicities(), &[2, 2]);
    }
}
