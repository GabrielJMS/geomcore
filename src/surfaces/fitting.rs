//! Fitting B-spline surfaces to grids of points: least-squares
//! approximation, with a given number of poles or within a deviation.
//!
//! On a full grid the tensor-product least-squares fit separates: each
//! grid row is fitted along `v`, then each column of those fits' poles
//! along `u`, every row (column) on the same parameters and knots, so one
//! factored system per direction serves them all (see
//! [`crate::BSplineCurve3D::approximate`] for the curve fit).

use crate::curves::BSplineConstructionError;
use crate::curves::fitting::{
    BasisTable, approximation_knots, averaged_knots, check_degree, fit_within_fewest,
    interpolation, least_squares, shared_parameters,
};
use crate::{BSplineSurface, Point3D, PointParametrization};

/// A grid of points `points[i][j]`, `i` along `u` and `j` along `v`,
/// checked rectangular, with its parameters in both directions.
struct Grid<'a> {
    points: &'a [Vec<Point3D>],
    rows: usize,
    cols: usize,
    u_params: Vec<f64>,
    v_params: Vec<f64>,
}

impl<'a> Grid<'a> {
    fn new(
        points: &'a [Vec<Point3D>],
        parametrization: PointParametrization,
    ) -> Result<Self, BSplineConstructionError> {
        let rows = points.len();
        let cols = points.first().map_or(0, Vec::len);
        if rows < 2 || cols < 2 {
            return Err(BSplineConstructionError::TooFewPoints);
        }
        if points.iter().any(|row| row.len() != cols) {
            return Err(BSplineConstructionError::PoleCountMismatch);
        }
        // Along `u`, the columns are the lines of points; along `v`, the rows.
        let u_params = shared_parameters(cols, rows, |j, i| points[i][j], parametrization)?;
        let v_params = shared_parameters(rows, cols, |i, j| points[i][j], parametrization)?;
        Ok(Grid {
            points,
            rows,
            cols,
            u_params,
            v_params,
        })
    }

    /// The least-squares poles along `v`, `count` per grid row, on `basis`
    /// (`rows x count`, row-major).
    fn fit_rows(&self, basis: &BasisTable, count: usize) -> Option<Vec<Point3D>> {
        least_squares(basis, count, self.rows, |i, j| self.points[i][j])
    }

    /// The least-squares poles along `u`, `count` per column of
    /// `row_poles` (`rows x columns`, row-major), on `basis`, as the
    /// surface's grid (`count x columns`, row-major).
    fn fit_columns(
        &self,
        basis: &BasisTable,
        count: usize,
        row_poles: &[Point3D],
        columns: usize,
    ) -> Option<Vec<Point3D>> {
        let by_column = least_squares(basis, count, columns, |l, i| row_poles[i * columns + l])?;
        Some(transposed(&by_column, columns, count))
    }

    /// The deviation of each row of the grid along `v`: of grid point
    /// `(i, j)` from the point at `v` parameter `j` of the curve with
    /// `basis` and the poles of row `i` in `row_poles` (`count` per row),
    /// the most over the rows, into `deviations[j]`.
    fn row_deviations(
        &self,
        basis: &BasisTable,
        row_poles: &[Point3D],
        count: usize,
        deviations: &mut [f64],
    ) {
        deviations.fill(0.0);
        for (row, line) in row_poles.chunks_exact(count).zip(self.points) {
            for (j, deviation) in deviations.iter_mut().enumerate() {
                *deviation = deviation.max(basis.point(j, row).distance(line[j]));
            }
        }
    }

    /// The deviation of each grid point `(i, j)` from the point at its
    /// parameters of the surface with the `u` and `v` bases
    /// `(u_basis, v_basis)` and the poles `poles` (row-major, `columns`
    /// per row), the most over `j`, into `deviations[i]`.
    fn deviations(
        &self,
        (u_basis, v_basis): (&BasisTable, &BasisTable),
        poles: &[Point3D],
        columns: usize,
        deviations: &mut [f64],
    ) {
        // The poles of the `v` curve at each `u` parameter, then its points.
        let mut curve = vec![Point3D::ORIGIN; columns];
        for (i, deviation) in deviations.iter_mut().enumerate() {
            let (first, values) = u_basis.at(i);
            for (l, pole) in curve.iter_mut().enumerate() {
                let mut x = [0.0; 3];
                for (a, &value) in values.iter().enumerate() {
                    let q = poles[(first + a) * columns + l];
                    x[0] += value * q.x;
                    x[1] += value * q.y;
                    x[2] += value * q.z;
                }
                *pole = Point3D::new(x[0], x[1], x[2]);
            }
            *deviation = (0..self.cols)
                .map(|j| v_basis.point(j, &curve).distance(self.points[i][j]))
                .fold(0.0, f64::max);
        }
    }
}

/// `values` (`lines` lines of `count`) as `count` lines of `lines`.
fn transposed(values: &[Point3D], lines: usize, count: usize) -> Vec<Point3D> {
    let mut out = Vec::with_capacity(values.len());
    out.extend((0..count).flat_map(|k| (0..lines).map(move |l| values[l * count + k])));
    out
}

impl BSplineSurface {
    /// Least-squares B-spline surface fit through the grid of points
    /// `points[i][j]` (`i` along `u`, `j` along `v`), with
    /// `u_pole_count x v_pole_count` poles.
    ///
    /// Each direction's parameters average those of its lines of points
    /// (the columns for `u`, the rows for `v`), as
    /// [`crate::BSplineCurve3D::approximate`] gives a curve's, over the
    /// lines that have a length, and its knots follow those parameters as
    /// the curve fit's do. The fit
    /// approximates rather than hits the points.
    ///
    /// # Errors
    ///
    /// [`BSplineConstructionError::TooFewPoints`] for fewer than two rows
    /// or columns, [`BSplineConstructionError::PoleCountMismatch`] for
    /// rows of differing lengths or when the points leave a pole
    /// undetermined, [`BSplineConstructionError::InvalidDegree`] outside
    /// `1..=25`, [`BSplineConstructionError::DegreeTooHigh`] when a
    /// degree exceeds its pole count minus one,
    /// [`BSplineConstructionError::TooManyPoles`] for more poles than
    /// points in a direction, or
    /// [`BSplineConstructionError::ConfusedPoints`] when a direction's
    /// parameters coincide.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, PointParametrization};
    /// let points: Vec<Vec<Point3D>> = (0..10)
    ///     .map(|i| {
    ///         (0..10)
    ///             .map(|j| {
    ///                 let (x, y) = (i as f64 / 9.0, j as f64 / 9.0);
    ///                 Point3D::new(x, y, x * x - y * y)
    ///             })
    ///             .collect()
    ///     })
    ///     .collect();
    /// let surface =
    ///     BSplineSurface::approximate(&points, 2, 2, 3, 3, PointParametrization::Chordal)
    ///         .unwrap();
    /// // A quadratic Bézier patch holds `x² - y²` exactly.
    /// let p = surface.eval_point(0.5, 0.5);
    /// assert!((p.z - (p.x * p.x - p.y * p.y)).abs() < 1e-12);
    /// ```
    pub fn approximate(
        points: &[Vec<Point3D>],
        u_degree: usize,
        v_degree: usize,
        u_pole_count: usize,
        v_pole_count: usize,
        parametrization: PointParametrization,
    ) -> Result<BSplineSurface, BSplineConstructionError> {
        check_degree(u_degree)?;
        check_degree(v_degree)?;
        let grid = Grid::new(points, parametrization)?;
        for (degree, count, available) in [
            (u_degree, u_pole_count, grid.rows),
            (v_degree, v_pole_count, grid.cols),
        ] {
            if degree > count.saturating_sub(1) {
                return Err(BSplineConstructionError::DegreeTooHigh);
            }
            if count > available {
                return Err(BSplineConstructionError::TooManyPoles);
            }
        }
        let v_flat = approximation_knots(&grid.v_params, v_pole_count, v_degree);
        let v_basis = BasisTable::new(&v_flat, v_degree, &grid.v_params);
        let u_flat = approximation_knots(&grid.u_params, u_pole_count, u_degree);
        let u_basis = BasisTable::new(&u_flat, u_degree, &grid.u_params);
        let poles = grid
            .fit_rows(&v_basis, v_pole_count)
            .and_then(|rows| grid.fit_columns(&u_basis, u_pole_count, &rows, v_pole_count))
            .ok_or(BSplineConstructionError::PoleCountMismatch)?;
        Ok(BSplineSurface::from_flat_knots(
            (u_degree, u_flat),
            (v_degree, v_flat),
            poles,
        ))
    }

    /// Least-squares B-spline surface fit through the grid of points
    /// `points[i][j]` (as [`BSplineSurface::approximate`]), with as few
    /// poles as keep every point within `max_deviation` of the surface.
    ///
    /// The deviation is measured from each point to the surface at the
    /// point's parameters (which bounds the distance to the surface).
    /// The `v` knots come first, until every row fits within 0.8 of
    /// `max_deviation`; then the `u` knots, until the whole grid fits
    /// within `max_deviation`. Each direction keeps the fewer poles of
    /// knots refined as [`crate::BSplineCurve3D::approximate_within`]
    /// refines a curve's and of the fewest poles that fit on knots spread
    /// along the parameters, as [`BSplineSurface::approximate`] places
    /// them. A direction no least-squares fit brings close enough
    /// interpolates its lines of points instead, one pole per point, as
    /// with a `max_deviation` of zero.
    ///
    /// # Errors
    ///
    /// As [`BSplineSurface::approximate`], a degree exceeding its
    /// direction's points minus one being
    /// [`BSplineConstructionError::DegreeTooHigh`].
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineSurface, Point3D, PointParametrization};
    /// let points: Vec<Vec<Point3D>> = (0..20)
    ///     .map(|i| {
    ///         (0..20)
    ///             .map(|j| {
    ///                 let (x, y) = (i as f64 / 19.0, j as f64 / 19.0);
    ///                 Point3D::new(x, y, (3.0 * x).sin() * (2.0 * y).cos())
    ///             })
    ///             .collect()
    ///     })
    ///     .collect();
    /// let surface = BSplineSurface::approximate_within(
    ///     &points, 3, 3, 1e-3, PointParametrization::Chordal,
    /// )
    /// .unwrap();
    /// assert!(surface.u_pole_count() < 20 && surface.v_pole_count() < 20);
    /// ```
    pub fn approximate_within(
        points: &[Vec<Point3D>],
        u_degree: usize,
        v_degree: usize,
        max_deviation: f64,
        parametrization: PointParametrization,
    ) -> Result<BSplineSurface, BSplineConstructionError> {
        check_degree(u_degree)?;
        check_degree(v_degree)?;
        let grid = Grid::new(points, parametrization)?;
        if u_degree > grid.rows - 1 || v_degree > grid.cols - 1 {
            return Err(BSplineConstructionError::DegreeTooHigh);
        }
        // Along `v`: every row within most of the deviation. Fewer `v`
        // poles cost few along `u`, whose fit is checked on the whole grid.
        let (q, v_params) = (v_degree, &grid.v_params);
        let fitted_rows = fit_within_fewest(
            v_params,
            q,
            0.8 * max_deviation,
            |flat, basis, deviations| {
                let count = flat.len() - q - 1;
                let poles = grid.fit_rows(basis, count)?;
                grid.row_deviations(basis, &poles, count, deviations);
                Some(poles)
            },
        );
        let (v_flat, row_poles) = fitted_rows.unwrap_or_else(|| {
            let flat = averaged_knots(v_params, q);
            let basis = BasisTable::new(&flat, q, v_params);
            (flat, interpolation(&basis, grid.rows, |i, j| points[i][j]))
        });
        let v_basis = BasisTable::new(&v_flat, q, v_params);
        let columns = row_poles.len() / grid.rows;
        // Along `u`: the whole grid within the deviation. Interpolating
        // the rows' poles leaves only the rows' deviation, within it.
        let (p, u_params) = (u_degree, &grid.u_params);
        let fitted = fit_within_fewest(u_params, p, max_deviation, |flat, basis, deviations| {
            let poles = grid.fit_columns(basis, flat.len() - p - 1, &row_poles, columns)?;
            grid.deviations((basis, &v_basis), &poles, columns, deviations);
            Some(poles)
        });
        let (u_flat, poles) = fitted.unwrap_or_else(|| {
            let flat = averaged_knots(u_params, p);
            let basis = BasisTable::new(&flat, p, u_params);
            let by_column = interpolation(&basis, columns, |l, i| row_poles[i * columns + l]);
            (flat, transposed(&by_column, columns, grid.rows))
        });
        Ok(BSplineSurface::from_flat_knots(
            (u_degree, u_flat),
            (v_degree, v_flat),
            poles,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wavy_grid(rows: usize, cols: usize) -> Vec<Vec<Point3D>> {
        (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| {
                        let (x, y) = (i as f64 / (rows - 1) as f64, j as f64 / (cols - 1) as f64);
                        // Uneven spacing along both directions.
                        let (x, y) = (x + 0.1 * x * x, y - 0.05 * y * y);
                        Point3D::new(x, y, (3.0 * x).sin() * (2.0 * y).cos() + 0.2 * x * y)
                    })
                    .collect()
            })
            .collect()
    }

    /// Dense Gaussian elimination with partial pivoting.
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

    /// The full basis row of `basis` at parameter `k`, `count` poles.
    fn dense_row(basis: &BasisTable, k: usize, count: usize) -> Vec<f64> {
        let (first, values) = basis.at(k);
        let mut row = vec![0.0; count];
        row[first..first + values.len()].copy_from_slice(values);
        row
    }

    #[test]
    fn test_approximation_matches_dense_tensor_least_squares() {
        for (rows, cols, p, q, mu, mv) in
            [(5, 6, 2, 1, 3, 4), (9, 7, 3, 3, 6, 5), (12, 12, 3, 2, 6, 6)]
        {
            let points = wavy_grid(rows, cols);
            let par = PointParametrization::Chordal;
            let surface = BSplineSurface::approximate(&points, p, q, mu, mv, par).unwrap();
            let grid = Grid::new(&points, par).unwrap();
            let u_basis = BasisTable::new(
                &approximation_knots(&grid.u_params, mu, p),
                p,
                &grid.u_params,
            );
            let v_basis = BasisTable::new(
                &approximation_knots(&grid.v_params, mv, q),
                q,
                &grid.v_params,
            );
            // One equation per grid point over the `mu * mv` poles.
            let mut normal = vec![vec![0.0; mu * mv]; mu * mv];
            let mut rhs = vec![Point3D::ORIGIN; mu * mv];
            for (i, line) in points.iter().enumerate() {
                let nu = dense_row(&u_basis, i, mu);
                for (j, &x) in line.iter().enumerate() {
                    let nv = dense_row(&v_basis, j, mv);
                    let row: Vec<f64> = nu
                        .iter()
                        .flat_map(|a| nv.iter().map(move |b| a * b))
                        .collect();
                    for (r, &ar) in row.iter().enumerate() {
                        for (c, &ac) in row.iter().enumerate() {
                            normal[r][c] += ar * ac;
                        }
                        rhs[r] = Point3D::new(
                            rhs[r].x + ar * x.x,
                            rhs[r].y + ar * x.y,
                            rhs[r].z + ar * x.z,
                        );
                    }
                }
            }
            let expected = dense_solve(normal, rhs);
            assert_eq!((surface.u_pole_count(), surface.v_pole_count()), (mu, mv));
            for (a, b) in surface.poles().iter().zip(&expected) {
                assert!(a.distance(*b) < 1e-10, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn test_approximation_within_keeps_grid_points_close() {
        let points = wavy_grid(25, 18);
        let par = PointParametrization::Centripetal;
        let grid = Grid::new(&points, par).unwrap();
        for (p, q) in [(1, 1), (2, 3), (3, 3)] {
            for tolerance in [1e-1, 1e-3, 1e-5, 0.0] {
                let surface =
                    BSplineSurface::approximate_within(&points, p, q, tolerance, par).unwrap();
                for (i, row) in points.iter().enumerate() {
                    for (j, x) in row.iter().enumerate() {
                        let at = surface.eval_point(grid.u_params[i], grid.v_params[j]);
                        assert!(
                            at.distance(*x) <= tolerance.max(1e-12),
                            "({p}, {q}) {tolerance}"
                        );
                    }
                }
                if tolerance == 1e-1 {
                    assert!(surface.u_pole_count() < 25 && surface.v_pole_count() < 18);
                }
                if tolerance == 0.0 {
                    assert_eq!((surface.u_pole_count(), surface.v_pole_count()), (25, 18));
                }
            }
        }
    }

    #[test]
    fn test_approximation_rejects_bad_grids() {
        let par = PointParametrization::Chordal;
        let points = wavy_grid(4, 5);
        let fixed =
            |points: &[Vec<Point3D>], p, mu| BSplineSurface::approximate(points, p, 1, mu, 3, par);
        assert_eq!(
            fixed(&points[..1], 1, 2),
            Err(BSplineConstructionError::TooFewPoints)
        );
        assert_eq!(
            fixed(&[], 1, 2),
            Err(BSplineConstructionError::TooFewPoints)
        );
        let mut ragged = points.clone();
        ragged[2].pop();
        assert_eq!(
            fixed(&ragged, 1, 2),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
        assert_eq!(
            fixed(&points, 0, 2),
            Err(BSplineConstructionError::InvalidDegree)
        );
        assert_eq!(
            fixed(&points, 2, 2),
            Err(BSplineConstructionError::DegreeTooHigh)
        );
        assert_eq!(
            fixed(&points, 1, 5),
            Err(BSplineConstructionError::TooManyPoles)
        );
        let within = |p| BSplineSurface::approximate_within(&points, p, 1, 1e-3, par);
        assert_eq!(within(4), Err(BSplineConstructionError::DegreeTooHigh));
        assert!(within(3).is_ok());
        // Every row the same point: no `v` parameters.
        let collapsed = vec![vec![Point3D::ORIGIN; 3]; 3];
        assert_eq!(
            BSplineSurface::approximate(&collapsed, 1, 1, 2, 2, par),
            Err(BSplineConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_parameters_skip_lines_without_length() {
        // A grid whose first row collapses to a point, as at a pole.
        let mut points = wavy_grid(4, 5);
        points[0] = vec![points[0][0]; 5];
        let grid = Grid::new(&points, PointParametrization::Chordal).unwrap();
        let rest = Grid::new(&points[1..], PointParametrization::Chordal).unwrap();
        for (a, b) in grid.v_params.iter().zip(&rest.v_params) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
