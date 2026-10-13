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
pub(crate) fn averaged_knots(params: &[f64], degree: usize) -> Vec<f64> {
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

/// Clamped flat knots of `degree` over `[0, 1]` for `pole_count` poles
/// fitted to points at `params` (at least `pole_count` of them), interior
/// knots spread along the parameters so that every knot span holds at
/// least one: the least-squares system then has a unique solution
/// however unevenly the points are spaced.
pub(crate) fn approximation_knots(params: &[f64], pole_count: usize, degree: usize) -> Vec<f64> {
    let (n, m, p) = (params.len(), pole_count, degree);
    let mut flat = Vec::with_capacity(m + p + 1);
    flat.resize(p + 1, 0.0);
    // Knot j sits at the fractional parameter index j (n - 1) / (m - p),
    // a step above one with fewer poles than points: evenly spaced
    // parameters give evenly spaced knots.
    let step = (n - 1) as f64 / (m - p) as f64;
    flat.extend((1..m - p).map(|j| {
        let at = j as f64 * step;
        let i = at as usize;
        let fraction = at - i as f64;
        (1.0 - fraction) * params[i] + fraction * params[i + 1]
    }));
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

pub(crate) fn check_degree(degree: usize) -> Result<(), BSplineConstructionError> {
    if (1..=math::MAX_DEGREE).contains(&degree) {
        Ok(())
    } else {
        Err(BSplineConstructionError::InvalidDegree)
    }
}

/// Increasing parameters in `[0, 1]` for the `count` points that each of
/// `lines` lines of points holds, `point(r, k)` being point `k` of line
/// `r`: every line's own parameters (as [`point_parameters`] gives them),
/// averaged over the lines that have a length. `ConfusedPoints` if none
/// has, or if two averaged parameters coincide.
pub(crate) fn shared_parameters(
    lines: usize,
    count: usize,
    point: impl Fn(usize, usize) -> Point3D,
    parametrization: PointParametrization,
) -> Result<Vec<f64>, BSplineConstructionError> {
    let exponent = parametrization.exponent();
    let mut params = vec![0.0; count];
    let mut line = vec![0.0; count];
    let mut used = 0usize;
    for r in 0..lines {
        let (mut length, mut total) = (0.0, 0.0);
        for (k, t) in line.iter_mut().enumerate().skip(1) {
            let d = point(r, k).distance(point(r, k - 1));
            length += d;
            total += if exponent == 1.0 { d } else { d.sqrt() };
            *t = total;
        }
        if length <= crate::tol::CONFUSION {
            continue;
        }
        used += 1;
        let inverse = 1.0 / total;
        for (sum, &t) in params[1..].iter_mut().zip(&line[1..]) {
            *sum += t * inverse;
        }
    }
    if used == 0 {
        return Err(BSplineConstructionError::ConfusedPoints);
    }
    let inverse = 1.0 / used as f64;
    for t in &mut params[1..] {
        *t *= inverse;
    }
    params[count - 1] = 1.0;
    if params.windows(2).any(|w| w[1] <= w[0]) {
        return Err(BSplineConstructionError::ConfusedPoints);
    }
    Ok(params)
}

/// The basis functions of `degree` on flat knots non-zero at each of
/// increasing parameters: the first of their poles and their
/// `degree + 1` values, found once for every line of data fitted on
/// those parameters.
pub(crate) struct BasisTable {
    width: usize,
    firsts: Vec<usize>,
    values: Vec<f64>,
}

impl BasisTable {
    pub(crate) fn new(flat: &[f64], degree: usize, params: &[f64]) -> Self {
        let width = degree + 1;
        let mut rows = BasisRows::new(flat, degree);
        let mut firsts = Vec::with_capacity(params.len());
        let mut values = Vec::with_capacity(params.len() * width);
        for &t in params {
            firsts.push(rows.at(t));
            values.extend_from_slice(&rows.values[..width]);
        }
        BasisTable {
            width,
            firsts,
            values,
        }
    }

    /// The number of parameters.
    pub(crate) fn len(&self) -> usize {
        self.firsts.len()
    }

    /// The first pole and the basis values at parameter `k`.
    pub(crate) fn at(&self, k: usize) -> (usize, &[f64]) {
        let values = &self.values[k * self.width..(k + 1) * self.width];
        (self.firsts[k], values)
    }

    /// The point at parameter `k` of the curve with `poles`.
    pub(crate) fn point(&self, k: usize, poles: &[Point3D]) -> Point3D {
        let (first, values) = self.at(k);
        let mut x = [0.0; 3];
        for (&v, q) in values.iter().zip(&poles[first..]) {
            x[0] += v * q.x;
            x[1] += v * q.y;
            x[2] += v * q.z;
        }
        Point3D::new(x[0], x[1], x[2])
    }
}

/// Least-squares poles, `pole_count` of them on `basis`'s knots, of each
/// of `lines` lines of data, point `k` of line `r` being `point(r, k)`:
/// the poles of line `r` from `r * pole_count` on. The normal equations
/// `A'A x = A'b`, of half-bandwidth `degree`, are factored once by banded
/// Cholesky for all the lines. `None` when the data leaves a pole (nearly)
/// undetermined.
pub(crate) fn least_squares(
    basis: &BasisTable,
    pole_count: usize,
    lines: usize,
    point: impl Fn(usize, usize) -> Point3D,
) -> Option<Vec<Point3D>> {
    let (width, m) = (basis.width, pole_count);
    let p = width - 1;
    let mut lower = vec![0.0; m * width];
    for k in 0..basis.len() {
        let (first, values) = basis.at(k);
        for (a, &va) in values.iter().enumerate() {
            let i = first + a;
            let row = &mut lower[i * width..(i + 1) * width];
            // Columns first..=i sit at offsets first + p - i..=p.
            for (b, &vb) in values[..=a].iter().enumerate() {
                row[b + p - a] += va * vb;
            }
        }
    }
    if !factor_symmetric_band(&mut lower, p, 1e-12) {
        return None;
    }
    let mut poles = vec![Point3D::ORIGIN; lines * m];
    for (r, line) in poles.chunks_exact_mut(m).enumerate() {
        for k in 0..basis.len() {
            let (first, values) = basis.at(k);
            let q = point(r, k);
            for (&v, pole) in values.iter().zip(&mut line[first..]) {
                pole.x += v * q.x;
                pole.y += v * q.y;
                pole.z += v * q.z;
            }
        }
        substitute_symmetric_band(&lower, p, line);
    }
    poles.iter().all(|q| q.is_finite()).then_some(poles)
}

/// The interpolating poles, one per parameter of `basis`, whose knots
/// average the parameters (see [`averaged_knots`]), of each of `lines`
/// lines of data as for [`least_squares`]. The band matrix is factored
/// once, by LU without pivoting (safe: averaged knots satisfy
/// Schoenberg-Whitney, so the matrix is totally positive), for all the
/// lines; each line's end poles are its end points.
pub(crate) fn interpolation(
    basis: &BasisTable,
    lines: usize,
    point: impl Fn(usize, usize) -> Point3D,
) -> Vec<Point3D> {
    let n = basis.len();
    let p = basis.width - 1;
    // Row j holds the basis functions at parameter j, over poles
    // `first..=first + p`. With averaged knots `A[j][i] = 0` whenever
    // `|i - j| >= p` (each such basis function's support ends at or
    // starts after parameter j), so the band is `p - 1` wide each side.
    let h = p - 1;
    let width = 2 * h + 1;
    let mut band = vec![0.0; n * width];
    for j in 0..n {
        let (first, values) = basis.at(j);
        let row = &mut band[j * width..(j + 1) * width];
        for (a, &value) in values.iter().enumerate() {
            let column = first + a;
            if column + h >= j && column <= j + h {
                row[column + h - j] = value;
            } else {
                debug_assert_eq!(value, 0.0, "row {j}, column {column}");
            }
        }
    }
    factor_band(&mut band, h);
    let mut poles = Vec::with_capacity(lines * n);
    for r in 0..lines {
        let start = poles.len();
        poles.extend((0..n).map(|k| point(r, k)));
        let line = &mut poles[start..];
        substitute_band(&band, h, line);
        // The end rows are unit rows: the end poles are the end points exactly.
        line[0] = point(r, 0);
        line[n - 1] = point(r, n - 1);
    }
    poles
}

/// The fit that `attempt` makes with a pole count in `first..last`, the
/// count found by galloping up from `first`, then bisecting: the fewest
/// when fitting is monotone in the count. `None` if no count below `last`
/// fits, `last` being the count the caller falls back on (and never
/// attempted).
pub(crate) fn fit_with_fewest_poles<F>(
    first: usize,
    last: usize,
    mut attempt: impl FnMut(usize) -> Option<F>,
) -> Option<F> {
    // `low` does not fit (or is below the range); `high` does.
    let (mut low, mut high) = (first - 1, last);
    let mut fit = None;
    let mut step = 1;
    while low + step < high {
        let count = low + step;
        if let Some(f) = attempt(count) {
            (high, fit) = (count, Some(f));
            break;
        }
        low = count;
        step *= 2;
    }
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        match attempt(middle) {
            Some(f) => (high, fit) = (middle, Some(f)),
            None => low = middle,
        }
    }
    fit
}

/// The fit that `attempt` makes on knots refined where it misses: from no
/// interior knots, each round splits the knot spans whose points deviate
/// beyond `max_deviation`, between the middle two of their parameters, so
/// knots gather where the data bends. `attempt(flat, basis, deviations)`
/// fits on the clamped flat knots `flat`, `basis` holding their basis at
/// `params`, filling the deviation at each parameter; `None` when the fit
/// is undetermined. A round splits only the spans whose excess over
/// `max_deviation` is at least half the largest, unless none of those can
/// be split: inserting fewer knots per round spends fewer poles. A span
/// holding a single parameter cannot, so its neighbours are split for
/// it. `None` when no span can be split, or the pole count would reach
/// the parameter count.
pub(crate) fn fit_with_refined_knots<F>(
    params: &[f64],
    degree: usize,
    max_deviation: f64,
    mut attempt: impl FnMut(&[f64], &BasisTable, &mut [f64]) -> Option<F>,
) -> Option<(Vec<f64>, F)> {
    let (n, p) = (params.len(), degree);
    let mut flat = vec![0.0; p + 1];
    flat.resize(2 * p + 2, 1.0);
    let mut deviations = vec![0.0; n];
    // Per span of points: its parameter range and largest deviation.
    let mut spans: Vec<(usize, usize, f64)> = Vec::new();
    let mut split: Vec<bool> = Vec::new();
    loop {
        let basis = BasisTable::new(&flat, p, params);
        let fit = attempt(&flat, &basis, &mut deviations)?;
        let largest = deviations.iter().fold(0.0f64, |a, &d| a.max(d));
        if largest <= max_deviation {
            return Some((flat, fit));
        }
        spans.clear();
        let mut start = 0;
        for k in 1..=n {
            if k == n || basis.firsts[k] != basis.firsts[start] {
                let worst = deviations[start..k].iter().fold(0.0f64, |a, &d| a.max(d));
                spans.push((start, k, worst));
                start = k;
            }
        }
        // The spans whose excess over `max_deviation` is at least half the
        // largest, else all those beyond it; a span with a single
        // parameter has its neighbours split instead.
        let half = 0.5 * (max_deviation + largest);
        split.clear();
        split.resize(spans.len(), false);
        for threshold in [half, max_deviation] {
            for (s, &(start, end, worst)) in spans.iter().enumerate() {
                if worst > max_deviation && worst >= threshold {
                    if end - start >= 2 {
                        split[s] = true;
                    } else {
                        for t in [s.wrapping_sub(1), s + 1] {
                            if spans.get(t).is_some_and(|&(a, b, _)| b - a >= 2) {
                                split[t] = true;
                            }
                        }
                    }
                }
            }
            if split.contains(&true) {
                break;
            }
        }
        let count = flat.len() - p - 1;
        let before = flat.len();
        for (&(start, end, _), _) in spans.iter().zip(&split).filter(|(_, split)| **split) {
            let middle = start + (end - start) / 2;
            flat.push(0.5 * (params[middle - 1] + params[middle]));
        }
        let inserted = flat.len() - before;
        if inserted == 0 || count + inserted >= n {
            return None;
        }
        flat.sort_by(f64::total_cmp);
    }
}

/// The fit within `max_deviation` that `attempt` makes (as for
/// [`fit_with_refined_knots`]) with few poles: on refined knots, or when
/// refining gets stuck, on [`approximation_knots`] with the fewest poles
/// of [`fit_with_fewest_poles`]. `None` if no fit with fewer poles than
/// parameters is close enough.
pub(crate) fn fit_within<F>(
    params: &[f64],
    degree: usize,
    max_deviation: f64,
    mut attempt: impl FnMut(&[f64], &BasisTable, &mut [f64]) -> Option<F>,
) -> Option<(Vec<f64>, F)> {
    fit_with_refined_knots(params, degree, max_deviation, &mut attempt)
        .or_else(|| fit_on_spread_knots(params, degree, max_deviation, params.len(), attempt))
}

/// As [`fit_within`], with fewer poles at about twice the cost: the
/// fewest poles on [`approximation_knots`] are searched below those of
/// the refined fit too, the fewer kept. Few points, unevenly bent, can
/// fit with fewer poles on knots spread along the parameters than on
/// knots refined in halves.
pub(crate) fn fit_within_fewest<F>(
    params: &[f64],
    degree: usize,
    max_deviation: f64,
    mut attempt: impl FnMut(&[f64], &BasisTable, &mut [f64]) -> Option<F>,
) -> Option<(Vec<f64>, F)> {
    let refined = fit_with_refined_knots(params, degree, max_deviation, &mut attempt);
    let last = refined
        .as_ref()
        .map_or(params.len(), |(flat, _)| flat.len() - degree - 1);
    fit_on_spread_knots(params, degree, max_deviation, last, attempt).or(refined)
}

/// The fit within `max_deviation` that `attempt` makes (as for
/// [`fit_with_refined_knots`]) on [`approximation_knots`], with the
/// fewest poles below `last` of [`fit_with_fewest_poles`].
fn fit_on_spread_knots<F>(
    params: &[f64],
    degree: usize,
    max_deviation: f64,
    last: usize,
    mut attempt: impl FnMut(&[f64], &BasisTable, &mut [f64]) -> Option<F>,
) -> Option<(Vec<f64>, F)> {
    let mut deviations = vec![0.0; params.len()];
    fit_with_fewest_poles(degree + 1, last, |count| {
        let flat = approximation_knots(params, count, degree);
        let basis = BasisTable::new(&flat, degree, params);
        let fit = attempt(&flat, &basis, &mut deviations)?;
        deviations
            .iter()
            .all(|&d| d <= max_deviation)
            .then_some((flat, fit))
    })
}

/// LU without pivoting, in place, of a band matrix stored row by row,
/// `band[i * width + (j + bandwidth - i)]` holding `A[i][j]` for
/// `|i - j| <= bandwidth` (`width = 2 * bandwidth + 1`): the multipliers
/// take the place of the entries they eliminate, for
/// [`substitute_band`].
fn factor_band(band: &mut [f64], bandwidth: usize) {
    let h = bandwidth;
    let w = 2 * h + 1;
    let n = band.len() / w;
    for k in 0..n {
        let end = (k + h + 1).min(n);
        let (done, rest) = band.split_at_mut((k + 1) * w);
        // Row k from its diagonal on: columns k..end.
        let pivot_row = &done[k * w + h..k * w + h + (end - k)];
        let inverse = 1.0 / pivot_row[0];
        for (d, row) in rest.chunks_exact_mut(w).take(end - k - 1).enumerate() {
            // Row i = k + 1 + d holds column k at offset h - d - 1.
            let start = h - d - 1;
            let factor = row[start] * inverse;
            row[start] = factor;
            if factor == 0.0 {
                continue;
            }
            for (a, &b) in row[start + 1..start + end - k]
                .iter_mut()
                .zip(&pivot_row[1..])
            {
                *a -= factor * b;
            }
        }
    }
}

/// Solves `A x = b` in place with the factors [`factor_band`] left in
/// `band`: `rhs` holds `b` and receives `x`.
fn substitute_band(band: &[f64], bandwidth: usize, rhs: &mut [Point3D]) {
    let (n, h) = (rhs.len(), bandwidth);
    let w = 2 * h + 1;
    for k in 0..n {
        let end = (k + h + 1).min(n);
        let bk = rhs[k];
        for d in 0..end - k - 1 {
            let factor = band[(k + 1 + d) * w + h - d - 1];
            if factor == 0.0 {
                continue;
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

/// Cholesky (`A = L L'`, `L` overwriting `lower`), in place, of a
/// symmetric positive definite band matrix whose lower band is stored row
/// by row, `lower[i * (h + 1) + (j + h - i)]` holding `A[i][j]` for
/// `i - h <= j <= i`. Returns false when a pivot falls to
/// `relative_floor` times the largest diagonal or below: a pole the data
/// leaves (nearly) undetermined.
fn factor_symmetric_band(lower: &mut [f64], h: usize, relative_floor: f64) -> bool {
    let w = h + 1;
    let n = lower.len() / w;
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
    true
}

/// Solves `A x = b` in place with the factor [`factor_symmetric_band`]
/// left in `lower`: `rhs` holds `b` and receives `x`.
fn substitute_symmetric_band(lower: &[f64], h: usize, rhs: &mut [Point3D]) {
    let n = rhs.len();
    let w = h + 1;
    let at = |i: usize, j: usize| i * w + j + h - i;
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
}

/// The interpolation of `points` at `params` by a clamped B-spline of
/// `degree`: its flat knots (averaged) and poles.
fn interpolated(points: &[Point3D], params: &[f64], degree: usize) -> (Vec<f64>, Vec<Point3D>) {
    let flat = averaged_knots(params, degree);
    let basis = BasisTable::new(&flat, degree, params);
    let poles = interpolation(&basis, 1, |_, k| points[k]);
    (flat, poles)
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
        let (flat, poles) = interpolated(points, &params, degree);
        let (knots, mults) = compress_knots(&flat);
        Ok(BSplineCurve3D::from_validated_parts(
            degree, poles, None, knots, mults, flat, false,
        ))
    }

    /// Least-squares B-spline fit through `points` with `num_poles` poles.
    ///
    /// Same chord-like parametrization as
    /// [`BSplineCurve3D::interpolate`], but an overdetermined system:
    /// interior knots spread along the parameters so that every knot span
    /// holds some, and the normal equations `A'A x = A'b`, whose matrix
    /// has half-bandwidth `degree`, solved by banded Cholesky. The fit
    /// approximates rather than hits the data; poles left numerically
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
        let flat = approximation_knots(&params, num_poles, degree);
        let basis = BasisTable::new(&flat, degree, &params);
        let poles = least_squares(&basis, num_poles, 1, |_, k| points[k])
            .ok_or(BSplineConstructionError::PoleCountMismatch)?;
        let (knots, mults) = compress_knots(&flat);
        Ok(BSplineCurve3D::from_validated_parts(
            degree, poles, None, knots, mults, flat, false,
        ))
    }

    /// Least-squares B-spline fit through `points`, with few poles, that
    /// keeps every point within `max_deviation` of the curve.
    ///
    /// Each candidate is a least-squares fit, as in
    /// [`BSplineCurve3D::approximate`], its deviation measured from each
    /// point to the curve at the point's parameter (which bounds the
    /// distance to the curve). Starting from a single Bézier piece, each
    /// round splits the knot spans that deviate the most between their
    /// middle points, so knots gather where the points bend. Should no
    /// span be left to split, the pole count of
    /// [`BSplineCurve3D::approximate`] is searched instead, galloping up
    /// from `degree + 1`, then bisecting. When no least-squares fit is
    /// close enough, as with a `max_deviation` of zero, the result is the
    /// interpolation of [`BSplineCurve3D::interpolate`], one pole per
    /// point.
    ///
    /// # Errors
    ///
    /// [`BSplineConstructionError::TooFewPoints`] for fewer than two
    /// points, [`BSplineConstructionError::InvalidDegree`] outside
    /// `1..=25`, [`BSplineConstructionError::DegreeTooHigh`] when
    /// `degree` exceeds points minus one, or
    /// [`BSplineConstructionError::ConfusedPoints`] for coincident
    /// consecutive points.
    ///
    /// # Examples
    ///
    /// ```
    /// use geomcore::{BSplineCurve3D, Point3D, PointParametrization, Tolerance};
    /// let points: Vec<Point3D> = (0..50)
    ///     .map(|i| {
    ///         let u = i as f64 / 49.0 * std::f64::consts::PI;
    ///         Point3D::new(u.cos(), u.sin(), 0.0)
    ///     })
    ///     .collect();
    /// let curve =
    ///     BSplineCurve3D::approximate_within(&points, 3, 1e-4, PointParametrization::Chordal)
    ///         .unwrap();
    /// // 50 points, 11 poles.
    /// assert_eq!(curve.poles().len(), 11);
    /// let tol = Tolerance::DEFAULT;
    /// for p in &points {
    ///     assert!(curve.project_point(*p, tol).distance <= 1e-4);
    /// }
    /// ```
    pub fn approximate_within(
        points: &[Point3D],
        degree: usize,
        max_deviation: f64,
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
        let fit = fit_within(&params, degree, max_deviation, |flat, basis, deviations| {
            let poles = least_squares(basis, flat.len() - degree - 1, 1, |_, k| points[k])?;
            for (k, deviation) in deviations.iter_mut().enumerate() {
                *deviation = basis.point(k, &poles).distance(points[k]);
            }
            Some(poles)
        });
        let (flat, poles) = fit.unwrap_or_else(|| interpolated(points, &params, degree));
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
            let flat = approximation_knots(&params, m, p);
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
        // Two clusters a millionth wide and far apart leave a 12-pole fit
        // numerically undetermined: the basis functions barely reach the
        // points.
        let mut points: Vec<Point3D> = (0..6)
            .map(|i| Point3D::new(i as f64 * 1e-6, 0.0, 0.0))
            .collect();
        points.extend((0..6).map(|i| Point3D::new(10.0 + i as f64 * 1e-6, 1.0, 0.0)));
        assert_eq!(
            BSplineCurve3D::approximate(&points, 3, 12, PointParametrization::Chordal),
            Err(BSplineConstructionError::PoleCountMismatch)
        );
    }

    #[test]
    fn test_approximation_within_keeps_points_close_with_few_poles() {
        let points = wavy_points(200);
        for p in 1..=4 {
            for tolerance in [1e-1, 1e-3, 1e-5] {
                let curve = BSplineCurve3D::approximate_within(
                    &points,
                    p,
                    tolerance,
                    PointParametrization::Chordal,
                )
                .unwrap();
                let params = point_parameters(&points, PointParametrization::Chordal).unwrap();
                for (q, &t) in points.iter().zip(&params) {
                    assert!(curve.eval_point(t).distance(*q) <= tolerance, "degree {p}");
                }
                assert!(curve.poles().len() < points.len(), "degree {p}");
            }
        }
    }

    #[test]
    fn test_approximation_within_gathers_knots_where_points_bend() {
        // Straight for x in [0, 2], then a tight wave over [2, 3], joined
        // smoothly enough for a cubic.
        let points: Vec<Point3D> = (0..300)
            .map(|i| {
                let x = 3.0 * i as f64 / 299.0;
                let t = (x - 2.0).max(0.0);
                let y = t.powi(4) * (12.0 * t).sin();
                Point3D::new(x, y, 0.0)
            })
            .collect();
        let curve =
            BSplineCurve3D::approximate_within(&points, 3, 1e-4, PointParametrization::Chordal)
                .unwrap();
        // The wave, under a third of the length, holds most interior knots.
        let start = curve
            .project_point(Point3D::new(2.0, 0.0, 0.0), crate::Tolerance::DEFAULT)
            .parameter;
        let interior = &curve.knots()[1..curve.knots().len() - 1];
        let in_wave = interior.iter().filter(|&&u| u > start).count();
        assert!(
            4 * in_wave > 3 * interior.len(),
            "{in_wave} of {}",
            interior.len()
        );
    }

    #[test]
    fn test_approximation_within_zero_deviation_interpolates() {
        let points = wavy_points(12);
        for par in [
            PointParametrization::Chordal,
            PointParametrization::Centripetal,
        ] {
            let within = BSplineCurve3D::approximate_within(&points, 3, 0.0, par).unwrap();
            assert_eq!(
                within,
                BSplineCurve3D::interpolate(&points, 3, par).unwrap()
            );
            let nan = BSplineCurve3D::approximate_within(&points, 3, f64::NAN, par).unwrap();
            assert_eq!(nan, within);
        }
    }

    #[test]
    fn test_approximation_within_rejects_bad_input() {
        let points = wavy_points(4);
        let fit = |points: &[Point3D], degree| {
            BSplineCurve3D::approximate_within(points, degree, 1e-3, PointParametrization::Chordal)
        };
        assert_eq!(
            fit(&points[..1], 1),
            Err(BSplineConstructionError::TooFewPoints)
        );
        assert_eq!(
            fit(&points, 0),
            Err(BSplineConstructionError::InvalidDegree)
        );
        assert_eq!(
            fit(&points, 4),
            Err(BSplineConstructionError::DegreeTooHigh)
        );
        let confused = [points[0], points[0], points[1]];
        assert_eq!(
            fit(&confused, 1),
            Err(BSplineConstructionError::ConfusedPoints)
        );
    }

    #[test]
    fn test_fewest_poles_search_finds_the_threshold() {
        for threshold in 3..=40 {
            let mut attempts = 0;
            let found = fit_with_fewest_poles(3, 41, |count| {
                attempts += 1;
                (count >= threshold).then_some(count)
            });
            assert_eq!(found, (threshold < 41).then_some(threshold));
            assert!(attempts <= 12, "{threshold}: {attempts}");
        }
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
