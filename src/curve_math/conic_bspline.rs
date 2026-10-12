//! Exact B-splines of conics, as control nets in the conic's plane: each
//! pole is a pair of coordinates along a frame's x and y directions, so a
//! curve places the net in its frame and a swept surface can reuse it per
//! profile pole.
//!
//! Circles (and, by scaling, ellipses) become rational quadratic pieces of
//! at most a quarter turn, hyperbolas one rational quadratic piece and
//! parabolas one polynomial quadratic piece. Every net is parametrised over
//! the conic's own range, with its knots at the parameters of the piece
//! boundaries.

use std::f64::consts::FRAC_PI_2;

/// A B-spline curve in a plane frame: poles as `[x, y]` coordinates along
/// the frame's directions, with the knots, multiplicities and (if rational)
/// weights of the curve.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlaneNet {
    pub(crate) degree: usize,
    pub(crate) poles: Vec<[f64; 2]>,
    pub(crate) weights: Option<Vec<f64>>,
    pub(crate) knots: Vec<f64>,
    pub(crate) mults: Vec<u32>,
    pub(crate) periodic: bool,
}

impl PlaneNet {
    /// Ends the clamped net exactly on its first pole, for a range that
    /// spans a whole period.
    pub(crate) fn close(&mut self) {
        let n = self.poles.len();
        self.poles[n - 1] = self.poles[0];
    }

    /// Turns a closed clamped net whose interior knots have multiplicity
    /// `degree` into a periodic net over the same knots: the repeated last
    /// pole goes, and the seam takes the interior multiplicity.
    pub(crate) fn make_periodic(&mut self) {
        debug_assert!(
            self.mults[1..self.mults.len() - 1]
                .iter()
                .all(|&m| m as usize == self.degree)
        );
        self.poles.pop();
        if let Some(weights) = &mut self.weights {
            weights.pop();
        }
        let n = self.mults.len();
        self.mults[0] = self.degree as u32;
        self.mults[n - 1] = self.degree as u32;
        self.periodic = true;
    }
}

/// The unit circle `(cos θ, sin θ)` over the angles `[first, last]`
/// (`0 < last - first <= 2π`, any start): rational quadratic pieces of
/// equal sweep, at most a quarter turn each, joined at double knots placed
/// at the piece-boundary angles. A piece from angle `m - h` to `m + h` has
/// its middle pole where the end tangents meet, `(cos m, sin m) / cos h`,
/// with weight `cos h` (The NURBS Book, A7.1).
pub(crate) fn circle_arc(first: f64, last: f64) -> PlaneNet {
    let sweep = last - first;
    // The slack keeps a range a rounding error above whole quarter turns
    // (a full turn passes `TrimmedCurve3D::new` up to `1e-12` over) from
    // getting an extra piece.
    let pieces = (sweep / FRAC_PI_2 - 1e-9).ceil().max(1.0) as usize;
    let mut knots = Vec::with_capacity(pieces + 1);
    knots.push(first);
    for k in 1..pieces {
        knots.push(first + sweep * (k as f64 / pieces as f64));
    }
    knots.push(last);

    let mut poles = Vec::with_capacity(2 * pieces + 1);
    let mut weights = Vec::with_capacity(2 * pieces + 1);
    poles.push([first.cos(), first.sin()]);
    weights.push(1.0);
    for piece in knots.windows(2) {
        let (start, end) = (piece[0], piece[1]);
        let (middle, half) = (0.5 * (start + end), 0.5 * (end - start));
        let w = half.cos();
        poles.push([middle.cos() / w, middle.sin() / w]);
        weights.push(w);
        poles.push([end.cos(), end.sin()]);
        weights.push(1.0);
    }

    let mut mults = vec![2; pieces + 1];
    mults[0] = 3;
    mults[pieces] = 3;
    PlaneNet {
        degree: 2,
        poles,
        weights: Some(weights),
        knots,
        mults,
        periodic: false,
    }
}

/// The whole unit circle as a periodic net over `[0, 2π]`: four quarter
/// turns, the seam at angle `0`.
pub(crate) fn full_circle() -> PlaneNet {
    let mut net = circle_arc(0.0, std::f64::consts::TAU);
    net.close();
    net.make_periodic();
    net
}

/// The unit hyperbola branch `(cosh u, sinh u)` over `[first, last]`: one
/// rational quadratic piece whose middle pole is where the end tangents
/// meet, `(cosh m, sinh m) / cosh h` with weight `cosh h`, for the middle
/// `m` and half width `h` of the range.
pub(crate) fn hyperbola_arc(first: f64, last: f64) -> PlaneNet {
    let (middle, half) = (0.5 * (first + last), 0.5 * (last - first));
    let w = half.cosh();
    PlaneNet {
        degree: 2,
        poles: vec![
            [first.cosh(), first.sinh()],
            [middle.cosh() / w, middle.sinh() / w],
            [last.cosh(), last.sinh()],
        ],
        weights: Some(vec![1.0, w, 1.0]),
        knots: vec![first, last],
        mults: vec![3, 3],
        periodic: false,
    }
}

/// The parabola `(u² / (4 focal), u)` over `[first, last]`: one polynomial
/// quadratic piece, the middle pole where the end tangents meet. Being a
/// polynomial of degree two in `u`, it keeps the parameter exactly.
pub(crate) fn parabola_arc(focal: f64, first: f64, last: f64) -> PlaneNet {
    PlaneNet {
        degree: 2,
        poles: vec![
            [first * first / (4.0 * focal), first],
            [first * last / (4.0 * focal), 0.5 * (first + last)],
            [last * last / (4.0 * focal), last],
        ],
        weights: None,
        knots: vec![first, last],
        mults: vec![3, 3],
        periodic: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    #[test]
    fn test_circle_arc_piece_count() {
        let pieces = |first: f64, last: f64| circle_arc(first, last).knots.len() - 1;
        assert_eq!(pieces(0.0, 1e-9), 1);
        assert_eq!(pieces(0.0, FRAC_PI_2), 1);
        assert_eq!(pieces(0.0, FRAC_PI_2 + 1e-6), 2);
        assert_eq!(pieces(-1.0, -1.0 + PI), 2);
        assert_eq!(pieces(5.0, 5.0 + TAU), 4);
        assert_eq!(pieces(0.0, TAU * (1.0 + 1e-12)), 4);
    }

    #[test]
    fn test_circle_arc_layout() {
        let net = circle_arc(0.0, PI);
        assert_eq!(net.knots, vec![0.0, FRAC_PI_2, PI]);
        assert_eq!(net.mults, vec![3, 2, 3]);
        assert_eq!(net.poles.len(), 5);
        let w = net.weights.as_ref().unwrap();
        assert!((w[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
        assert!((net.poles[1][0] - 1.0).abs() < 1e-15 && (net.poles[1][1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_full_circle_is_periodic_without_the_repeated_pole() {
        let net = full_circle();
        assert!(net.periodic);
        assert_eq!(net.knots, vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2, TAU]);
        assert_eq!(net.mults, vec![2; 5]);
        assert_eq!(net.poles.len(), 8);
        assert_eq!(net.weights.as_ref().unwrap().len(), 8);
        assert_eq!(net.poles[0], [1.0, 0.0]);
    }

    #[test]
    fn test_hyperbola_and_parabola_middle_poles_meet_the_end_tangents() {
        let (u0, u1) = (-0.5, 2.0);
        let net = hyperbola_arc(u0, u1);
        let tangent_cross =
            |p: [f64; 2], t: [f64; 2], q: [f64; 2]| (q[0] - p[0]) * t[1] - (q[1] - p[1]) * t[0];
        let mid = net.poles[1];
        assert!(tangent_cross(net.poles[0], [u0.sinh(), u0.cosh()], mid).abs() < 1e-14);
        assert!(tangent_cross(net.poles[2], [u1.sinh(), u1.cosh()], mid).abs() < 1e-14);

        let f = 0.75;
        let net = parabola_arc(f, u0, u1);
        let mid = net.poles[1];
        assert!(tangent_cross(net.poles[0], [u0 / (2.0 * f), 1.0], mid).abs() < 1e-14);
        assert!(tangent_cross(net.poles[2], [u1 / (2.0 * f), 1.0], mid).abs() < 1e-14);
    }
}
