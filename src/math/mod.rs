//! Small numerical toolbox for the kernel's iterative algorithms.
//!
//! Everything here is tiny and fixed-size: real roots of polynomials of
//! degree ≤ 4, Newton iteration and polishing on 1–3 unknowns,
//! box-bounded Newton minimisation (with a box hierarchy for
//! nearest-first search over B-spline pieces), Brent refinement, and
//! 2x2 to 4x4 solves on stack arrays. That is deliberate: projection, extrema, and
//! intersection never need large dense systems, so the kernel keeps zero
//! runtime dependencies and no linear-algebra crate in its public API.
//!
//! Near-multiple roots are first-class: `real_roots` clusters with
//! multiplicity, because tangencies (a grazing line, kissing circles)
//! *are* near-double roots and the intersection APIs must report them,
//! not lose them to floating-point noise.

pub(crate) mod bernstein;
pub(crate) mod boxtree;
pub(crate) mod equation;
pub(crate) mod poly;
pub(crate) mod quadrature;
pub(crate) mod solve;

pub(crate) use equation::{
    QuadraticSolution, quadratic_centre, roots_with_tangency, solve_quadratic, solve_trig,
    solve_trig_linear,
};
pub(crate) use poly::{RealRoot, real_roots};
pub(crate) use solve::{
    brent_minimum, newton_3d, newton_min_1d, newton_min_2d, polish_root, solve_2x2, solve_3x3,
    solve_4x4,
};
