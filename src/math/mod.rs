//! Small numerical toolbox for the kernel's iterative algorithms.
//!
//! Everything here is tiny and fixed-size: real roots of polynomials of
//! degree ≤ 4, Newton iteration on 1–3 unknowns, Gauss-Newton steps for
//! least-squares projection, Brent refinement, and 2x2/3x3 solves on
//! stack arrays. That is deliberate: projection, extrema, and
//! intersection never need large dense systems, so the kernel keeps zero
//! runtime dependencies and no linear-algebra crate in its public API.
//!
//! Near-multiple roots are first-class: [`real_roots`] clusters with
//! multiplicity, because tangencies (a grazing line, kissing circles)
//! *are* near-double roots and the intersection APIs must report them,
//! not lose them to floating-point noise.

pub(crate) mod poly;
pub(crate) mod solve;

// Re-exported for the curve-curve/generic consumers landing next;
// the allow expires then.
#[allow(dead_code, unused_imports)]
pub(crate) use poly::RealRoot;
pub(crate) use poly::real_roots;
#[allow(dead_code, unused_imports)]
pub(crate) use solve::{brent_minimum, newton_1d};
pub(crate) use solve::{gauss_newton_1d, gauss_newton_2d, newton_3d, solve_2x2};
