# B-spline poles stay Cartesian, their weights apart

A B-spline stores its poles once, as Cartesian points, with its weights in a separate list when it is rational; a surface keeps both flat, row after row. The poles a caller reads are the poles it gave, and a polynomial B-spline carries no weights at all. Editing (knot insertion and removal, degree elevation, splitting) works on homogeneous poles only inside the window of poles it changes: the untouched poles and weights are copied as they are, bit for bit, and only the changed ones are divided back. Converting just that window is cheap: rational knot insertion runs at 0.87 to 0.95 of the time of the faster C++ library benchmarked.

## Considered options

- **Homogeneous storage (`[w x, w y, w z, w]`)**: rejected. Every reader of the poles, and every polynomial B-spline, would pay for the weights or a division, and an edit would round untouched poles through a multiplication and a division instead of leaving them exact.
- **Caching the homogeneous poles beside the Cartesian ones**: rejected. It doubles the memory of a rational B-spline for a conversion that editing needs only over a few poles.
- **Nested rows of poles for surfaces**: rejected. One flat list per surface keeps a whole net contiguous for evaluation and editing; the constructors still accept rows.
