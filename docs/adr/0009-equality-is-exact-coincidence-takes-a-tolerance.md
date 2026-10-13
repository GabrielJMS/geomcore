# Equality is exact; coincidence takes a tolerance

`==` on a geomcore value (point, vector, frame, curve, surface) compares it field by field, exactly, in Rust and in Python alike: two values are equal when they are the same value. Whether two points coincide is a geometric decision, and like every other one it is a distance decision against the confusion tolerance the caller passes: `p.distance(q) <= tol.confusion`, `contains(point, tol)`, the intersections' `Tolerance` argument. A tolerance is always a parameter of a query, never part of a value: no geomcore object stores one, and no operator hides one.

A tolerance stored on an object, and widened by operations to close the gaps between the geometry that meets there, is an entity tolerance: a vertex whose edges' curves end up to 3e-5 apart carries 3e-5. That is topology, which belongs to the caller. A point that remembered its own tolerance would be a vertex.

## Considered options

- **Tolerance-aware `==`** (within `tol.confusion`, or within a scaled floating-point zero as one mature kernel does for its vectors): rejected. It is not transitive (`a == b` and `b == c` do not give `a == c`), so sorting, deduplication and hashing break; it hides a tolerance in an operator, against the rule that every geometric decision is a distance decision against a tolerance the caller chose; and no single global threshold is right for every model.
- **A tolerance stored on points, curves or surfaces**: rejected. A per-entity tolerance that grows with operations is the topology layer's state; geometry answers queries for whatever tolerance it is given and stays stateless.
- **No value equality in the Python bindings** (identity, the default): rejected. `Point3D(1, 2, 3) == Point3D(1, 2, 3)` was `False` and points hashed by identity, so sets never deduplicated equal points and the documented asserts failed (#65). Python mirrors Rust; since `f64` is not `Eq`, the value types are unhashable.
