# API overview

A map of the Python API. Every class and method carries a docstring
(`help(geomcore.curves.Circle3D)`), and the Rust API reference on
[docs.rs/geomcore](https://docs.rs/geomcore) documents the exact semantics —
the bindings mirror it one-to-one.

## `geomcore` — core types

| Class | Purpose | Key members |
|---|---|---|
| `Point3` / `Point2` | locations | `new`, `origin`, `x`/`y`/`z`, `distance` |
| `Vector3` / `Vector2` | directions & displacements | `new`, `x()`/`y()`/`z()`/`zero()`, `dot`, `cross`, `magnitude`, `components` |
| `Axis3` | located direction | `new`, `origin`, `direction` |
| `Frame3` | right-handed coordinate system | `new`, `from_z`, `world`, `origin`, `x_direction`, `y_direction`, `z_direction` |
| `Transform` | rigid motion + scaling/mirroring | `translation`, `rotation`, `scaling`, `mirror_point`, `mirror_axis`, `mirror_plane`, `identity`, `then`, `apply_point`, `apply_vector` |

## `geomcore.curves`

All curves support `eval_point(t)`, `eval_derivative(t, order)`,
`eval_points(list)`, and `parametrize_on(surface)` (closed forms exist today
for lines and circles on the five elementary surfaces; other pairs raise
`ValueError`). Analytic curves add `parameter_of(point)`.

| Class | Extra constructors | Notable members |
|---|---|---|
| `Line3D` | `from_two_points`, `from_axis` | `origin`, `direction` |
| `Circle3D` | `from_axis`, `from_frame`, `from_three_points` | `center`, `radius`, `normal` |
| `Ellipse3D` | `from_frame`, `from_center_and_points` | `major_radius`, `minor_radius` |
| `Parabola3D` | `from_frame` | `focal` |
| `Hyperbola3D` | `from_frame`, `from_center_and_points` | `center`, `major_radius`, `minor_radius` |
| `BSplineCurve3D` | `new_rational` | `degree`, `bounds`, `is_rational`, `is_periodic` |
| `Line2D`, `Circle2D` | — | 2D twins, returned by `parametrize_on` |

## `geomcore.surfaces`

All surfaces support `eval_point(u, v)`, `eval_derivative(u, v, nu, nv)`,
and `eval_points(list_of_uv_pairs)`; the elementary ones add
`parameters_of(point)`.

| Class | Extra constructors | Notable members |
|---|---|---|
| `Plane` | `from_frame`, `from_three_points`, `from_coefficients` | `normal` |
| `Cylinder` | `from_frame`, `from_axis`, `from_circle` | `radius` |
| `Cone` | `from_frame`, `from_two_points_and_radii` | `apex`, `semi_angle`, `ref_radius` |
| `Sphere` | `from_frame` | `center`, `radius` |
| `Torus` | `from_frame` | `major_radius`, `minor_radius` |
| `BSplineSurface` | `new_rational` | `u_degree`, `v_degree`, `is_u_periodic`, `is_v_periodic`, `is_rational` |

## Conventions

- Angles are radians; parameters follow standard CAD conventions (angle
  parameters for circles/spheres/tori, arc length for lines).
- Constructors validate and raise `ValueError`; nothing else raises.
- All objects are immutable and safe to share across threads.
