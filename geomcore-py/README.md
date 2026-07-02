# geomcore (Python bindings)

Python bindings for [geomcore](https://github.com/GabrielJMS/geomcore), a
pure, standalone geometric kernel for CAD-grade curves and surfaces —
parametric evaluation, rigid transformations, and analytic curve-on-surface
parametrization, implemented in Rust.

```python
from geomcore import Point3, Vector3
from geomcore.curves import Circle3D
import math

circle = Circle3D.new(Point3.origin(), Vector3.z(), 2.0)
point = circle.eval_point(math.pi / 4)
```

## Install

```sh
pip install geomcore
```

Pre-built wheels are published for Linux (x86_64, aarch64), macOS, and
Windows, for Python 3.9+ (abi3 — one wheel covers all versions).

## What's inside

- Parametric evaluation for lines, circles, ellipses, parabolas, hyperbolas,
  and B-spline curves (rational and periodic), plus planes, cylinders, cones,
  spheres, tori, and B-spline surfaces.
- Bulk evaluation (`eval_points`) with a single native call per batch.
- Rigid transformations via a single `Transform` type.
- Analytic curve-on-surface parametrization (`parametrize_on`) for lines and
  circles on the five elementary surfaces.

All numeric results are validated against golden fixtures at CAD-kernel-grade
tolerances (1e-7). See the [repository](https://github.com/GabrielJMS/geomcore)
for full documentation, the [roadmap](https://github.com/GabrielJMS/geomcore/blob/main/ROADMAP.md),
and the Rust crate.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
