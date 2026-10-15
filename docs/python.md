# Python

```sh
pip install geomcore
```

Pre-built wheels are published for Linux (x86_64, aarch64), macOS and
Windows, for Python 3.9+ (abi3: one wheel covers every version). The
package ships type stubs (PEP 561), so IDEs and type checkers see the full
API.

=== "Python"

    ```python
    from geomcore import Point3D, Vector3D
    from geomcore.curves import Circle3D
    import math

    circle = Circle3D(Point3D.origin(), Vector3D.unit_z(), 2.0)
    point = circle.eval_point(math.pi / 4)
    ```

=== "Rust"

    ```rust
    use geomcore::{Point3D, Vector3D};
    use geomcore::curves::Circle3D;

    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let point = circle.eval_point(std::f64::consts::FRAC_PI_4);
    ```

Reprs print floats as Python does and show coordinates below `1e-15` as
`0.0`, so rounding noise such as `cos(pi / 2)` does not clutter them. A repr is
therefore for reading, not an exact round trip of such tiny coordinates.

A Python API reference generated from docstrings, and tutorials, are on
the roadmap. Until then, the Rust items on
[docs.rs](https://docs.rs/geomcore) mirror the Python surface one-to-one.
