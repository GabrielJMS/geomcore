# Viewer

Live [geomcore-viewer](https://github.com/geomcore/geomcore-viewer) instance
below — Babylon.js 3D pane plus a PixiJS 2D pane for parameter-space curves.
The demo scene shows an ellipse, a circle, a hyperbola and a line, with the points where they meet.

<iframe src="https://geomcore.github.io/geomcore-viewer/" width="100%" height="640" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="geomcore-viewer live demo"></iframe>

## Install

=== "Python"

    ```sh
    pip install geomcore geomcore-viewer
    ```

    geomcore-viewer 0.2 has no dependencies and ships the built viewer
    in the wheel. It reads geomcore objects through their public
    getters rather than pinning a geomcore version, and works with
    geomcore 0.33 to 0.35.

=== "Rust"

    ```toml
    [dependencies]
    geomcore = "0.32"
    geomcore-viewer = { version = "0.2", features = ["geomcore"] }
    ```

    The `geomcore` feature of geomcore-viewer 0.2 is built against
    geomcore 0.32. Paired with a later geomcore, Cargo pulls in both
    versions and the viewer does not accept the newer types, so keep
    geomcore at 0.32 in code that uses the viewer until a viewer
    release supports a later one. The Rust client never starts the server:
    run `geomcore-viewer serve` from the Python package once.

## From your own code

The embedded page above is static. From a script, a debugger or a notebook, `view.add(...)` sends geomcore objects to a local viewer tab that updates live:

=== "Python"

    ```python
    from geomcore import Frame3D, Point3D, Vector3D
    from geomcore.curves import Circle3D, Ellipse3D
    import geomcore_viewer as gv

    view = gv.Viewer("Ellipse ∩ circle")          # starts the local viewer and opens a tab
    E = Ellipse3D.from_frame(Frame3D.world(), 3.0, 2.0)
    C = Circle3D(Point3D(2, 1, 0), Vector3D.z(), 1.5)
    view.add(E, label="E")
    view.add(C, label="C")
    view.add(C.intersect(E), label="P")           # intersection points P₁, P₂
    ```

=== "Rust"

    ```rust
    use geomcore::{Circle3D, Cylinder, Point3D, Tolerance, Vector3D};
    use geomcore_viewer::{Geom, Opts, Viewer};

    let mut view = Viewer::new("A circle on a cylinder")?;
    let cylinder = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0)?;
    let ring = Circle3D::new(Point3D::new(0.0, 0.0, 1.0), Vector3D::Z, 2.0)?;
    let s = view.add(Geom::from(&cylinder).window(None, Some([-2.0, 2.0])), Opts::new().label("S"))?;
    view.add(&ring, Opts::new().label("C"))?;
    view.pcurve(&ring.parametrize_on(cylinder, Tolerance::DEFAULT)?, &s, Opts::new().label("C in (u, v)"))?;
    ```

Protocol details: [protocol/SCENE.md](https://github.com/geomcore/geomcore-viewer/blob/main/protocol/SCENE.md).
