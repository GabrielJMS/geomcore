# Curve on surface

A curve that lies on a surface has an exact 2D image in the surface's
parameter space — the *pcurve*. `parametrize_on` computes it in closed
form where one exists (lines and circles on the five elementary
surfaces); `parametrize_numeric` samples anywhere else. Adapted from
[Parametric domain](https://avenirgeometry.com/articles/parametric-domain/).

## Why the rectangle matters

Kernels work in $(u, v)$ because the rectangle is flat: offsetting,
trimming and marching are all easier there than in space. But a uniform
grid in the rectangle is not uniform on the surface — the first
fundamental form stretches it — so every downstream algorithm must know
what the parameters measure (see [Elementary](../geometry/elementary.md#what-the-parameters-measure)).

<figure class="geom-fig">
<img src="../../assets/figures/parametric-domain/01-a-uniform-grid-drawn-on-the.svg" alt="A uniform grid drawn on the surface">
<figcaption>A uniform grid in (u, v) is not uniform on the surface: the metric stretches it.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/param-grid.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="A uniform (u, v) grid on a sphere, one cell at the equator and one against the pole"></iframe>

## Analytic pcurves

A circle coaxial with a cylinder is a horizontal line in $(u, v)$; a
line in a plane is a line. The result is a `Line2D`/`Circle2D` (or a
`Curve2D` run-time value) that evaluates through the surface back onto
the 3D curve within `tol.confusion` — that check is what "lies on"
means, and anything farther off is `CurveNotOnSurface`.

=== "Python"

    ```python
    pc = circle.parametrize_on(cylinder)   # Line2D in (u, v)
    ```

=== "Rust"

    ```rust
    let pc = circle.parametrize_on(cylinder, tol).unwrap();
    ```

A few images across the surfaces:

<figure class="geom-fig">
<img src="../../assets/figures/parametric-domain/02-three-curves-on-a-cylinder.svg" alt="Curves on a cylinder">
<figcaption>On a cylinder: coaxial circles are horizontal lines, generators vertical ones.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/param-cylinder.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Three curves on a cylinder and their images in (u, v)"></iframe>

<figure class="geom-fig">
<img src="../../assets/figures/parametric-domain/04-three-curves-on-a-cone.svg" alt="Curves on a cone">
<figcaption>On a cone the same construction winds with the growing radius.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/param-cone.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Three curves on a cone and their images in (u, v)"></iframe>

<figure class="geom-fig">
<img src="../../assets/figures/parametric-domain/05-three-curves-on-a-sphere.svg" alt="Curves on a sphere">
<figcaption>On a sphere: parallels are horizontal, meridians vertical — and both pinch at the poles.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/param-sphere.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Three curves on a sphere and their images in (u, v)"></iframe>

<figure class="geom-fig">
<img src="../../assets/figures/parametric-domain/06-three-curves-on-a-torus.svg" alt="Curves on a torus">
<figcaption>On a torus both families close: the rectangle glues into the surface itself.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/param-torus.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="Three curves on a torus and their images in (u, v)"></iframe>

## Numeric fallback

Where no closed form exists, `parametrize_numeric` inverts uniform
curve samples through surface projection and is validated against
every analytic pcurve.

=== "Python"

    ```python
    from geomcore import parametrize_numeric

    uv = parametrize_numeric(curve, surface, sample_count=64)   # [(u, v), ...]
    ```

=== "Rust"

    ```rust
    use geomcore::parametrize_numeric;

    let uv = parametrize_numeric(&curve, &surface, 64, tol);
    ```

## Viewing pcurves

The viewer draws a pcurve twice: on the surface in 3D and in its own
$(u, v)$ pane. Pass the 2D curve with the already-added surface's
handle.

=== "Python"

    ```python
    s = view.add(cylinder, label="S")
    view.add(circle, label="C")
    view.pcurve(circle.parametrize_on(cylinder), s, label="C in (u, v)")
    ```

=== "Rust"

    ```rust
    use geomcore_viewer::{Geom, Opts};

    let s = view.add(Geom::from(&cylinder), Opts::new().label("S"))?;
    view.add(&circle, Opts::new().label("C"))?;
    view.pcurve(&circle.parametrize_on(cylinder, tol)?, &s, Opts::new().label("C in (u, v)"))?;
    ```
