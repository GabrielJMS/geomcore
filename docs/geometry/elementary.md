# Elementary curves & surfaces

The ten geometries a kernel knows in closed form: a line, four conics,
and five surfaces. Each is a short formula placed in a frame — a
**canonical shape** plus a **placement** — with a domain, an inverse
map, and known defects at the seams. This page is the reference; the
full story, with measured checks against a production kernel, is
[The mathematics behind geometry](https://avenirgeometry.com/articles/elementary-geometry/),
from which the formulas and figures here are adapted.

## Foundations

**Point, vector, direction** are three separate types. A point has
position and nothing else; a vector has length and orientation but no
position, so translations do not apply to it; a direction is a vector
constrained to length 1. `geomcore` enforces this in its type system
(`Point3D`, `Vector3D`), because mixing them produces bugs that look
like geometry errors — the classic one is translating a normal.

**Canonical shape plus placement.** Nothing is stored as a general
shape in space. A circle is a radius plus an origin and a frame; every
geometry below follows the same division. Placements come in three
sizes: a point and one direction (an axis — lines, surfaces of
revolution), a point and two directions (a plane frame — the conics are
planar curves), a point and three directions (a full frame — the
surfaces). Frames are right-handed: flip the third direction and every
normal flips with it.

**A curve is a map** $C: I \subset \mathbb{R} \to \mathbb{R}^3$ **from an
interval; a surface is a map** $S: D \subset \mathbb{R}^2 \to
\mathbb{R}^3$ **from a rectangle.** The domain is part of the geometry:
the circle formula on a quarter turn is an arc, and a closed domain
$[0, 2\pi]$ whose ends evaluate to the same point is what *periodic*
means. Forward evaluation is cheap; the way back — point to parameter —
is a separate algorithm, closed-form for all ten below, which is
largely why they are kept as special cases.

## The five curves

### Line

The only elementary curve with no frame: a point and a direction, and
the parameter is arc length — equal steps in $u$ are equal steps in
space.

=== "Python"

    ```python
    from geomcore import Point3D, Vector3D
    from geomcore.curves import Line3D

    line = Line3D(Point3D.origin(), Vector3D.z())
    p = line.eval_point(2.0)  # parameter -> 3D point
    assert (p.x, p.y, p.z) == (0.0, 0.0, 2.0)
    assert line.parameter_of(p) == 2.0  # 3D point -> parameter
    ```

=== "Rust"

    ```rust
    use geomcore::{Point3D, Vector3D};
    use geomcore::curves::Line3D;

    let line = Line3D::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    let p = line.eval_point(2.0); // parameter -> 3D point
    assert_eq!(p, Point3D::new(0.0, 0.0, 2.0));
    assert_eq!(line.parameter_of(p), 2.0); // 3D point -> parameter
    ```

$$
C(u) = O + u\,\vec{D}, \qquad u \in (-\infty, +\infty)
$$

The way back: $u = (P - O) \cdot \vec{D}$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/06-a-straight-line-with-equally-spaced.svg" alt="A line with equally spaced parameter marks">
<figcaption>A line. u is the signed distance from the origin along the direction.</figcaption>
</figure>

### Circle

The parameter is the angle from X, and arc length is $R\,u$ — the only
conic whose parameter is proportional to distance travelled.

=== "Python"

    ```python
    from geomcore.curves import Circle3D

    circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
    assert abs(circle.circumference() - 4.0 * 3.141592653589793) < 1e-12
    p = circle.eval_point(1.0)  # parameter -> 3D point
    assert abs(circle.parameter_of(p) - 1.0) < 1e-12  # 3D point -> parameter
    ```

=== "Rust"

    ```rust
    use geomcore::curves::Circle3D;

    let circle = Circle3D::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let p = circle.eval_point(1.0); // parameter -> 3D point
    assert!((circle.parameter_of(p) - 1.0).abs() < 1e-12); // 3D point -> parameter
    ```

$$
C(u) = O + R\cos u\,\vec{X} + R\sin u\,\vec{Y}, \qquad u \in [0, 2\pi]
$$

The way back: $u = \mathrm{atan2}(y, x)$, with $x, y$ the point
resolved in the placement frame. Periodic: $C(u + 2\pi) = C(u)$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/07-a-circle-with-a-point-resolved.svg" alt="A circle with a point resolved into its coordinates">
<figcaption>A circle. Equal steps in u are equal steps along the curve.</figcaption>
</figure>

### Ellipse

Reads like the circle with two radii, but the parameter is the
**eccentric** angle — taken on the circle of radius $a$, projected onto
the curve — not the polar angle from the centre. It is not arc length
either: speed runs from $b$ to $a$, so uniform steps bunch at the ends
of the major axis.

=== "Python"

    ```python
    from geomcore.curves import Ellipse3D

    ellipse = Ellipse3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 3.0, 1.0)
    p = ellipse.eval_point(1.0)  # parameter -> 3D point
    assert abs(ellipse.parameter_of(p) - 1.0) < 1e-12  # 3D point -> parameter
    ```

=== "Rust"

    ```rust
    use geomcore::curves::Ellipse3D;

    let ellipse = Ellipse3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 3.0, 1.0).unwrap();
    let p = ellipse.eval_point(1.0); // parameter -> 3D point
    assert!((ellipse.parameter_of(p) - 1.0).abs() < 1e-12); // 3D point -> parameter
    ```

$$
C(u) = O + a\cos u\,\vec{X} + b\sin u\,\vec{Y}, \qquad u \in [0, 2\pi]
$$

The way back: $u = \mathrm{atan2}(y/b,\; x/a)$ — divide out the radii
and the ellipse becomes the circle whose angle is $u$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/08-an-ellipse-with-a-point-resolved.svg" alt="An ellipse with a point resolved into its coordinates">
<figcaption>An ellipse. u is the eccentric angle, not the polar angle.</figcaption>
</figure>

### Hyperbola

$\cosh$ is never below 1, so this map reaches **one branch** only; the
mirror branch is a second curve no value of $u$ reaches.

=== "Python"

    ```python
    from geomcore.curves import Hyperbola3D

    hyp = Hyperbola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 2.0, 1.0)
    p = hyp.eval_point(1.0)  # parameter -> 3D point
    assert abs(hyp.parameter_of(p) - 1.0) < 1e-12  # 3D point -> parameter
    ```

=== "Rust"

    ```rust
    use geomcore::curves::Hyperbola3D;

    let hyp = Hyperbola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 2.0, 1.0).unwrap();
    let p = hyp.eval_point(1.0); // parameter -> 3D point
    assert!((hyp.parameter_of(p) - 1.0).abs() < 1e-12); // 3D point -> parameter
    ```

$$
C(u) = O + a\cosh u\,\vec{X} + b\sinh u\,\vec{Y}, \qquad u \in (-\infty, +\infty)
$$

The way back: $u = \mathrm{arsinh}(y/b)$. Implicit form:
$\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/09-one-branch-of-a-hyperbola-with.svg" alt="One branch of a hyperbola">
<figcaption>A hyperbola. One branch only — the opposite branch is a second curve.</figcaption>
</figure>

### Parabola

The transpose of the schoolbook $y = x^2$: the parameter runs along
$Y$, the curve opens along $X$. $f$ is the focal distance — focus at
$(f, 0)$, directrix $x = -f$.

=== "Python"

    ```python
    from geomcore.curves import Parabola3D

    parab = Parabola3D(Point3D.origin(), Vector3D.z(), Vector3D.x(), 1.5)
    p = parab.eval_point(2.0)  # parameter -> 3D point
    assert abs(parab.parameter_of(p) - 2.0) < 1e-12  # 3D point -> parameter
    ```

=== "Rust"

    ```rust
    use geomcore::curves::Parabola3D;

    let parab = Parabola3D::new(Point3D::ORIGIN, Vector3D::Z, Vector3D::X, 1.5).unwrap();
    let p = parab.eval_point(2.0); // parameter -> 3D point
    assert!((parab.parameter_of(p) - 2.0).abs() < 1e-12); // 3D point -> parameter
    ```

$$
C(u) = O + \frac{u^2}{4f}\,\vec{X} + u\,\vec{Y}, \qquad u \in (-\infty, +\infty)
$$

The way back: $u = y$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/10-a-parabola-with-focus-directrix-and.svg" alt="A parabola with focus and directrix">
<figcaption>A parabola. The parameter is the offset along Y; the axis is X.</figcaption>
</figure>

## The five surfaces

Four are surfaces of revolution — a profile curve turned about an
axis — so the two horizontal components are always some radius times
$\cos u$ and $\sin u$. They differ only in that radius and in the third
component. With $\rho = \sqrt{x^2 + y^2}$ the distance from the axis,
four of the five inverses start with the same $u = \mathrm{atan2}(y, x)$.

### Plane

Both parameters are lengths; the only surface with no angles, no
periodicity, no degenerate point.

=== "Python"

    ```python
    from geomcore.surfaces import Plane

    plane = Plane(Point3D.origin(), Vector3D.z())
    p = plane.eval_point(1.0, 2.0)  # parameters -> 3D point
    assert (p.x, p.y, p.z) == (1.0, 2.0, 0.0)
    assert plane.parameters_of(p) == (1.0, 2.0)  # 3D point -> parameters
    ```

=== "Rust"

    ```rust
    use geomcore::surfaces::Plane;

    let plane = Plane::new(Point3D::ORIGIN, Vector3D::Z).unwrap();
    let p = plane.eval_point(1.0, 2.0); // parameters -> 3D point
    assert_eq!(p, Point3D::new(1.0, 2.0, 0.0));
    assert_eq!(plane.parameters_of(p), (1.0, 2.0)); // 3D point -> parameters
    ```

$$
S(u, v) = O + u\,\vec{X} + v\,\vec{Y}, \qquad (u, v) \in (-\infty, +\infty)^2
$$

The way back: $u = x,\; v = y$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/11-a-plane-with-a-point-resolved.svg" alt="A plane with a point resolved into its coordinates">
<figcaption>A plane. u and v are lengths along the frame directions.</figcaption>
</figure>

### Cylinder

$u$ is an angle, $v$ a length — the only surface whose parameters are
not the same kind of quantity.

=== "Python"

    ```python
    from geomcore.surfaces import Cylinder

    cyl = Cylinder(Point3D.origin(), Vector3D.z(), 2.0)
    p = cyl.eval_point(1.0, 2.0)  # parameters -> 3D point
    u, v = cyl.parameters_of(p)  # 3D point -> parameters
    assert abs(u - 1.0) < 1e-12 and abs(v - 2.0) < 1e-12
    ```

=== "Rust"

    ```rust
    use geomcore::surfaces::Cylinder;

    let cyl = Cylinder::new(Point3D::ORIGIN, Vector3D::Z, 2.0).unwrap();
    let p = cyl.eval_point(1.0, 2.0); // parameters -> 3D point
    let (u, v) = cyl.parameters_of(p); // 3D point -> parameters
    assert!((u - 1.0).abs() < 1e-12 && (v - 2.0).abs() < 1e-12);
    ```

$$
S(u, v) = O + R\cos u\,\vec{X} + R\sin u\,\vec{Y} + v\,\vec{Z}, \qquad (u, v) \in [0, 2\pi] \times (-\infty, +\infty)
$$

The way back: $u = \mathrm{atan2}(y, x),\; v = z$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/12-a-cylinder-with-a-point-resolved.svg" alt="A cylinder with a point resolved into its coordinates">
<figcaption>A cylinder. u goes round the axis, v along it.</figcaption>
</figure>

### Cone

The cylinder's formula with a radius that grows along the axis:
$\rho = R + v\sin\alpha$. `geomcore` measures $v$ **along the ruling**,
so $\lVert \partial S/\partial v \rVert = 1$ exactly — a step of 1 in
$v$ is a step of 1 across the surface. The alternative height
convention ($v = z$, $\rho = R + v\tan\alpha$) shares the cylinder's
parameter plane instead; the article compares both.

=== "Python"

    ```python
    from geomcore.surfaces import Cone
    import math

    cone = Cone(Point3D.origin(), Vector3D.z(), math.radians(28), 4.0)
    p = cone.eval_point(1.0, 2.0)  # parameters -> 3D point
    u, v = cone.parameters_of(p)  # 3D point -> parameters
    assert abs(u - 1.0) < 1e-9 and abs(v - 2.0) < 1e-9
    ```

=== "Rust"

    ```rust
    use geomcore::surfaces::Cone;

    let cone = Cone::new(Point3D::ORIGIN, Vector3D::Z, 0.4887, 4.0).unwrap();
    let p = cone.eval_point(1.0, 2.0); // parameters -> 3D point
    let (u, v) = cone.parameters_of(p); // 3D point -> parameters
    assert!((u - 1.0).abs() < 1e-9 && (v - 2.0).abs() < 1e-9);
    ```

$$
S(u, v) = O + \rho\,(\cos u\,\vec{X} + \sin u\,\vec{Y}) + v\cos\alpha\,\vec{Z}, \qquad (u, v) \in [0, 2\pi] \times (-\infty, +\infty)
$$

The way back: $u = \mathrm{atan2}(y, x),\; v = \sin\alpha\,(\rho - R) + \cos\alpha\,z$.

At $v_{\mathrm{apex}} = -R / \sin\alpha$ the radius reaches zero: a
whole edge of the rectangle is one point, the normal vanishes, and past
it $\rho$ goes negative into the opposite nappe. The frame sits on the
reference circle ($v = 0$), not at the apex — the only point with no
normal is a bad datum, it is often hundreds of radii from the part, and
anchoring there would break the cone→cylinder limit as $\alpha \to 0$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/13-a-cone-with-a-point-resolved.svg" alt="A cone with a point resolved into its coordinates">
<figcaption>A cone. The radius grows along the ruling and reaches zero at the apex.</figcaption>
</figure>

### Sphere

Both parameters are angles: longitude $u$, latitude $v$ from the
equator. Both ends of $v$ are degenerate — at the poles the whole
circle of longitudes collapses onto the axis.

=== "Python"

    ```python
    from geomcore.surfaces import Sphere

    sphere = Sphere(Point3D.origin(), 2.0)
    assert abs(sphere.volume() - 4.0 / 3.0 * 3.141592653589793 * 8.0) < 1e-9
    p = sphere.eval_point(0.5, 0.3)  # parameters -> 3D point
    u, v = sphere.parameters_of(p)  # 3D point -> parameters
    assert abs(u - 0.5) < 1e-12 and abs(v - 0.3) < 1e-12
    ```

=== "Rust"

    ```rust
    use geomcore::surfaces::Sphere;

    let sphere = Sphere::new(Point3D::ORIGIN, 2.0).unwrap();
    let p = sphere.eval_point(0.5, 0.3); // parameters -> 3D point
    let (u, v) = sphere.parameters_of(p); // 3D point -> parameters
    assert!((u - 0.5).abs() < 1e-12 && (v - 0.3).abs() < 1e-12);
    ```

$$
S(u, v) = O + \rho\,(\cos u\,\vec{X} + \sin u\,\vec{Y}) + R\sin v\,\vec{Z}, \quad \rho = R\cos v, \quad (u, v) \in [0, 2\pi] \times [-\tfrac{\pi}{2}, \tfrac{\pi}{2}]
$$

The way back: $u = \mathrm{atan2}(y, x),\; v = \mathrm{atan2}(z, \rho)$.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/15-a-sphere-with-a-point-resolved.svg" alt="A sphere with a point resolved into its coordinates">
<figcaption>A sphere. The red dots are the poles, where a whole circle of u is one point.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/elem-sphere.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="A sphere, a point P(u, v) and the two iso curves through it"></iframe>

### Torus

The cone's shape with $v$ turning the radius instead of growing it:
$\rho = R + r\cos v$. Both parameters close, so the domain — both
pairs of edges glued — is itself a torus, the most faithful
rectangle of the five.

=== "Python"

    ```python
    from geomcore.surfaces import Torus

    torus = Torus(Point3D.origin(), Vector3D.z(), 5.0, 1.5)
    p = torus.eval_point(0.5, 1.0)  # parameters -> 3D point
    u, v = torus.parameters_of(p)  # 3D point -> parameters
    assert abs(u - 0.5) < 1e-12 and abs(v - 1.0) < 1e-12
    ```

=== "Rust"

    ```rust
    use geomcore::surfaces::Torus;

    let torus = Torus::new(Point3D::ORIGIN, Vector3D::Z, 5.0, 1.5).unwrap();
    let p = torus.eval_point(0.5, 1.0); // parameters -> 3D point
    let (u, v) = torus.parameters_of(p); // 3D point -> parameters
    assert!((u - 0.5).abs() < 1e-12 && (v - 1.0).abs() < 1e-12);
    ```

$$
S(u, v) = O + \rho\,(\cos u\,\vec{X} + \sin u\,\vec{Y}) + r\sin v\,\vec{Z}, \quad \rho = R + r\cos v, \quad (u, v) \in [0, 2\pi]^2
$$

The way back: $u = \mathrm{atan2}(y, x),\; v = \mathrm{atan2}(z, \rho - R)$.
If $r > R$, $\rho$ has two zeros and the spindle self-intersects —
the same five numbers describe a ring or a spindle, and only $R > r$
says which.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/16-a-torus-with-a-point-resolved.svg" alt="A torus with a point resolved into its coordinates">
<figcaption>A torus. v rounds the tube, changing the radius that u sweeps about the axis.</figcaption>
</figure>

<iframe src="https://geomcore.github.io/geomcore-viewer/?scene=https://geomcore.github.io/geomcore/assets/scenes/elem-torus.json" width="100%" height="480" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="A torus, a point P(u, v) and the two iso curves through it"></iframe>

## What the parameters measure

The derivative magnitude $\lVert \partial S/\partial u \rVert$ is the
unit conversion of a parameter: where it is 1 the parameter is arc
length; where it is a radius the parameter is an angle.

| Surface | $\lVert \partial S/\partial u \rVert$ | $u$ is | $\lVert \partial S/\partial v \rVert$ | $v$ is |
|---|---|---|---|---|
| Plane | 1 | a length | 1 | a length |
| Cylinder | $R$ | an angle | 1 | a length |
| Cone | $\rho = R + v\sin\alpha$ | an angle | 1 | a length along the ruling |
| Sphere | $R\cos v$ | an angle | $R$ | an angle |
| Torus | $\rho = R + r\cos v$ | an angle | $r$ | an angle |

Where the $u$-speed depends on $v$, a uniform $(u, v)$ grid is not
uniform on the surface — triangles crowd at the poles. Every mesher
corrects for exactly this table.

## Seams and degenerate edges

Three things go wrong, and all the special cases in surface code live
at these three places:

- **Seam** — two parameters, one point. On a cylinder, $u = 0$ and $u =
  2\pi$ evaluate $9.8\times10^{-16}$ apart. The surface is fine; the
  inverse must *choose*, and the geometry does not determine the
  choice. `geomcore` reports seam parameters inside `bounds()` and
  normalises hits there to 0.
- **Degenerate edge** — a whole edge that is one point: the cone apex,
  the sphere poles. $\lVert \partial S/\partial u \rVert$ measures
  $1.5\times10^{-15}$ at a pole, the normal cross product is the zero
  vector, and there is no normal to ask for.
- **Self-intersection** — the spindle torus above.

<figure class="geom-fig">
<img src="../../assets/figures/elementary-geometry/17-the-parameter-rectangles-of-the-plane.svg" alt="The five parameter rectangles with seams and degenerate edges">
<figcaption>The five domains. Dashed border: unbounded. Shaded edge: seam, two edges one curve. Red edge: a whole edge that is a single point.</figcaption>
</figure>

## Why ten special cases

Everything above can be written as NURBS — conics exactly, as rational
quadratics — so a kernel could keep one representation. Keeping the ten
buys four things: exactness that survives editing (a radius is stored,
not derived), closed-form inversion (`atan2` and a dot product instead
of Newton), closed-form intersections (plane∩cylinder is a conic, not a
march), and recognisability (STEP names these entities; a hole stays a
cylinder through a round trip). The cost is the pairing table in the
intersector — see [Intersection](../guides/intersection.md).
