# Viewer

Live [geomcore-viewer](https://github.com/GabrielJMS/geomcore-viewer) instance
below — Babylon.js 3D pane plus a PixiJS 2D pane for parameter-space curves.
The demo scene shows a circle coaxial with a cylinder and its 2D p-curve image.

<iframe src="https://gabrieljms.github.io/geomcore-viewer/" width="100%" height="640" style="border: 1px solid #333; border-radius: 8px;" allow="fullscreen" title="geomcore-viewer live demo"></iframe>

## From your own code

The embedded page above is static. For incremental, live-updating plots,
serve the viewer locally and push entities from Rust or Python:

```python
from geomcore_viewer import Viewer, sample_curve
import math

viewer = Viewer()
viewer.add_curve3d("circle", sample_curve(circle.eval_point, 0.0, 2 * math.pi))
server = viewer.serve("path/to/frontend/dist")  # opens the browser, updates live
```

```rust
use geomcore_viewer::{Client, Entity};

let client = Client::local();
client.add(&Entity::curve3d("circle", "#ff5533", points))?;
```

Protocol details: [protocol/SCENE.md](https://github.com/GabrielJMS/geomcore-viewer/blob/main/protocol/SCENE.md).
