# Python

```sh
pip install geomcore
```

Not yet published to PyPI — coming with the first release
(see [releasing](https://github.com/GabrielJMS/geomcore/blob/main/docs/releasing.md)).

```python
from geomcore import Point3D, Vector3D
from geomcore.curves import Circle3D
import math

circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
point = circle.eval_point(math.pi / 4)
```

Full API reference with tutorials will be generated from docstrings once
type stubs land. Until then, the Rust items on
[docs.rs](https://docs.rs/geomcore) mirror the Python surface one-to-one.
