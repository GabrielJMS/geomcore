# Python

```sh
pip install geomcore
```

Pre-built wheels are published for Linux (x86_64, aarch64), macOS and
Windows, for Python 3.9+ (abi3: one wheel covers every version). The
package ships type stubs (PEP 561), so IDEs and type checkers see the full
API.

```python
from geomcore import Point3D, Vector3D
from geomcore.curves import Circle3D
import math

circle = Circle3D(Point3D.origin(), Vector3D.z(), 2.0)
point = circle.eval_point(math.pi / 4)
```

A Python API reference generated from docstrings, and tutorials, are on
the roadmap. Until then, the Rust items on
[docs.rs](https://docs.rs/geomcore) mirror the Python surface one-to-one.
