# Discretization fills caller-owned buffers

A discretization is computed in two steps: one call returns how many points the curve needs for the given criteria, and a second writes exactly that many into slices the caller provides. A convenience call that allocates its own result wraps the two. geomcore never decides where the output lives or which thread computes it, so a kernel can reuse one buffer per thread across all its edges, or carve each edge's slice out of one large array, without copying. Counts depend only on the geometry and the criteria, so the result is the same however the caller schedules the work.

## Considered options

- **Returning a new result per call only**: rejected. On 100,000 model-like edges (12 to 34 points each), a buffer reused across edges was faster than a fresh result per edge, and reused per-thread buffers scaled 6.6 times on 16 threads.
- **Counting every edge first and filling one large freshly allocated array**: not ruled out, the caller can still do it, but not the recommended use. At realistic point counts it was twice as slow as reused buffers, the cost being in first-touching the new memory and the per-edge bookkeeping.
- **Parallelism inside geomcore**: rejected. Thread pools belong to the caller; geomcore stays dependency-free and only guarantees that calls are reentrant and do not allocate when given buffers.
