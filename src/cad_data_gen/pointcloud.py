from __future__ import annotations

import numpy as np
import trimesh


def mesh_to_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample a fixed-size surface point cloud from a mesh."""
    points, _ = trimesh.sample.sample_surface(mesh, int(num_points))
    return np.asarray(points, dtype=np.float32)
