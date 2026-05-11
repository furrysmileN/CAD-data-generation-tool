from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import trimesh

try:
    import cadquery as cq
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cq = None


def iter_step_files(input_dir: str | Path, recursive: bool = False, filename_pattern: str = "*") -> Iterator[Path]:
    root = Path(input_dir)
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".step", ".stp"):
            continue
        if not fnmatch.fnmatch(path.name, filename_pattern):
            continue
        if path.stat().st_size == 0:
            continue
        yield path


def sample_id_from_relative_path(relative_step_path: str) -> str:
    rel = Path(relative_step_path).with_suffix("")
    safe_parts = []
    for part in rel.parts:
        if part in ("", "."):
            continue
        safe_parts.append("".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in part))
    return "__".join(safe_parts)


def _load_step_with_cadquery(step_path: Path, triangle_face_tol: float, angle_tol_rads: float) -> trimesh.Trimesh:
    if cq is None:
        raise ImportError("cadquery is required to read STEP files")
    workplane = cq.importers.importStep(str(step_path))
    if hasattr(workplane, "vals"):
        shapes = list(workplane.vals())
    elif hasattr(workplane, "val"):
        shapes = [workplane.val()]
    else:
        shapes = [workplane]

    all_verts = []
    all_tris = []
    vert_offset = 0
    for shape in shapes:
        vertices, faces = shape.tessellate(triangle_face_tol, angularTolerance=angle_tol_rads)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        verts = np.asarray([(v.x, v.y, v.z) for v in vertices], dtype=np.float32)
        tris = np.asarray(faces, dtype=np.int32) + vert_offset
        all_verts.append(verts)
        all_tris.append(tris)
        vert_offset += len(verts)
    if not all_verts or not all_tris:
        raise ValueError(f"empty mesh after cadquery tessellation: {step_path}")
    return trimesh.Trimesh(vertices=np.concatenate(all_verts, axis=0), faces=np.concatenate(all_tris, axis=0), process=False)


def load_step_as_mesh(
    step_path: str | Path,
    triangle_face_tol: float = 0.01,
    angle_tol_rads: float = 0.1,
) -> Tuple[trimesh.Trimesh, Optional[np.ndarray], str]:
    mesh = _load_step_with_cadquery(Path(step_path), triangle_face_tol, angle_tol_rads)
    return mesh, None, "cadquery"
