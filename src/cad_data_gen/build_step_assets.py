#!/usr/bin/env python3
"""
Build point-cloud and multi-view image assets from STEP files.

This module keeps the helper API that `build_occlusion_assets.py` depends on:
STEP loading, unit-cube normalization, deterministic camera fronts, a small CPU
renderer, and the Blender wrapper used for higher quality renders.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

try:
    import cadquery as cq
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cq = None

DEFAULT_VISUALIZATION_ROOT = Path("/root/autodl-tmp/visualization/visualization-tar")


@dataclass(frozen=True)
class ProcessArgs:
    input_dir: Path
    output_dir: Path
    recursive: bool
    filename_pattern: str
    num_points: int
    num_views: int
    img_size: int
    camera_distance: float
    triangle_face_tol: float
    angle_tol_rads: float
    normalize: str
    render_backend: str
    blender_bin: str
    blender_script: Optional[Path]
    blender_engine: str
    blender_samples: int
    blender_style: str
    visualization_root: Optional[str]
    foreground_occluder: bool
    foreground_occluder_seed: int
    foreground_occluder_color: tuple[int, int, int]
    foreground_occluder_size_min: float
    foreground_occluder_size_max: float
    foreground_occluder_depth: float
    skip_existing: bool


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


def sample_id_from_relative_path(relative_step_path: str | Path) -> str:
    rel = Path(relative_step_path).with_suffix("")
    safe_parts: list[str] = []
    for part in rel.parts:
        if part in ("", "."):
            continue
        safe_parts.append("".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in part))
    return "__".join(safe_parts)


def _load_step_with_cadquery(
    step_path: Path,
    triangle_face_tol: float,
    angle_tol_rads: float,
) -> tuple[trimesh.Trimesh, Optional[np.ndarray]]:
    if cq is None:
        raise ImportError("cadquery is required to read STEP files")
    workplane = cq.importers.importStep(str(step_path))
    if hasattr(workplane, "vals"):
        shapes = list(workplane.vals())
    elif hasattr(workplane, "val"):
        shapes = [workplane.val()]
    else:
        shapes = [workplane]

    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    tri_mapping: list[int] = []
    vert_offset = 0
    for shape_index, shape in enumerate(shapes):
        vertices, faces = shape.tessellate(triangle_face_tol, angularTolerance=angle_tol_rads)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        verts = np.asarray([(v.x, v.y, v.z) for v in vertices], dtype=np.float64)
        tris = np.asarray(faces, dtype=np.int64) + vert_offset
        all_verts.append(verts)
        all_tris.append(tris)
        tri_mapping.extend([shape_index] * len(tris))
        vert_offset += len(verts)
    if not all_verts or not all_tris:
        raise ValueError(f"empty mesh after cadquery tessellation: {step_path}")
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(all_verts, axis=0),
        faces=np.concatenate(all_tris, axis=0),
        process=False,
    )
    return mesh, np.asarray(tri_mapping, dtype=np.int32)


def load_step_as_mesh(
    step_path: str | Path,
    triangle_face_tol: float = 0.01,
    angle_tol_rads: float = 0.1,
) -> Tuple[trimesh.Trimesh, Optional[np.ndarray], str]:
    mesh, tri_mapping = _load_step_with_cadquery(Path(step_path), triangle_face_tol, angle_tol_rads)
    return mesh, tri_mapping, "cadquery"


def mesh_to_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    points, _ = trimesh.sample.sample_surface(mesh, int(num_points))
    return np.asarray(points, dtype=np.float32)


def normalize_trimesh_unit_cube(
    mesh: trimesh.Trimesh,
    points: Optional[np.ndarray] = None,
) -> tuple[trimesh.Trimesh, Optional[np.ndarray], np.ndarray, float, np.ndarray]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if not np.isfinite(bounds).all():
        raise ValueError("mesh bounds are not finite")
    extents = bounds[1] - bounds[0]
    scale = 1.0 / max(float(extents.max()), 1e-12)
    raw_center = (bounds[0] + bounds[1]) * 0.5

    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - raw_center) * scale
    centered_points = None
    if points is not None:
        centered_points = (np.asarray(points, dtype=np.float64) - raw_center) * scale
    return mesh, centered_points, bounds, float(scale), raw_center.astype(np.float64)


def get_view_fronts(num_views: int) -> List[List[float]]:
    if num_views == 1:
        return [[1, 1, 1]]
    if num_views == 2:
        return [[1, 1, 1], [-1, -1, -1]]
    if num_views == 4:
        return [[1, 1, 1], [-1, -1, -1], [-1, 1, -1], [1, -1, 1]]
    if num_views == 6:
        return [
            [1, 1, 1],
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, 1],
            [0, 1, 0],
            [0, -1, 0],
        ]

    # Deterministic fallback for arbitrary view counts.
    fronts: list[list[float]] = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_views):
        y = 1.0 - (2.0 * i + 1.0) / float(num_views)
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        fronts.append([float(np.cos(theta) * radius), float(y), float(np.sin(theta) * radius)])
    return fronts


def _view_basis(front: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = np.asarray(front, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("camera front must be non-zero")
    direction = direction / norm
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up, direction)
    if float(np.linalg.norm(right)) < 1e-8:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(up, direction)
    right = right / float(np.linalg.norm(right))
    true_up = np.cross(direction, right)
    true_up = true_up / float(np.linalg.norm(true_up))
    return right, true_up, direction


def _project_vertices(vertices: np.ndarray, front: Sequence[float], img_size: int) -> tuple[np.ndarray, np.ndarray]:
    right, true_up, direction = _view_basis(front)
    verts = np.asarray(vertices, dtype=np.float64)
    xy = np.stack([verts @ right, verts @ true_up], axis=1)
    span = float(np.max(np.ptp(xy, axis=0)))
    if span < 1e-8:
        span = 1.0
    pad = img_size * 0.12
    scale = (img_size - 2.0 * pad) / span
    pixels = np.empty_like(xy)
    pixels[:, 0] = xy[:, 0] * scale + img_size / 2.0
    pixels[:, 1] = img_size / 2.0 - xy[:, 1] * scale
    depth = verts @ direction
    return pixels, depth


def _rasterize_mesh(mesh: trimesh.Trimesh, front: Sequence[float], img_size: int) -> Image.Image:
    pixels, depths = _project_vertices(np.asarray(mesh.vertices), front, img_size)
    z_buffer = np.full((img_size, img_size), -np.inf, dtype=np.float64)
    rgb = np.full((img_size, img_size, 3), 245, dtype=np.uint8)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    _, _, direction = _view_basis(front)
    light = direction / max(float(np.linalg.norm(direction)), 1e-12)

    for face_index, tri in enumerate(np.asarray(mesh.faces, dtype=np.int32)):
        pts = pixels[tri]
        min_xy = np.floor(pts.min(axis=0)).astype(int)
        max_xy = np.ceil(pts.max(axis=0)).astype(int)
        min_x = max(0, int(min_xy[0]))
        min_y = max(0, int(min_xy[1]))
        max_x = min(img_size - 1, int(max_xy[0]))
        max_y = min(img_size - 1, int(max_xy[1]))
        if min_x > max_x or min_y > max_y:
            continue

        p0, p1, p2 = pts
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(float(denom)) < 1e-12:
            continue

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        px = xx + 0.5
        py = yy + 0.5
        w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        z_tri = depths[tri]
        z_pixels = w0 * z_tri[0] + w1 * z_tri[1] + w2 * z_tri[2]
        current = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (z_pixels > current)
        if not update.any():
            continue

        normal = normals[face_index] if face_index < len(normals) else np.asarray([0.0, 0.0, 1.0])
        shade = 0.58 + 0.42 * max(0.0, float(normal @ light))
        color = np.asarray([150, 176, 205], dtype=np.float64) * shade
        color = np.clip(color + 28.0, 0.0, 255.0).astype(np.uint8)
        current[update] = z_pixels[update]
        patch = rgb[min_y : max_y + 1, min_x : max_x + 1]
        patch[update] = color

    return Image.fromarray(rgb, mode="RGB")


def render_views_to_png(
    mesh_centered: trimesh.Trimesh,
    image_dir: Path,
    fronts: List[List[float]],
    img_size: int,
    camera_distance: float = -0.9,
    render_width: int = 512,
    render_height: int = 512,
    render_backend: str = "trimesh",
) -> List[str]:
    del camera_distance, render_width, render_height, render_backend
    image_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for i, front in enumerate(fronts):
        image = _rasterize_mesh(mesh_centered, front, img_size)
        out_name = f"view_{i:03d}.png"
        image.save(image_dir / out_name)
        rel_paths.append(str(Path("images") / image_dir.name / out_name))
    return rel_paths


def _color_to_unit_rgba(color: Sequence[int | float]) -> list[float]:
    values = list(color)
    if len(values) == 3:
        values.append(255)
    return [float(v) / 255.0 if float(v) > 1.0 else float(v) for v in values[:4]]


def resolve_blender_bin(
    blender_bin: str,
    blender_style: str,
    visualization_root: Optional[str],
) -> str:
    if blender_bin != "blender":
        return blender_bin

    root: Optional[Path]
    if visualization_root is not None:
        root = Path(visualization_root).expanduser().resolve()
    elif blender_style == "visualization" and DEFAULT_VISUALIZATION_ROOT.is_dir():
        root = DEFAULT_VISUALIZATION_ROOT
    else:
        root = None

    candidates: list[Path] = []
    if root is not None:
        candidates.extend(
            [
                root / "blender" / "blender",
                root / "blender-3.6.0-linux-x64" / "blender",
                root / "blender-3.6.5-linux-x64" / "blender",
                root / "blender-4.0.0-linux-x64" / "blender",
            ]
        )
        candidates.extend(root.glob("blender*/blender"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return blender_bin


def make_foreground_occluder_config(
    num_views: int,
    seed: int,
    color: Sequence[int | float],
    size_min: float,
    size_max: float,
    depth: float = 0.45,
    shapes: Sequence[str] = ("rectangle", "ellipse", "triangle", "hexagon"),
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    views: list[dict[str, Any]] = []
    for _ in range(num_views):
        width = float(rng.uniform(size_min, size_max))
        height = float(rng.uniform(size_min, size_max))
        max_x = max(0.0, 0.18 - width * 0.25)
        max_y = max(0.0, 0.18 - height * 0.25)
        views.append(
            {
                "offset_xy": [
                    float(rng.uniform(-max_x, max_x)),
                    float(rng.uniform(-max_y, max_y)),
                ],
                "size_xy": [width, height],
                "angle_degrees": float(rng.uniform(-18.0, 18.0)),
                "shape": str(rng.choice(list(shapes))),
                "depth": float(depth),
            }
        )
    return {
        "color": _color_to_unit_rgba([*color, 255]),
        "depth": float(depth),
        "views": views,
        "placement": "screen_random",
    }


def _export_temp_stl(step_path: Path, triangle_face_tol: float = 0.01, angle_tol_rads: float = 0.1) -> str:
    mesh, _, _ = load_step_as_mesh(step_path, triangle_face_tol=triangle_face_tol, angle_tol_rads=angle_tol_rads)
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp.close()
    mesh.export(tmp.name)
    return tmp.name


def render_step_views_with_blender(
    step_path: Path,
    image_dir: Path,
    fronts: List[List[float]],
    img_size: int,
    camera_distance: float,
    raw_center: Sequence[float],
    scale: float,
    blender_bin: str = "blender",
    blender_script: Optional[Path] = None,
    occluder: Optional[dict[str, Any]] = None,
    foreground_occluder: Optional[dict[str, Any]] = None,
    engine: str = "BLENDER_EEVEE_NEXT",
    samples: int = 64,
    render_style: str = "default",
) -> List[str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    script = blender_script or Path(__file__).resolve().with_name("render_step_with_blender.py")
    if not script.is_file():
        raise FileNotFoundError(f"Blender renderer not found: {script}")

    config: dict[str, Any] = {
        "step_path": str(step_path),
        "output_dir": str(image_dir),
        "fronts": fronts,
        "img_size": int(img_size),
        "camera_distance": float(camera_distance),
        "raw_center": [float(v) for v in raw_center],
        "scale": float(scale),
        "engine": engine,
        "samples": int(samples),
        "render_style": render_style,
    }
    if render_style == "visualization":
        config.setdefault("model_color", [67.0 / 255.0, 147.0 / 255.0, 233.0 / 255.0, 1.0])
    if occluder is not None:
        config["occluder"] = {
            **occluder,
            "color": _color_to_unit_rgba(occluder.get("color", [220, 30, 30, 255])),
        }
    if foreground_occluder is not None:
        config["foreground_occluder"] = {
            **foreground_occluder,
            "color": _color_to_unit_rgba(foreground_occluder.get("color", [28, 34, 45, 255])),
        }

    mesh_path: Optional[str] = None
    config_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        # Pre-convert with CadQuery/OCC in the parent process; this avoids relying
        # on Blender STEP import add-ons or FreeCAD inside Blender.
        mesh_path = _export_temp_stl(Path(step_path))
        config["mesh_path"] = mesh_path
        json.dump(config, config_file, ensure_ascii=True)
        config_file.close()
        subprocess.run(
            [
                blender_bin,
                "-b",
                "--python",
                str(script),
                "--",
                "--config",
                config_file.name,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Blender executable not found: {blender_bin}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
        raise RuntimeError(f"Blender render failed: {detail}") from exc
    finally:
        Path(config_file.name).unlink(missing_ok=True)
        if mesh_path is not None:
            Path(mesh_path).unlink(missing_ok=True)

    rel_paths = []
    for i in range(len(fronts)):
        out_name = f"view_{i:03d}.png"
        out_path = image_dir / out_name
        if not out_path.is_file():
            raise RuntimeError(f"Blender did not produce expected image: {out_path}")
        rel_paths.append(str(Path("images") / image_dir.name / out_name))
    return rel_paths


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _failure(sample_path: Path, stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "step_path": str(sample_path),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def process_one(job: tuple[Path, ProcessArgs]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    step_path, args = job
    try:
        rel_step = step_path.relative_to(args.input_dir)
    except ValueError:
        rel_step = Path(step_path.name)
    sample_id = sample_id_from_relative_path(rel_step)
    point_path = args.output_dir / "points" / f"{sample_id}.npz"
    image_dir = args.output_dir / "images" / sample_id
    if args.skip_existing and point_path.is_file() and image_dir.is_dir():
        return None, None

    try:
        mesh_raw, tri_mapping, loader = load_step_as_mesh(
            step_path,
            triangle_face_tol=args.triangle_face_tol,
            angle_tol_rads=args.angle_tol_rads,
        )
    except Exception as exc:
        return None, _failure(step_path, "load_step", exc)

    try:
        mesh = mesh_raw.copy()
        raw_center = np.zeros(3, dtype=np.float64)
        scale = 1.0
        raw_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if args.normalize == "unit_cube":
            mesh, _, raw_bounds, scale, raw_center = normalize_trimesh_unit_cube(mesh)
        points = mesh_to_point_cloud(mesh, args.num_points)
        point_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            point_path,
            points=points.astype(np.float32),
            source_step_path=str(rel_step),
            sample_id=sample_id,
            raw_bounds=raw_bounds.astype(np.float32),
            raw_center=raw_center.astype(np.float32),
            scale=np.asarray(scale, dtype=np.float32),
        )
    except Exception as exc:
        return None, _failure(step_path, "point_cloud", exc)

    try:
        fronts = get_view_fronts(args.num_views)
        if args.render_backend == "blender-step":
            if args.normalize != "unit_cube":
                raise ValueError("blender-step requires --normalize unit_cube")
            foreground_occluder = None
            if args.foreground_occluder:
                foreground_occluder = make_foreground_occluder_config(
                    len(fronts),
                    args.foreground_occluder_seed,
                    args.foreground_occluder_color,
                    args.foreground_occluder_size_min,
                    args.foreground_occluder_size_max,
                    args.foreground_occluder_depth,
                )
            image_rels = render_step_views_with_blender(
                step_path,
                image_dir,
                fronts,
                img_size=args.img_size,
                camera_distance=args.camera_distance,
                raw_center=raw_center,
                scale=scale,
                blender_bin=args.blender_bin,
                blender_script=args.blender_script,
                foreground_occluder=foreground_occluder,
                engine=args.blender_engine,
                samples=args.blender_samples,
                render_style=args.blender_style,
            )
        else:
            image_rels = render_views_to_png(
                mesh,
                image_dir,
                fronts,
                img_size=args.img_size,
                camera_distance=args.camera_distance,
                render_width=args.img_size,
                render_height=args.img_size,
                render_backend=args.render_backend,
            )
    except Exception as exc:
        return None, _failure(step_path, "render", exc)

    record = {
        "sample_id": sample_id,
        "step_path": str(rel_step),
        "point_path": str(Path("points") / point_path.name),
        "image_paths": image_rels,
        "num_points": int(args.num_points),
        "num_views": int(args.num_views),
        "img_size": int(args.img_size),
        "normalize": args.normalize,
        "loader": loader,
        "triangle_mapping": tri_mapping is not None,
    }
    return record, None


def _pool_initializer() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build STEP point-cloud and image assets.")
    parser.add_argument("--input-dir", required=True, help="Directory containing STEP/STP files")
    parser.add_argument("--output-dir", required=True, help="Output asset directory")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--filename-pattern", default="*", help="fnmatch pattern for STEP file names")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of STEP files")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--num-points", type=int, default=4096, help="Surface points per sample")
    parser.add_argument("--num-views", type=int, default=6, help="Number of rendered views")
    parser.add_argument("--img-size", type=int, default=512, help="Square render size")
    parser.add_argument("--camera-distance", type=float, default=-0.9, help="Camera distance used by Blender")
    parser.add_argument("--triangle-face-tol", type=float, default=0.01)
    parser.add_argument("--angle-tol-rads", type=float, default=0.1)
    parser.add_argument("--normalize", choices=("none", "unit_cube"), default="unit_cube")
    parser.add_argument("--render-backend", choices=("trimesh", "blender-step"), default="trimesh")
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--blender-script", type=Path, default=None)
    parser.add_argument("--blender-engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--blender-samples", type=int, default=64)
    parser.add_argument("--blender-style", choices=("default", "visualization"), default="default")
    parser.add_argument("--visualization-root", default=None)
    parser.add_argument("--foreground-occluder", action="store_true")
    parser.add_argument("--foreground-occluder-seed", type=int, default=0)
    parser.add_argument("--foreground-occluder-color", default="28,34,45")
    parser.add_argument("--foreground-occluder-size-min", type=float, default=0.20)
    parser.add_argument("--foreground-occluder-size-max", type=float, default=0.34)
    parser.add_argument("--foreground-occluder-depth", type=float, default=0.45)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in raw.split(",")]
    if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
        raise ValueError("color must be R,G,B with values in [0,255]")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main(argv: Optional[Sequence[str]] = None) -> None:
    ns = parse_args(argv)
    input_dir = Path(ns.input_dir).expanduser().resolve()
    output_dir = Path(ns.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    blender_bin = resolve_blender_bin(ns.blender_bin, ns.blender_style, ns.visualization_root)
    args = ProcessArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=bool(ns.recursive),
        filename_pattern=str(ns.filename_pattern),
        num_points=int(ns.num_points),
        num_views=int(ns.num_views),
        img_size=int(ns.img_size),
        camera_distance=float(ns.camera_distance),
        triangle_face_tol=float(ns.triangle_face_tol),
        angle_tol_rads=float(ns.angle_tol_rads),
        normalize=str(ns.normalize),
        render_backend=str(ns.render_backend),
        blender_bin=blender_bin,
        blender_script=ns.blender_script,
        blender_engine=str(ns.blender_engine),
        blender_samples=int(ns.blender_samples),
        blender_style=str(ns.blender_style),
        visualization_root=ns.visualization_root,
        foreground_occluder=bool(ns.foreground_occluder),
        foreground_occluder_seed=int(ns.foreground_occluder_seed),
        foreground_occluder_color=_parse_rgb(ns.foreground_occluder_color),
        foreground_occluder_size_min=float(ns.foreground_occluder_size_min),
        foreground_occluder_size_max=float(ns.foreground_occluder_size_max),
        foreground_occluder_depth=float(ns.foreground_occluder_depth),
        skip_existing=bool(ns.skip_existing),
    )

    step_files = sorted(iter_step_files(input_dir, recursive=args.recursive, filename_pattern=args.filename_pattern))
    if ns.limit is not None:
        step_files = step_files[: int(ns.limit)]
    if not step_files:
        raise ValueError(f"no STEP files found under {input_dir}")

    jobs = list(zip(step_files, repeat(args)))
    if ns.num_processes <= 1:
        results = [process_one(job) for job in tqdm(jobs, desc="step assets")]
    else:
        pool = Pool(processes=int(ns.num_processes), initializer=_pool_initializer)
        try:
            results = list(tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs), desc="step assets"))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        finally:
            pool.close()
            pool.join()

    records = [record for record, failure in results if record is not None]
    failures = [failure for record, failure in results if failure is not None]
    _write_jsonl(output_dir / "manifest.jsonl", records)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "requested": len(step_files),
        "ok": len(records),
        "failed": len(failures),
        "render_backend": args.render_backend,
        "num_points": args.num_points,
        "num_views": args.num_views,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Done. ok={len(records)} failed={len(failures)} manifest={output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
