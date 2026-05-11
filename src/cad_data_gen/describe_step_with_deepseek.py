#!/usr/bin/env python3
"""
Batch-generate Chinese text descriptions for STEP files with DeepSeek.

The script extracts deterministic geometry/STEP statistics locally, then sends
those facts plus a bounded STEP excerpt to DeepSeek for a natural-language CAD
description. It is designed to run in the deepcad_occ conda environment.

Example:
  PYTHONPATH=/root/autodl-tmp/cad_data_gen/src conda run -n deepcad_occ python -m cad_data_gen.describe_step_with_deepseek \
    --input-dir ABCdataset/step/abc_0000_step_v00 \
    --manifest generatedata/manifest.jsonl \
    --output-dir cad_data_gen/runs/descriptions \
    --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
import yaml
from tqdm import tqdm

from cad_data_gen.step_assets import iter_step_files, load_step_as_mesh, sample_id_from_relative_path


DEFAULT_API_KEY_FILE = Path(__file__).resolve().parents[2] / ".secrets" / "deepseek_api_key"
STEP_ENTITY_RE = re.compile(r"#\d+\s*=\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)
STEP_RECORD_RE = re.compile(r"(#\d+)\s*=\s*(.*?);", re.IGNORECASE | re.DOTALL)
STEP_ENTITY_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9_]*)\s*\(", re.IGNORECASE)
STEP_REF_RE = re.compile(r"#\d+")
STEP_QUOTED_RE = re.compile(r"'((?:''|[^'])*)'")
STEP_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
STEP_UNIT_RE = re.compile(r"\b(SI_UNIT|LENGTH_UNIT|PLANE_ANGLE_UNIT|SOLID_ANGLE_UNIT)\s*\(([^;]*)\)", re.IGNORECASE)
STEP_SCHEMA_RE = re.compile(r"FILE_SCHEMA\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
STEP_NAME_RE = re.compile(r"FILE_NAME\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)

SURFACE_ENTITIES = {
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "B_SPLINE_SURFACE",
}

FEATURE_TYPE_LABELS = {
    "newSketch": "草图",
    "extrude": "拉伸",
    "revolve": "旋转",
    "fillet": "圆角",
    "chamfer": "倒角",
    "shell": "抽壳",
    "booleanBodies": "布尔实体操作",
    "boolean": "布尔操作",
    "loft": "放样",
    "sweep": "扫掠",
    "hole": "孔",
    "mirror": "镜像",
    "linearPattern": "线性阵列",
    "circularPattern": "圆周阵列",
    "transform": "变换",
    "draft": "拔模",
}

IMPORTANT_PARAMETER_IDS = {
    "operationType",
    "bodyType",
    "endBound",
    "endBoundEntity",
    "depth",
    "depthBack",
    "angle",
    "angleBack",
    "radius",
    "distance",
    "distanceTwo",
    "width",
    "width1",
    "width2",
    "offset",
    "thickness",
    "revolveType",
    "oppositeDirection",
    "defaultScope",
    "tangentPropagation",
    "rho",
    "chamferType",
    "booleanScope",
    "entities",
    "surfaceEntities",
    "axis",
    "sketchPlane",
}

COMMAND_INPUT_PARAMETER_IDS = {
    "sketchPlane",
    "entities",
    "surfaceEntities",
    "axis",
    "booleanScope",
    "endBoundEntity",
}
POINTERCAD_OPERATION_MAP = {
    "NEW": ("NewBodyFeatureOperation", "<|extrude_new|>"),
    "ADD": ("JoinFeatureOperation", "<|extrude_join|>"),
    "REMOVE": ("CutFeatureOperation", "<|extrude_cut|>"),
    "CUT": ("CutFeatureOperation", "<|extrude_cut|>"),
    "INTERSECT": ("IntersectFeatureOperation", "<|extrude_intersect|>"),
}
POINTERCAD_REVOLVE_OPERATION_MAP = {
    "NEW": ("NewBodyFeatureOperation", "<|revolve_new|>"),
    "ADD": ("JoinFeatureOperation", "<|revolve_join|>"),
    "REMOVE": ("CutFeatureOperation", "<|revolve_cut|>"),
    "CUT": ("CutFeatureOperation", "<|revolve_cut|>"),
    "INTERSECT": ("IntersectFeatureOperation", "<|revolve_intersect|>"),
}
POINTERCAD_EXTENT_TYPE_MAP = {
    "BLIND": "OneSideFeatureExtentType",
    "SYMMETRIC": "SymmetricFeatureExtentType",
    "TWO_SIDES": "TwoSidesFeatureExtentType",
    "TWO_SIDED": "TwoSidesFeatureExtentType",
    "UP_TO_NEXT": "OneSideFeatureExtentType",
    "UP_TO_FACE": "OneSideFeatureExtentType",
}
POINTERCAD_REBUILD_RULES = {
    "base_pointercad_supported_feature_types": ["newSketch", "extrude", "fillet", "chamfer"],
    "local_extension_feature_types": ["revolve"],
    "supported_feature_types": ["newSketch", "extrude", "revolve", "fillet", "chamfer"],
    "unsupported_feature_policy": (
        "If a feature is not supported by Pointer-CAD, keep the raw OFS command in "
        "command_sequence but do not invent a vector token translation."
    ),
    "sketch_vector_order": [
        "<|sketch_start|>",
        "<|pointer_enable|>(sketchPlane)",
        "<|direction_x/y/z +/-|> + plane offsets + rotation + scale",
        "<|profile_start|>",
        "<|loop_start|>",
        "<|curve_start|> for each line/circle/arc",
    ],
    "extrude_vector_order": [
        "sketch vector",
        "<|extrude_start|>",
        "extent_one",
        "extent_two",
        "<|extrude_new|>|<|extrude_join|>|<|extrude_cut|>|<|extrude_intersect|>",
    ],
    "revolve_vector_order": [
        "sketch vector",
        "<|revolve_start|>",
        "<|pointer_enable|>(axis)",
        "angle_one",
        "angle_two",
        "<|revolve_new|>|<|revolve_join|>|<|revolve_cut|>|<|revolve_intersect|>",
    ],
    "fillet_vector_order": ["<|fillet_start|>", "radius", "<|pointer_enable|>(edge list)"],
    "chamfer_vector_order": ["<|chamfer_start|>", "distance", "<|pointer_enable|>(edge list)"],
}
QUERY_GEOMETRY_ID_LIMIT = 80
SKETCH_ENTITY_LIMIT = 120
SKETCH_CONSTRAINT_LIMIT = 120
DIMENSION_CONSTRAINT_TYPES = {
    "LENGTH",
    "DISTANCE",
    "RADIUS",
    "DIAMETER",
    "ANGLE",
}


IMPORTANT_ENTITIES = [
    "ADVANCED_FACE",
    "CLOSED_SHELL",
    "MANIFOLD_SOLID_BREP",
    "SHELL_BASED_SURFACE_MODEL",
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "CIRCLE",
    "ELLIPSE",
    "LINE",
    "EDGE_CURVE",
    "ORIENTED_EDGE",
    "VERTEX_POINT",
    "CARTESIAN_POINT",
    "DIRECTION",
    "AXIS2_PLACEMENT_3D",
]


@dataclass(frozen=True)
class StepRecord:
    sample_id: str
    step_path: Path
    relative_step_path: str
    source_index: Optional[int] = None
    dataset_key: Optional[str] = None
    ofs_path: Optional[Path] = None
    feat_path: Optional[Path] = None
    meta_path: Optional[Path] = None


def _round_array(values: np.ndarray, digits: int = 6) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def read_text_lossy(path: Path, max_chars: Optional[int] = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    marker = f"\n\n... [TRUNCATED {omitted} CHARACTERS FROM THE MIDDLE] ...\n\n"
    return text[:head] + marker + text[-tail:], True


def extract_header(text: str, max_chars: int = 8000) -> str:
    upper = text.upper()
    header_start = upper.find("HEADER;")
    data_start = upper.find("DATA;")
    if header_start >= 0 and data_start > header_start:
        return text[header_start:data_start][:max_chars]
    return text[:max_chars]


def _step_strings(body: str) -> list[str]:
    return [value.replace("''", "'").strip() for value in STEP_QUOTED_RE.findall(body)]


def _entity_names(body: str) -> list[str]:
    names = [name.upper() for name in STEP_ENTITY_NAME_RE.findall(body)]
    seen = set()
    unique = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _numeric_tail(body: str, max_values: int = 4) -> list[float]:
    body_without_refs = STEP_REF_RE.sub("", body)
    values = [float(match.group(0)) for match in STEP_FLOAT_RE.finditer(body_without_refs)]
    return [round(value, 6) for value in values[-max_values:]]


def _sample_values(values: list[float], limit: int = 12) -> list[float]:
    unique = sorted({round(value, 6) for value in values})
    return unique[:limit]


def summarize_step_model_features(text: str) -> dict[str, Any]:
    """Extract a compact, modeling-oriented STEP digest for the LLM prompt."""
    records: dict[str, str] = {}
    entity_names_by_id: dict[str, list[str]] = {}
    for match in STEP_RECORD_RE.finditer(text):
        record_id = match.group(1)
        body = " ".join(match.group(2).split())
        records[record_id] = body
        entity_names_by_id[record_id] = _entity_names(body)

    products = []
    solid_names = []
    representation_names = []
    circle_radii: list[float] = []
    ellipse_params: list[list[float]] = []
    surface_parameter_samples: list[dict[str, Any]] = []
    surface_counts = Counter()

    for record_id, body in records.items():
        names = entity_names_by_id[record_id]
        strings = _step_strings(body)
        if "PRODUCT" in names and strings:
            products.append(
                {
                    "name": strings[0],
                    "label": strings[1] if len(strings) > 1 else "",
                    "description": strings[2] if len(strings) > 2 else "",
                }
            )
        if "MANIFOLD_SOLID_BREP" in names and strings:
            solid_names.append(strings[0])
        if "ADVANCED_BREP_SHAPE_REPRESENTATION" in names and strings:
            representation_names.append(strings[0])
        if "CIRCLE" in names:
            values = _numeric_tail(body, max_values=1)
            if values:
                circle_radii.append(values[-1])
        if "ELLIPSE" in names:
            values = _numeric_tail(body, max_values=2)
            if values:
                ellipse_params.append(values)

        effective_names = list(names)
        if "B_SPLINE_SURFACE_WITH_KNOTS" in effective_names and "B_SPLINE_SURFACE" in effective_names:
            effective_names.remove("B_SPLINE_SURFACE")
        for name in effective_names:
            if name in SURFACE_ENTITIES:
                surface_counts[name] += 1
                if (
                    name not in {"PLANE", "B_SPLINE_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS"}
                    and len(surface_parameter_samples) < 30
                ):
                    surface_parameter_samples.append(
                        {"type": name, "numeric_parameters": _numeric_tail(body)}
                    )

    face_surface_counts = Counter()
    for body in records.values():
        names = _entity_names(body)
        if "ADVANCED_FACE" not in names:
            continue
        for ref in STEP_REF_RE.findall(body):
            ref_surface = set(entity_names_by_id.get(ref, [])) & SURFACE_ENTITIES
            if ref_surface:
                if "B_SPLINE_SURFACE_WITH_KNOTS" in ref_surface:
                    ref_surface.discard("B_SPLINE_SURFACE")
                face_surface_counts.update(ref_surface)
                break

    product_names = sorted({item["name"] for item in products if item["name"]})
    solid_names = sorted({name for name in solid_names if name})
    representation_names = sorted({name for name in representation_names if name})

    modeling_hints = []
    if surface_counts.get("PLANE", 0):
        modeling_hints.append("平面可对应拉伸/切除后的平坦端面、侧壁或基准面")
    if surface_counts.get("CYLINDRICAL_SURFACE", 0) or circle_radii:
        modeling_hints.append("圆柱面和圆弧通常对应孔、圆柱凸台、轴套或旋转特征")
    if surface_counts.get("CONICAL_SURFACE", 0):
        modeling_hints.append("圆锥面通常对应锥台、倒角或渐缩过渡")
    if surface_counts.get("TOROIDAL_SURFACE", 0):
        modeling_hints.append("圆环面常见于圆角、环形过渡或圆管弯折")
    if surface_counts.get("SPHERICAL_SURFACE", 0):
        modeling_hints.append("球面通常对应球头、圆顶或球状装饰/过渡")
    if surface_counts.get("B_SPLINE_SURFACE_WITH_KNOTS", 0):
        modeling_hints.append("B样条曲面提示存在放样、自由曲面或复杂过渡面")
    if len(product_names) > 1 or len(solid_names) > 1:
        modeling_hints.append("多个产品/实体名称提示该 STEP 更像装配体或多实体模型")

    return {
        "product_names": product_names[:30],
        "solid_names": solid_names[:30],
        "representation_names": representation_names[:30],
        "surface_counts": dict(surface_counts),
        "face_surface_counts": dict(face_surface_counts),
        "circle_radii_samples": _sample_values(circle_radii),
        "circle_radius_count": len(circle_radii),
        "ellipse_parameter_samples": ellipse_params[:12],
        "surface_parameter_samples": surface_parameter_samples,
        "modeling_hints_from_step": modeling_hints,
    }


def parse_step_statistics(path: Path, max_step_chars: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    excerpt, was_truncated = truncate_middle(text, max_step_chars)
    entities = Counter(m.group(1).upper() for m in STEP_ENTITY_RE.finditer(text))
    schema = None
    schema_match = STEP_SCHEMA_RE.search(text)
    if schema_match:
        schema = " ".join(schema_match.group(1).split())
    file_name = None
    name_match = STEP_NAME_RE.search(text)
    if name_match:
        file_name = " ".join(name_match.group(1).split())
    units = []
    for unit_match in STEP_UNIT_RE.finditer(text):
        units.append(
            {
                "kind": unit_match.group(1).upper(),
                "raw": " ".join(unit_match.group(2).split()),
            }
        )

    important_counts = {
        name: int(entities[name])
        for name in IMPORTANT_ENTITIES
        if entities.get(name, 0) > 0
    }
    top_entities = [
        {"entity": name, "count": int(count)}
        for name, count in entities.most_common(30)
    ]
    return {
        "file_size_bytes": path.stat().st_size,
        "line_count": text.count("\n") + 1,
        "schema": schema,
        "file_name_record": file_name,
        "units": units[:20],
        "entity_count_total": int(sum(entities.values())),
        "important_entity_counts": important_counts,
        "top_entity_counts": top_entities,
        "model_features": summarize_step_model_features(text),
        "header_excerpt": extract_header(text),
        "step_excerpt": excerpt,
        "step_excerpt_truncated": was_truncated,
        "step_excerpt_chars": len(excerpt),
        "step_total_chars": len(text),
    }


def extract_mesh_metrics(
    step_path: Path,
    triangle_face_tol: float,
    angle_tol_rads: float,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        mesh, tri_mapping, loader = load_step_as_mesh(
            step_path,
            triangle_face_tol=triangle_face_tol,
            angle_tol_rads=angle_tol_rads,
        )
    except Exception as exc:
        return None, str(exc)

    bounds = mesh.bounds.astype(np.float64)
    bbox_min = bounds[0]
    bbox_max = bounds[1]
    bbox_extent = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2.0
    diagonal = float(np.linalg.norm(bbox_extent))

    metrics: dict[str, Any] = {
        "loader": loader,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "bbox_min": _round_array(bbox_min),
        "bbox_max": _round_array(bbox_max),
        "bbox_extent": _round_array(bbox_extent),
        "bbox_center": _round_array(center),
        "bbox_diagonal": round(diagonal, 6),
        "surface_area": round(float(mesh.area), 6),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
    }
    if mesh.is_watertight:
        metrics["volume"] = round(abs(float(mesh.volume)), 6)
    else:
        metrics["volume"] = None
    try:
        components = mesh.split(only_watertight=False)
        metrics["connected_components"] = int(len(components))
    except Exception:
        metrics["connected_components"] = None

    if tri_mapping is not None:
        unique_faces = np.unique(tri_mapping.astype(np.int64))
        metrics["brep_face_count_from_mapping"] = int(len(unique_faces))
        metrics["has_point_to_brep_face_mapping_source"] = True
    else:
        metrics["brep_face_count_from_mapping"] = None
        metrics["has_point_to_brep_face_mapping_source"] = False

    return metrics, None


def candidate_ofs_names(relative_step_path: str) -> list[str]:
    stem = Path(relative_step_path).stem
    candidates = []
    match = re.match(r"(.+)_step_(\d+)$", stem)
    if match:
        candidates.append(f"{match.group(1)}_featurescript_{match.group(2)}.yml")
    candidates.append(stem.replace("_step_", "_featurescript_") + ".yml")
    candidates.append(stem + ".yml")
    seen = set()
    unique = []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def build_ofs_index(ofs_dir: Optional[Path]) -> dict[str, Path]:
    if ofs_dir is None or not ofs_dir.is_dir():
        return {}
    index: dict[str, Path] = {}
    for path in sorted(ofs_dir.rglob("*.yml")):
        index.setdefault(path.name, path)
    return index


def resolve_ofs_path(record: StepRecord, ofs_index: dict[str, Path]) -> Optional[Path]:
    if record.ofs_path is not None and record.ofs_path.is_file():
        return record.ofs_path
    for name in candidate_ofs_names(record.relative_step_path):
        path = ofs_index.get(name)
        if path is not None:
            return path
    sibling_candidates = candidate_ofs_names(record.relative_step_path)
    for candidate_name in sibling_candidates:
        path = record.step_path.with_name(candidate_name)
        if path.is_file():
            return path
    return None


def resolve_sibling_data_path(record: StepRecord, marker: str) -> Optional[Path]:
    stem = Path(record.relative_step_path).stem
    match = re.match(r"(.+)_step_(\d+)$", stem)
    candidates = []
    if match:
        candidates.append(f"{match.group(1)}{marker}{match.group(2)}.yml")
    candidates.append(stem.replace("_step_", marker) + ".yml")
    for name in dict.fromkeys(candidates):
        path = record.step_path.with_name(name)
        if path.is_file():
            return path
    return None


def _resolve_manifest_target(value: str, input_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = input_dir / path
    return path.resolve()


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def load_organized_manifest_records(manifest_path: Path, input_dir: Path) -> list[StepRecord]:
    grouped: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            if row.get("category") != "complete":
                continue
            key = row.get("key")
            kind = row.get("kind")
            target = row.get("target")
            if not key or kind not in {"step", "ofs", "feat", "meta"} or not target:
                continue
            group = grouped.setdefault(
                key,
                {
                    "sample_id": row.get("sample_id") or sample_id_from_relative_path(target),
                    "source_index": row_index,
                },
            )
            group[kind] = _resolve_manifest_target(target, input_dir)

    records = []
    for key in sorted(grouped):
        group = grouped[key]
        step_path = group.get("step")
        if not isinstance(step_path, Path):
            continue
        records.append(
            StepRecord(
                sample_id=str(group.get("sample_id") or sample_id_from_relative_path(step_path.name)),
                step_path=step_path,
                relative_step_path=_relative_or_name(step_path, input_dir),
                source_index=group.get("source_index"),
                dataset_key=key,
                ofs_path=group.get("ofs") if isinstance(group.get("ofs"), Path) else None,
                feat_path=group.get("feat") if isinstance(group.get("feat"), Path) else None,
                meta_path=group.get("meta") if isinstance(group.get("meta"), Path) else None,
            )
        )
    return records


def _message(node: Any) -> dict[str, Any]:
    if isinstance(node, dict):
        value = node.get("message")
        if isinstance(value, dict):
            return value
    return {}


def _round_float(value: Any, digits: int = 9) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    rounded = round(float(value), digits)
    return 0.0 if abs(rounded) < 10 ** (-digits) else rounded


def _point_on_line(geometry: dict[str, Any], parameter: Any) -> Optional[list[float]]:
    pnt_x = _round_float(geometry.get("pntX"))
    pnt_y = _round_float(geometry.get("pntY"))
    dir_x = _round_float(geometry.get("dirX"))
    dir_y = _round_float(geometry.get("dirY"))
    t = _round_float(parameter)
    if None in {pnt_x, pnt_y, dir_x, dir_y, t}:
        return None
    return [_round_float(pnt_x + dir_x * t), _round_float(pnt_y + dir_y * t)]


def _query_summary(param_message: dict[str, Any]) -> Optional[dict[str, Any]]:
    queries = param_message.get("queries")
    if not isinstance(queries, list):
        return None
    geometry_ids = []
    for query in queries:
        qmsg = _message(query)
        ids = qmsg.get("geometryIds")
        if isinstance(ids, list):
            geometry_ids.extend(str(item) for item in ids)
    summary = {
        "query_count": len(queries),
        "geometry_ids_sample": geometry_ids[:12],
    }
    if geometry_ids:
        summary["geometry_ids"] = geometry_ids[:QUERY_GEOMETRY_ID_LIMIT]
        summary["geometry_ids_truncated"] = len(geometry_ids) > QUERY_GEOMETRY_ID_LIMIT
    return summary


def _extract_feature_parameters(parameters: Any) -> list[dict[str, Any]]:
    if not isinstance(parameters, list):
        return []
    extracted = []
    for param in parameters:
        pmsg = _message(param)
        parameter_id = pmsg.get("parameterId")
        if not parameter_id or parameter_id not in IMPORTANT_PARAMETER_IDS:
            continue

        item: dict[str, Any] = {"parameter_id": parameter_id}
        if "expression" in pmsg:
            item["expression"] = pmsg.get("expression")
        elif "value" in pmsg and isinstance(pmsg.get("value"), (str, int, float, bool)):
            item["value"] = pmsg.get("value")

        query_info = _query_summary(pmsg)
        if query_info is not None:
            item.update(query_info)

        if len(item) > 1:
            extracted.append(item)
    return extracted


def _parameter_value(param: dict[str, Any]) -> Any:
    if "expression" in param:
        return param["expression"]
    if "value" in param:
        return param["value"]
    if "geometry_ids" in param:
        return param["geometry_ids"]
    if "geometry_ids_sample" in param:
        return param["geometry_ids_sample"]
    return None


def _command_inputs_from_parameters(parameters: list[dict[str, Any]]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for param in parameters:
        parameter_id = param.get("parameter_id")
        if parameter_id not in COMMAND_INPUT_PARAMETER_IDS:
            continue
        value = _parameter_value(param)
        if value not in (None, [], {}):
            inputs[str(parameter_id)] = value
        if param.get("query_count") is not None:
            inputs[f"{parameter_id}_query_count"] = param.get("query_count")
        if param.get("geometry_ids_truncated"):
            inputs[f"{parameter_id}_truncated"] = True
    return inputs


def _scalar_parameters_from_parameters(parameters: list[dict[str, Any]]) -> dict[str, Any]:
    scalars: dict[str, Any] = {}
    for param in parameters:
        parameter_id = param.get("parameter_id")
        if not parameter_id or parameter_id in COMMAND_INPUT_PARAMETER_IDS:
            continue
        value = _parameter_value(param)
        if value not in (None, [], {}):
            scalars[str(parameter_id)] = value
    return scalars


def _get_scalar(feature: dict[str, Any], key: str) -> Any:
    scalars = feature.get("scalar_parameters")
    if isinstance(scalars, dict) and key in scalars:
        return scalars[key]
    parameters = feature.get("parameters")
    if isinstance(parameters, list):
        for param in parameters:
            if isinstance(param, dict) and param.get("parameter_id") == key:
                return _parameter_value(param)
    return None


def _nonnull_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pointercad_feature_template(feature: dict[str, Any]) -> Optional[dict[str, Any]]:
    feature_type = feature.get("feature_type")
    inputs = _nonnull_dict(feature.get("inputs"))
    scalars = _nonnull_dict(feature.get("scalar_parameters"))

    if feature_type == "newSketch":
        template = {
            "supported_by_pointercad": True,
            "support_level": "base_pointercad",
            "cadmodel_class": "Sketch",
            "token_order": POINTERCAD_REBUILD_RULES["sketch_vector_order"],
            "required_inputs": ["sketchPlane"],
            "required_parameters": ["sketch_entities"],
            "pointer_inputs": {
                "sketchPlane": inputs.get("sketchPlane") or inputs.get("sketchPlane_geometry_ids_sample")
            },
            "curve_encoding": {
                "line": "curve_start + start point + pointer flag; end point follows from next curve in the closed loop",
                "circle": "curve_start + center point + pointer flag + radius",
                "arc": "curve_start + start point + pointer flag + sweep_angle + clockwise/counter_clockwise",
            },
        }
        summary = feature.get("sketch_entity_summary")
        if isinstance(summary, dict) and summary:
            template["sketch_entity_counts"] = summary
        if feature.get("constraint_details"):
            template["dimension_constraints_source"] = "parameters.constraints"
        return template

    if feature_type == "extrude":
        raw_operation = _get_scalar(feature, "operationType")
        operation_key = str(raw_operation).upper() if raw_operation is not None else ""
        cad_operation, operation_token = POINTERCAD_OPERATION_MAP.get(operation_key, (None, None))
        raw_end_bound = _get_scalar(feature, "endBound")
        end_bound_key = str(raw_end_bound).upper() if raw_end_bound is not None else ""
        extent_type = POINTERCAD_EXTENT_TYPE_MAP.get(end_bound_key, str(raw_end_bound) if raw_end_bound else None)
        depth = _get_scalar(feature, "depth")
        depth_back = _get_scalar(feature, "depthBack")
        extent_two = depth_back
        if extent_two in (None, "", [], {}):
            extent_two = depth if end_bound_key == "SYMMETRIC" else 0
        return {
            "supported_by_pointercad": operation_token is not None,
            "support_level": "base_pointercad" if operation_token is not None else None,
            "cadmodel_class": "Extrude",
            "token_order": POINTERCAD_REBUILD_RULES["extrude_vector_order"],
            "required_inputs": ["entities"],
            "pointer_inputs": {
                key: value
                for key, value in {
                    "entities": inputs.get("entities"),
                    "booleanScope": inputs.get("booleanScope"),
                }.items()
                if value not in (None, [], {})
            },
            "cadmodel_operation": cad_operation,
            "operation_token": operation_token,
            "extent_type": extent_type,
            "extent_one": depth,
            "extent_two": extent_two,
            "source_parameters": {
                key: value
                for key, value in scalars.items()
                if key in {"bodyType", "operationType", "endBound", "depth", "depthBack", "oppositeDirection"}
            },
        }

    if feature_type == "revolve":
        raw_operation = _get_scalar(feature, "operationType")
        operation_key = str(raw_operation).upper() if raw_operation is not None else ""
        cad_operation, operation_token = POINTERCAD_REVOLVE_OPERATION_MAP.get(operation_key, (None, None))
        raw_revolve_type = _get_scalar(feature, "revolveType")
        revolve_type_key = str(raw_revolve_type).upper() if raw_revolve_type is not None else ""
        angle = _get_scalar(feature, "angle")
        angle_back = _get_scalar(feature, "angleBack")
        if revolve_type_key in {"FULL", "FULL_REVOLVE"}:
            angle = angle or "360.0*deg"
            angle_back = 0
        elif revolve_type_key == "SYMMETRIC" and angle_back in (None, "", [], {}):
            angle_back = angle
        elif angle_back in (None, "", [], {}):
            angle_back = 0
        return {
            "supported_by_pointercad": operation_token is not None,
            "support_level": "local_extension" if operation_token is not None else None,
            "cadmodel_class": "Revolve",
            "token_order": POINTERCAD_REBUILD_RULES["revolve_vector_order"],
            "required_inputs": ["entities", "axis"],
            "pointer_inputs": {
                key: value
                for key, value in {
                    "entities": inputs.get("entities"),
                    "axis": inputs.get("axis"),
                    "booleanScope": inputs.get("booleanScope"),
                }.items()
                if value not in (None, [], {})
            },
            "cadmodel_operation": cad_operation,
            "operation_token": operation_token,
            "revolve_type": raw_revolve_type,
            "angle_one": angle,
            "angle_two": angle_back,
            "axis": inputs.get("axis"),
            "source_parameters": {
                key: value
                for key, value in scalars.items()
                if key
                in {
                    "bodyType",
                    "operationType",
                    "revolveType",
                    "angle",
                    "angleBack",
                    "oppositeDirection",
                }
            },
        }

    if feature_type == "fillet":
        return {
            "supported_by_pointercad": True,
            "support_level": "base_pointercad",
            "cadmodel_class": "Fillet",
            "token_order": POINTERCAD_REBUILD_RULES["fillet_vector_order"],
            "required_inputs": ["entities"],
            "pointer_inputs": {"entities": inputs.get("entities")},
            "radius": _get_scalar(feature, "radius"),
            "tangent_chain": _get_scalar(feature, "tangentPropagation"),
        }

    if feature_type == "chamfer":
        distance = (
            _get_scalar(feature, "distance")
            or _get_scalar(feature, "width")
            or _get_scalar(feature, "width1")
            or _get_scalar(feature, "offset")
        )
        return {
            "supported_by_pointercad": True,
            "support_level": "base_pointercad",
            "cadmodel_class": "Chamfer",
            "token_order": POINTERCAD_REBUILD_RULES["chamfer_vector_order"],
            "required_inputs": ["entities"],
            "pointer_inputs": {"entities": inputs.get("entities")},
            "distance": distance,
            "distance_two": _get_scalar(feature, "distanceTwo") or _get_scalar(feature, "width2"),
            "angle": _get_scalar(feature, "angle"),
            "chamfer_type": _get_scalar(feature, "chamferType"),
            "tangent_chain": _get_scalar(feature, "tangentPropagation"),
        }

    if feature_type:
        return {
            "supported_by_pointercad": False,
            "cadmodel_class": None,
            "unsupported_reason": f"Pointer-CAD cadmodel has no local translator for {feature_type}",
        }
    return None


def _compact_pointercad_template(template: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not template:
        return None
    compact = {
        key: value
        for key, value in template.items()
        if key
        in {
            "supported_by_pointercad",
            "support_level",
            "cadmodel_class",
            "operation_token",
            "cadmodel_operation",
            "extent_type",
            "extent_one",
            "extent_two",
            "revolve_type",
            "angle_one",
            "angle_two",
            "axis",
            "radius",
            "distance",
            "distance_two",
            "angle",
            "chamfer_type",
            "tangent_chain",
            "pointer_inputs",
            "required_inputs",
            "required_parameters",
            "curve_encoding",
            "source_parameters",
            "sketch_entity_counts",
            "dimension_constraints_source",
            "token_order",
            "unsupported_reason",
        }
        and value not in (None, "", [], {})
    }
    return compact or None


def _extract_constraint_parameters(parameters: Any) -> list[dict[str, Any]]:
    if not isinstance(parameters, list):
        return []
    extracted = []
    for param in parameters:
        pmsg = _message(param)
        parameter_id = pmsg.get("parameterId")
        if not parameter_id:
            continue
        item: dict[str, Any] = {"parameter_id": parameter_id}
        if "expression" in pmsg:
            item["expression"] = pmsg.get("expression")
        elif "value" in pmsg and isinstance(pmsg.get("value"), (str, int, float, bool)):
            item["value"] = pmsg.get("value")
        query_info = _query_summary(pmsg)
        if query_info is not None:
            item.update(query_info)
        if len(item) > 1:
            extracted.append(item)
    return extracted


def _constraint_summary(constraints: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not isinstance(constraints, list):
        return {}
    for constraint in constraints:
        cmsg = _message(constraint)
        ctype = cmsg.get("constraintType")
        if ctype:
            counts[str(ctype)] += 1
    return dict(counts)


def _constraint_details(constraints: Any, limit: int = SKETCH_CONSTRAINT_LIMIT) -> list[dict[str, Any]]:
    if not isinstance(constraints, list):
        return []
    details = []
    for constraint in constraints:
        cmsg = _message(constraint)
        ctype = cmsg.get("constraintType")
        if not ctype:
            continue
        parameters = _extract_constraint_parameters(cmsg.get("parameters"))
        has_dimension_expression = any("expression" in param for param in parameters)
        if str(ctype) not in DIMENSION_CONSTRAINT_TYPES and not has_dimension_expression:
            continue
        item: dict[str, Any] = {
            "constraint_type": str(ctype),
        }
        if cmsg.get("entityId"):
            item["constraint_id"] = cmsg.get("entityId")
        if parameters:
            item["parameters"] = parameters
        details.append(item)
        if len(details) >= limit:
            break
    return details


def _sketch_entity_kind(
    entity_msg: dict[str, Any],
    geometry_msg: dict[str, Any],
    entity_type: Optional[str],
    geometry_type: Optional[str],
) -> str:
    if entity_type == "BTMSketchPoint":
        return "point"
    if geometry_type == "BTCurveGeometryLine":
        return "line"
    if geometry_type == "BTCurveGeometryCircle":
        if "startParam" in entity_msg and "endParam" in entity_msg:
            return "arc"
        return "circle"
    if geometry_type == "BTCurveGeometryEllipse":
        return "ellipse"
    return str(geometry_type or entity_type or "unknown")


def _sketch_entity_detail(entity: Any) -> Optional[dict[str, Any]]:
    emsg = _message(entity)
    if not emsg:
        return None
    entity_type = entity.get("typeName") if isinstance(entity, dict) else None
    geometry_node = emsg.get("geometry")
    geometry_type = geometry_node.get("typeName") if isinstance(geometry_node, dict) else None
    geometry_msg = _message(geometry_node)
    kind = _sketch_entity_kind(emsg, geometry_msg, entity_type, geometry_type)
    raw_entity_id = emsg.get("entityId")
    item: dict[str, Any] = {
        "entity_id": raw_entity_id,
        "raw_entity_id": raw_entity_id,
        "kind": kind,
        "is_construction": bool(emsg.get("isConstruction", False)),
    }
    if not item["entity_id"]:
        item.pop("entity_id")
        item.pop("raw_entity_id")

    if kind == "point":
        x = _round_float(emsg.get("x"))
        y = _round_float(emsg.get("y"))
        if x is not None and y is not None:
            item["point"] = [x, y]
    elif kind == "line":
        start = _point_on_line(geometry_msg, emsg.get("startParam"))
        end = _point_on_line(geometry_msg, emsg.get("endParam"))
        if start is not None and end is not None:
            item["start"] = start
            item["end"] = end
        pnt_x = _round_float(geometry_msg.get("pntX"))
        pnt_y = _round_float(geometry_msg.get("pntY"))
        dir_x = _round_float(geometry_msg.get("dirX"))
        dir_y = _round_float(geometry_msg.get("dirY"))
        if None not in {pnt_x, pnt_y}:
            item["point_on_line"] = [pnt_x, pnt_y]
        if None not in {dir_x, dir_y}:
            item["direction"] = [dir_x, dir_y]
    elif kind in {"circle", "arc"}:
        center_x = _round_float(geometry_msg.get("xCenter"))
        center_y = _round_float(geometry_msg.get("yCenter"))
        radius = _round_float(geometry_msg.get("radius"))
        if center_x is not None and center_y is not None:
            item["center"] = [center_x, center_y]
        if radius is not None:
            item["radius"] = radius
        if "clockwise" in geometry_msg:
            item["clockwise"] = bool(geometry_msg.get("clockwise"))
        if kind == "arc":
            start_angle = _round_float(emsg.get("startParam"))
            end_angle = _round_float(emsg.get("endParam"))
            if start_angle is not None:
                item["start_angle"] = start_angle
            if end_angle is not None:
                item["end_angle"] = end_angle
    else:
        numeric_geometry = {
            key: _round_float(value)
            for key, value in geometry_msg.items()
            if isinstance(value, (int, float))
        }
        if numeric_geometry:
            item["geometry_numeric"] = numeric_geometry

    return item


def _extract_sketch_entities(entities: Any, limit: int = SKETCH_ENTITY_LIMIT) -> list[dict[str, Any]]:
    if not isinstance(entities, list):
        return []
    extracted = []
    for entity in entities:
        detail = _sketch_entity_detail(entity)
        if detail:
            extracted.append(detail)
        if len(extracted) >= limit:
            break
    return extracted


def _sketch_entity_summary(entities: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in _extract_sketch_entities(entities):
        counts[str(item.get("kind", "unknown"))] += 1
    return dict(counts)


def _geometry_index_entry(command: dict[str, Any], entity: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "entity_id",
        "raw_entity_id",
        "kind",
        "is_construction",
        "start",
        "end",
        "point",
        "center",
        "radius",
        "start_angle",
        "end_angle",
        "clockwise",
        "point_on_line",
        "direction",
    )
    entry = {
        key: entity[key]
        for key in keys
        if key in entity and entity[key] not in (None, "", [], {})
    }
    entry.update(
        {
            "source": "sketch_entity",
            "step_index": command.get("index") or command.get("step_index"),
            "source_feature_id": command.get("feature_id"),
        }
    )
    return entry


def build_geometry_id_index(sequence: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for command in sequence:
        if not isinstance(command, dict):
            continue
        for entity in command.get("sketch_entities") or []:
            if not isinstance(entity, dict):
                continue
            raw_id = entity.get("raw_entity_id") or entity.get("entity_id")
            if not isinstance(raw_id, str) or not raw_id:
                continue
            entry = _geometry_index_entry(command, entity)
            index.setdefault(raw_id, entry)
    return index


def _iter_geometry_refs(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "", [], {})]
    if value not in (None, "", [], {}):
        return [str(value)]
    return []


def _lookup_geometry_ref(ref: str, geometry_id_index: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    if ref in geometry_id_index:
        return geometry_id_index[ref]
    for raw_id, entry in geometry_id_index.items():
        if ref.startswith(f"{raw_id}."):
            return entry
    return None


def resolve_command_inputs(
    inputs: dict[str, Any],
    geometry_id_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key in ("axis", "entities", "surfaceEntities", "booleanScope", "sketchPlane", "endBoundEntity"):
        matches = []
        for ref in _iter_geometry_refs(inputs.get(key)):
            entry = _lookup_geometry_ref(ref, geometry_id_index)
            if entry is not None:
                matches.append(dict(entry, query_ref=ref))
        if matches:
            resolved[key] = matches
    return resolved


def _line_length_2d(entity: dict[str, Any]) -> float:
    start = entity.get("start")
    end = entity.get("end")
    if (
        not isinstance(start, list)
        or not isinstance(end, list)
        or len(start) < 2
        or len(end) < 2
        or not all(isinstance(value, (int, float)) for value in [*start[:2], *end[:2]])
    ):
        return 0.0
    return float(np.hypot(float(start[0]) - float(end[0]), float(start[1]) - float(end[1])))


def _axis_line_score(entity: dict[str, Any]) -> tuple[int, float, float]:
    start = entity.get("start")
    end = entity.get("end")
    construction_score = 1 if entity.get("is_construction") else 0
    if not isinstance(start, list) or not isinstance(end, list) or len(start) < 2 or len(end) < 2:
        return construction_score, float("-inf"), 0.0
    if not all(isinstance(value, (int, float)) for value in [*start[:2], *end[:2]]):
        return construction_score, float("-inf"), 0.0
    x_axis_distance = abs(float(start[1])) + abs(float(end[1]))
    y_axis_distance = abs(float(start[0])) + abs(float(end[0]))
    axis_closeness = -min(x_axis_distance, y_axis_distance)
    return construction_score, axis_closeness, _line_length_2d(entity)


def fallback_axis_from_previous_sketches(
    sequence: list[dict[str, Any]],
    feature_index: int,
    query_refs: list[str],
) -> Optional[dict[str, Any]]:
    for command in reversed(sequence[:feature_index]):
        if command.get("feature_type") != "newSketch":
            continue
        line_entities = [
            entity
            for entity in command.get("sketch_entities") or []
            if isinstance(entity, dict)
            and entity.get("kind") == "line"
            and isinstance(entity.get("start"), list)
            and isinstance(entity.get("end"), list)
        ]
        if not line_entities:
            continue
        best_entity = max(line_entities, key=_axis_line_score)
        if _line_length_2d(best_entity) <= 0:
            continue
        entry = _geometry_index_entry(command, best_entity)
        entry["source"] = "sketch_entity_axis_fallback"
        entry["resolution"] = (
            "construction_line"
            if best_entity.get("is_construction")
            else "axis_like_line_from_latest_sketch"
        )
        if query_refs:
            entry["query_ref"] = query_refs[0]
        return entry
    return None


def _build_command_skeleton(feature: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(feature.get("scalar_parameters") or {})
    if feature.get("sketch_entities"):
        parameters["sketch_entities"] = feature["sketch_entities"]
    if feature.get("constraint_details"):
        parameters["constraints"] = feature["constraint_details"]
    pointercad_template = _compact_pointercad_template(
        feature.get("pointercad_rebuild_template")
        if isinstance(feature.get("pointercad_rebuild_template"), dict)
        else _pointercad_feature_template(feature)
    )
    if pointercad_template:
        parameters["pointercad_vector_template"] = pointercad_template
    return {
        "step_index": feature.get("index"),
        "command_name": feature.get("name") or feature.get("feature_type"),
        "command_type": feature.get("feature_type"),
        "feature_id": feature.get("feature_id"),
        "inputs": dict(feature.get("inputs") or {}),
        "resolved_inputs": dict(feature.get("resolved_inputs") or {}),
        "parameters": parameters,
        "result": "",
        "command_text": "",
    }


def _join_refs(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value[:12])
    if value not in (None, "", [], {}):
        return str(value)
    return ""


def _fallback_command_text(command: dict[str, Any]) -> tuple[str, str]:
    command_type = command.get("command_type")
    name = command.get("command_name") or command_type or "Command"
    inputs = command.get("inputs") if isinstance(command.get("inputs"), dict) else {}
    params = command.get("parameters") if isinstance(command.get("parameters"), dict) else {}

    if command_type == "newSketch":
        plane = _join_refs(inputs.get("sketchPlane"))
        entity_count = len(params.get("sketch_entities") or [])
        result = f"Created sketch with {entity_count} deterministic sketch entities."
        text = f"{name}: create sketch"
        if plane:
            text += f" on {plane}"
        return result, text + "."

    if command_type == "extrude":
        entities = _join_refs(inputs.get("entities"))
        scope = _join_refs(inputs.get("booleanScope"))
        operation = params.get("operationType", "")
        body_type = params.get("bodyType", "")
        depth = params.get("depth", "")
        end_bound = params.get("endBound", "")
        parts = [f"{name}: extrude"]
        if entities:
            parts.append(f"entities {entities}")
        if body_type or operation:
            parts.append(f"as {body_type} {operation}".strip())
        if depth:
            parts.append(f"with depth {depth}")
        if end_bound:
            parts.append(f"using {end_bound}")
        if scope:
            parts.append(f"into scope {scope}")
        return "Applied deterministic extrude parameters from OFS.", " ".join(parts) + "."

    if command_type == "revolve":
        entities = _join_refs(inputs.get("entities"))
        axis = _join_refs(inputs.get("axis"))
        scope = _join_refs(inputs.get("booleanScope"))
        operation = params.get("operationType", "")
        body_type = params.get("bodyType", "")
        angle = params.get("angle", "")
        revolve_type = params.get("revolveType", "")
        parts = [f"{name}: revolve"]
        if entities:
            parts.append(f"entities {entities}")
        if axis:
            parts.append(f"around axis {axis}")
        if body_type or operation:
            parts.append(f"as {body_type} {operation}".strip())
        if angle:
            parts.append(f"with angle {angle}")
        if revolve_type:
            parts.append(f"using {revolve_type}")
        if scope:
            parts.append(f"into scope {scope}")
        return "Applied deterministic revolve parameters from OFS.", " ".join(parts) + "."

    if command_type == "fillet":
        entities = _join_refs(inputs.get("entities"))
        radius = params.get("radius", "")
        text = f"{name}: fillet"
        if entities:
            text += f" entities {entities}"
        if radius:
            text += f" with radius {radius}"
        return "Applied deterministic fillet parameters from OFS.", text + "."

    if command_type == "chamfer":
        entities = _join_refs(inputs.get("entities"))
        distance = params.get("distance") or params.get("width") or params.get("width1")
        text = f"{name}: chamfer"
        if entities:
            text += f" entities {entities}"
        if distance:
            text += f" with distance {distance}"
        return "Applied deterministic chamfer parameters from OFS.", text + "."

    return "Kept OFS command parameters for downstream handling.", f"{name}: keep OFS {command_type} command."


def fallback_description_from_ofs(
    record: StepRecord,
    ofs_summary: Optional[dict[str, Any]],
    step_stats: dict[str, Any],
) -> dict[str, Any]:
    features = ofs_summary.get("feature_sequence") if isinstance(ofs_summary, dict) else None
    commands = [_build_command_skeleton(feature) for feature in features or [] if isinstance(feature, dict)]
    for command in commands:
        result, command_text = _fallback_command_text(command)
        command.setdefault("result", result)
        command.setdefault("command_text", command_text)

    model_features = step_stats.get("model_features") if isinstance(step_stats, dict) else {}
    surface_types = sorted((model_features or {}).get("surface_counts", {}).keys())
    shape_feature = "STEP 摘要包含平面和曲面边界特征。"
    if surface_types:
        shape_feature = "STEP 摘要包含 " + "、".join(surface_types[:6]) + " 等边界曲面。"

    return {
        "object_name": "model",
        "one_sentence_caption": "A CAD model reconstructed from deterministic OFS feature operations.",
        "short_caption": "OFS重建模型",
        "final_structure_summary": "该模型的建模历史由 OFS 特征树直接给出，包含草图、实体操作和边界处理命令。",
        "command_sequence": commands,
        "key_shape_features": [shape_feature, "command_sequence 保留 OFS 中的引用输入、标量参数和 Pointer-CAD vector 模板。"],
        "confidence": "medium" if commands else "low",
    }


def _merge_command_with_skeleton(command: dict[str, Any], skeleton: dict[str, Any]) -> dict[str, Any]:
    merged = dict(skeleton)
    for key in ("step_index", "command_name", "command_type", "feature_id", "result", "command_text"):
        if command.get(key) not in (None, "", [], {}):
            merged[key] = command[key]
    merged_inputs = dict(skeleton.get("inputs") or {})
    if isinstance(command.get("inputs"), dict):
        merged_inputs.update(command["inputs"])
    merged["inputs"] = merged_inputs

    merged_parameters = dict(skeleton.get("parameters") or {})
    if isinstance(command.get("parameters"), dict):
        merged_parameters.update(command["parameters"])
    # Geometry parsed from OFS is deterministic; do not let LLM-normalized
    # variants drop coordinates, flags, or constraint metadata.
    for deterministic_key in ("sketch_entities", "constraints", "pointercad_vector_template"):
        if deterministic_key in (skeleton.get("parameters") or {}):
            merged_parameters[deterministic_key] = skeleton["parameters"][deterministic_key]
    merged["parameters"] = merged_parameters
    if skeleton.get("resolved_inputs") not in (None, "", [], {}):
        merged["resolved_inputs"] = skeleton["resolved_inputs"]
    elif isinstance(command.get("resolved_inputs"), dict):
        merged["resolved_inputs"] = command["resolved_inputs"]
    return {key: value for key, value in merged.items() if value not in (None, "", [], {})}


def enrich_description_with_ofs(
    description: dict[str, Any],
    ofs_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not ofs_summary:
        return description
    features = ofs_summary.get("feature_sequence")
    if not isinstance(features, list):
        return description

    skeletons = [_build_command_skeleton(feature) for feature in features if isinstance(feature, dict)]
    by_feature_id = {
        skeleton.get("feature_id"): skeleton
        for skeleton in skeletons
        if skeleton.get("feature_id")
    }
    by_step_index = {
        skeleton.get("step_index"): skeleton
        for skeleton in skeletons
        if skeleton.get("step_index") is not None
    }

    commands = description.get("command_sequence")
    if not isinstance(commands, list) or not commands:
        description = dict(description)
        description["command_sequence"] = skeletons
        return description

    enriched = []
    used_feature_ids = set()
    for command in commands:
        if not isinstance(command, dict):
            continue
        skeleton = by_feature_id.get(command.get("feature_id")) or by_step_index.get(command.get("step_index"))
        if skeleton is None:
            enriched.append(command)
            continue
        used_feature_ids.add(skeleton.get("feature_id"))
        enriched.append(_merge_command_with_skeleton(command, skeleton))

    for skeleton in skeletons:
        feature_id = skeleton.get("feature_id")
        if feature_id and feature_id not in used_feature_ids:
            enriched.append(skeleton)

    enriched.sort(key=lambda item: item.get("step_index", 10**9))
    description = dict(description)
    description["command_sequence"] = enriched
    return description


def _slugify_entity_prefix(value: Any, max_len: int = 28) -> str:
    if not isinstance(value, str):
        return ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        return ""
    return slug[:max_len].strip("_")


def _caption_entity_prefix(value: Any, max_words: int = 3, max_len: int = 32) -> str:
    if not isinstance(value, str):
        return ""
    stop_words = {
        "a",
        "an",
        "the",
        "with",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "by",
        "as",
        "object",
        "model",
        "assembly",
        "solid",
        "consisting",
        "consists",
        "composed",
        "main",
        "body",
        "feature",
        "features",
        "structure",
        "part",
        "parts",
    }
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    content_words = [word for word in words if word not in stop_words]
    if not content_words:
        return ""
    return "_".join(content_words[:max_words])[:max_len].strip("_")


def _entity_alias_prefix(description: dict[str, Any]) -> str:
    caption_prefix = _caption_entity_prefix(description.get("one_sentence_caption"))
    if caption_prefix:
        return caption_prefix
    for key in ("object_name", "short_caption", "final_structure_summary"):
        prefix = _slugify_entity_prefix(description.get(key))
        if prefix:
            return prefix
    return "model"


def _make_entity_alias(prefix: str, step_index: Any, kind: Any, ordinal: int) -> str:
    kind_slug = _slugify_entity_prefix(kind, max_len=12) or "entity"
    try:
        step_number = int(step_index)
    except (TypeError, ValueError):
        step_number = 0
    return f"{prefix}_s{step_number:02d}_{kind_slug}{ordinal:02d}"


def _make_constraint_alias(prefix: str, step_index: Any, constraint_type: Any, ordinal: int) -> str:
    constraint_slug = _slugify_entity_prefix(constraint_type, max_len=12) or "constraint"
    try:
        step_number = int(step_index)
    except (TypeError, ValueError):
        step_number = 0
    return f"{prefix}_s{step_number:02d}_{constraint_slug}_constraint{ordinal:02d}"


def _build_entity_alias_map(description: dict[str, Any], prefix: str) -> dict[str, str]:
    commands = description.get("command_sequence")
    if not isinstance(commands, list):
        return {}
    alias_map: dict[str, str] = {}
    kind_counts_by_step: dict[tuple[int, str], int] = {}
    for command in commands:
        if not isinstance(command, dict):
            continue
        parameters = command.get("parameters")
        if not isinstance(parameters, dict):
            continue
        entities = parameters.get("sketch_entities")
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id:
                continue
            try:
                step_number = int(command.get("step_index"))
            except (TypeError, ValueError):
                step_number = 0
            kind = str(entity.get("kind") or "entity")
            count_key = (step_number, kind)
            kind_counts_by_step[count_key] = kind_counts_by_step.get(count_key, 0) + 1
            alias_map.setdefault(
                entity_id,
                _make_entity_alias(prefix, step_number, kind, kind_counts_by_step[count_key]),
            )
        constraints = parameters.get("constraints")
        if not isinstance(constraints, list):
            continue
        constraint_counts: dict[str, int] = {}
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            constraint_id = constraint.get("constraint_id")
            if not isinstance(constraint_id, str) or not constraint_id:
                continue
            constraint_type = str(constraint.get("constraint_type") or "constraint")
            constraint_counts[constraint_type] = constraint_counts.get(constraint_type, 0) + 1
            alias_map.setdefault(
                constraint_id,
                _make_constraint_alias(
                    prefix,
                    command.get("step_index"),
                    constraint_type,
                    constraint_counts[constraint_type],
                ),
            )
    return alias_map


def _replace_entity_reference(value: str, alias_map: dict[str, str]) -> str:
    replaced = value
    for raw_id, alias in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if replaced == raw_id:
            return alias
        if replaced.startswith(f"{raw_id}."):
            return f"{alias}{replaced[len(raw_id):]}"
        replaced = replaced.replace(raw_id, alias)
    return replaced


def _replace_entity_references(value: Any, alias_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_entity_reference(value, alias_map)
    if isinstance(value, list):
        return [_replace_entity_references(item, alias_map) for item in value]
    if isinstance(value, dict):
        replaced = {}
        for key, item in value.items():
            if key in {"inputs", "raw_entity_id", "query_ref"}:
                replaced[key] = item
            else:
                replaced[key] = _replace_entity_references(item, alias_map)
        return replaced
    return value


def apply_readable_entity_aliases(description: dict[str, Any]) -> dict[str, Any]:
    prefix = _entity_alias_prefix(description)
    alias_map = _build_entity_alias_map(description, prefix)
    description = dict(description)
    description["entity_id_prefix"] = prefix
    if not alias_map:
        return description
    return _replace_entity_references(description, alias_map)


SOURCE_REF_KEYS = {"raw_entity_id", "query_ref", "source_feature_id"}


def _canonical_feature_id(step_index: Any) -> str:
    try:
        step_number = int(step_index)
    except (TypeError, ValueError):
        step_number = 0
    return f"feature_s{step_number:02d}"


def _replace_unknown_ref(value: str, ref_map: dict[str, str]) -> str:
    if value in {"Top", "Front", "Right"}:
        return value
    if value not in ref_map:
        ref_map[value] = f"ref_{len(ref_map) + 1:04d}"
    return ref_map[value]


def _canonicalize_ref_value(value: Any, lookup: dict[str, str], ref_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        if value in lookup:
            return lookup[value]
        for raw_ref, alias in sorted(lookup.items(), key=lambda item: len(item[0]), reverse=True):
            if value.startswith(f"{raw_ref}."):
                return f"{alias}{value[len(raw_ref):]}"
        return _replace_unknown_ref(value, ref_map)
    if isinstance(value, list):
        return [_canonicalize_ref_value(item, lookup, ref_map) for item in value]
    if isinstance(value, dict):
        return {
            key: _canonicalize_ref_value(item, lookup, ref_map)
            for key, item in value.items()
        }
    return value


def _canonicalize_ref_text(text: str, lookup: dict[str, str], ref_map: dict[str, str]) -> str:
    replaced = text
    replacements = {**ref_map, **lookup}
    for raw_ref, canonical_ref in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        replaced = re.sub(
            rf"(?<![A-Za-z0-9_-]){re.escape(raw_ref)}(?![A-Za-z0-9_-])",
            canonical_ref,
            replaced,
        )
    return replaced


def _resolved_input_lookup(command: dict[str, Any]) -> dict[str, str]:
    lookup = {}
    resolved_inputs = command.get("resolved_inputs")
    if not isinstance(resolved_inputs, dict):
        return lookup
    for items in resolved_inputs.values():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            alias = item.get("entity_id")
            if not isinstance(alias, str) or not alias:
                continue
            for key in ("query_ref", "raw_entity_id"):
                raw_ref = item.get(key)
                if isinstance(raw_ref, str) and raw_ref:
                    lookup[raw_ref] = alias
    return lookup


def _strip_source_refs(value: Any) -> Any:
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _strip_source_refs(item)) not in (None, "", [], {})
        ]
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if key not in SOURCE_REF_KEYS
            if (cleaned := _strip_source_refs(item)) not in (None, "", [], {})
        }
    return value


def canonicalize_description_for_training(description: dict[str, Any]) -> dict[str, Any]:
    description = dict(description)
    ref_map: dict[str, str] = {}
    commands = description.get("command_sequence")
    if not isinstance(commands, list):
        return _strip_source_refs(description)

    canonical_commands = []
    for command in commands:
        if not isinstance(command, dict):
            continue
        canonical = dict(command)
        canonical["feature_id"] = _canonical_feature_id(canonical.get("step_index"))
        lookup = _resolved_input_lookup(canonical)
        if isinstance(canonical.get("inputs"), dict):
            canonical["inputs"] = _canonicalize_ref_value(canonical["inputs"], lookup, ref_map)
        for text_key in ("result", "command_text"):
            if isinstance(canonical.get(text_key), str):
                canonical[text_key] = _canonicalize_ref_text(canonical[text_key], lookup, ref_map)
        params = canonical.get("parameters")
        if isinstance(params, dict):
            params = dict(params)
            template = params.get("pointercad_vector_template")
            if isinstance(template, dict):
                template = dict(template)
                pointer_inputs = template.get("pointer_inputs")
                if isinstance(pointer_inputs, dict):
                    template["pointer_inputs"] = _canonicalize_ref_value(pointer_inputs, lookup, ref_map)
                for ref_key in ("axis", "entities", "booleanScope", "sketchPlane", "surfaceEntities", "endBoundEntity"):
                    if ref_key in template:
                        template[ref_key] = _canonicalize_ref_value(template[ref_key], lookup, ref_map)
                params["pointercad_vector_template"] = template
            canonical["parameters"] = params
        canonical_commands.append(_strip_source_refs(canonical))

    description["command_sequence"] = canonical_commands
    return _strip_source_refs(description)


def summarize_ofs_features(ofs_path: Path, max_features: int = 80) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        data = yaml.safe_load(ofs_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "OFS YAML root is not a mapping"

    feature_states = {}
    raw_states = data.get("featureStates")
    if isinstance(raw_states, list):
        for item in raw_states:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            state_msg = _message(item.get("value"))
            if key and state_msg.get("featureStatus"):
                feature_states[str(key)] = state_msg.get("featureStatus")

    sequence = []
    counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    features = data.get("features", [])
    if not isinstance(features, list):
        features = []

    for index, feature in enumerate(features):
        msg = _message(feature)
        feature_type = msg.get("featureType")
        name = msg.get("name")
        feature_id = msg.get("featureId")
        if not feature_type:
            continue
        if feature_type in {"origin", "defaultPlane"}:
            continue

        parameters = _extract_feature_parameters(msg.get("parameters"))
        operation_type = None
        for param in parameters:
            if param.get("parameter_id") == "operationType":
                operation_type = param.get("value")
                break

        counts[str(feature_type)] += 1
        if operation_type:
            operation_counts[str(operation_type)] += 1

        record = {
            "index": len(sequence) + 1,
            "feature_id": feature_id,
            "name": name,
            "feature_type": feature_type,
            "operation_label": FEATURE_TYPE_LABELS.get(str(feature_type), str(feature_type)),
            "status": feature_states.get(str(feature_id)),
            "suppressed": bool(msg.get("suppressed", False)),
            "inputs": _command_inputs_from_parameters(parameters),
            "scalar_parameters": _scalar_parameters_from_parameters(parameters),
            "parameters": parameters,
        }
        if feature_type == "newSketch":
            constraints = msg.get("constraints")
            entities = msg.get("entities")
            sketch_entities = _extract_sketch_entities(entities)
            constraint_details = _constraint_details(constraints)
            record["constraint_summary"] = _constraint_summary(constraints)
            record["constraint_details"] = constraint_details
            record["sketch_entity_summary"] = _sketch_entity_summary(entities)
            record["sketch_entities"] = sketch_entities
            record["sketch_entities_truncated"] = (
                isinstance(entities, list) and len(entities) > len(sketch_entities)
            )
            record["constraint_details_truncated"] = (
                isinstance(constraints, list) and len(constraints) > len(constraint_details)
            )
        pointercad_template = _pointercad_feature_template(record)
        if pointercad_template:
            record["pointercad_rebuild_template"] = pointercad_template
        sequence.append(record)
        if len(sequence) >= max_features:
            break

    geometry_id_index = build_geometry_id_index(sequence)
    for record_index, record in enumerate(sequence):
        inputs = record.get("inputs")
        if isinstance(inputs, dict):
            resolved_inputs = resolve_command_inputs(inputs, geometry_id_index)
            if record.get("feature_type") == "revolve" and not resolved_inputs.get("axis"):
                axis_refs = _iter_geometry_refs(inputs.get("axis"))
                fallback_axis = fallback_axis_from_previous_sketches(sequence, record_index, axis_refs)
                if fallback_axis is not None:
                    resolved_inputs["axis"] = [fallback_axis]
            if resolved_inputs:
                record["resolved_inputs"] = resolved_inputs

    key_operations = [
        item
        for item in sequence
        if item["feature_type"]
        in {
            "newSketch",
            "extrude",
            "revolve",
            "fillet",
            "chamfer",
            "shell",
            "booleanBodies",
            "loft",
            "sweep",
            "hole",
            "mirror",
            "linearPattern",
            "circularPattern",
        }
    ]

    return {
        "ofs_path": str(ofs_path),
        "feature_count_total": len(features),
        "feature_type_counts": dict(counts),
        "operation_type_counts": dict(operation_counts),
        "feature_sequence": sequence,
        "geometry_id_index": geometry_id_index,
        "key_operations": key_operations[:max_features],
        "pointercad_translation_rules": POINTERCAD_REBUILD_RULES,
        "note": "OFS feature_sequence is the primary source for real modeling operations and dimensions.",
    }, None


def _round_numeric_value(value: Any, digits: int = 6) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        rounded = round(float(value), digits)
        return 0.0 if abs(rounded) < 10 ** (-digits) else rounded
    return value


def _compact_list(value: Any, limit: int = 8) -> Any:
    if not isinstance(value, list):
        return _round_numeric_value(value)
    compact = [_round_numeric_value(item) for item in value[:limit]]
    if len(value) > limit:
        compact.append("...")
    return compact


def _compact_brep_item(item: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    compact = {}
    for key in allowed_keys:
        value = item.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            compact[key] = _compact_list(value)
        elif isinstance(value, (str, int, float, bool)):
            compact[key] = _round_numeric_value(value)
    return compact


def _bounds_from_locations(items: list[Any]) -> Optional[dict[str, list[float]]]:
    points = []
    for item in items:
        if not isinstance(item, dict):
            continue
        location = item.get("location") or item.get("center")
        if (
            isinstance(location, list)
            and len(location) >= 3
            and all(isinstance(value, (int, float)) for value in location[:3])
        ):
            points.append([float(value) for value in location[:3]])
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float64)
    return {
        "min": _round_array(arr.min(axis=0)),
        "max": _round_array(arr.max(axis=0)),
    }


def summarize_feat_file(feat_path: Optional[Path], sample_limit: int = 8) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if feat_path is None:
        return None, None
    if not feat_path.is_file():
        return None, f"FEAT file not found: {feat_path}"
    try:
        data = yaml.safe_load(feat_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "FEAT YAML root is not a mapping"

    curves = data.get("curves")
    surfaces = data.get("surfaces")
    curves = curves if isinstance(curves, list) else []
    surfaces = surfaces if isinstance(surfaces, list) else []
    curve_type_counts = Counter(str(item.get("type", "unknown")) for item in curves if isinstance(item, dict))
    surface_type_counts = Counter(str(item.get("type", "unknown")) for item in surfaces if isinstance(item, dict))

    radius_values = []
    for item in [*curves, *surfaces]:
        if isinstance(item, dict) and isinstance(item.get("radius"), (int, float)):
            radius_values.append(float(item["radius"]))

    curve_sample_keys = {"type", "radius", "location", "x_axis", "y_axis", "z_axis", "closed", "degree", "sharp"}
    surface_sample_keys = {"type", "radius", "location", "x_axis", "y_axis", "z_axis", "coefficients"}
    return {
        "feat_path": str(feat_path),
        "curve_count": len(curves),
        "surface_count": len(surfaces),
        "curve_type_counts": dict(curve_type_counts),
        "surface_type_counts": dict(surface_type_counts),
        "radius_samples": _sample_values(radius_values),
        "location_bounds": _bounds_from_locations([*curves, *surfaces]),
        "curve_samples": [
            _compact_brep_item(item, curve_sample_keys)
            for item in curves
            if isinstance(item, dict)
        ][:sample_limit],
        "surface_samples": [
            _compact_brep_item(item, surface_sample_keys)
            for item in surfaces
            if isinstance(item, dict)
        ][:sample_limit],
        "note": "FEAT B-rep summary is a cleaned geometric validation source; OFS remains primary for command order and parameters.",
    }, None


def summarize_meta_file(meta_path: Optional[Path]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if meta_path is None:
        return None, None
    if not meta_path.is_file():
        return None, f"META file not found: {meta_path}"
    try:
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "META YAML root is not a mapping"
    owner = data.get("owner")
    created_by = data.get("createdBy")
    modified_by = data.get("modifiedBy")
    return {
        "meta_path": str(meta_path),
        "document_id": data.get("id"),
        "document_name": data.get("name"),
        "description": data.get("description"),
        "tags": data.get("tags") if isinstance(data.get("tags"), list) else None,
        "created_at": data.get("createdAt"),
        "modified_at": data.get("modifiedAt"),
        "owner_name": owner.get("name") if isinstance(owner, dict) else None,
        "created_by_name": created_by.get("name") if isinstance(created_by, dict) else None,
        "modified_by_name": modified_by.get("name") if isinstance(modified_by, dict) else None,
        "note": "Metadata is only used for source naming/context; it must not invent CAD operations.",
    }, None


def build_prompt_payload(
    record: StepRecord,
    step_stats: dict[str, Any],
    mesh_metrics: Optional[dict[str, Any]],
    mesh_error: Optional[str],
    ofs_summary: Optional[dict[str, Any]],
    ofs_error: Optional[str],
    feat_summary: Optional[dict[str, Any]],
    feat_error: Optional[str],
    meta_summary: Optional[dict[str, Any]],
    meta_error: Optional[str],
) -> dict[str, Any]:
    features = step_stats.get("model_features", {})
    component_names = features.get("product_names") or features.get("solid_names") or []
    surface_types = sorted(set(features.get("surface_counts", {}).keys()))
    face_surface_types = sorted(set(features.get("face_surface_counts", {}).keys()))
    concise_features = {
        "component_names": component_names,
        "solid_names": features.get("solid_names", []),
        "surface_types_present": surface_types,
        "face_surface_types_present": face_surface_types,
        "circle_radii_samples": features.get("circle_radii_samples", []),
        "ellipse_parameter_samples": features.get("ellipse_parameter_samples", []),
        "surface_parameter_samples": features.get("surface_parameter_samples", []),
        "modeling_hints_from_step": features.get("modeling_hints_from_step", []),
    }
    return {
        "sample_id": record.sample_id,
        "dataset_key": record.dataset_key,
        "step_path": record.relative_step_path,
        "local_mesh_error": mesh_error,
        "ofs_feature_tree": ofs_summary,
        "ofs_error": ofs_error,
        "feat_brep_summary": feat_summary,
        "feat_error": feat_error,
        "metadata_summary": meta_summary,
        "metadata_error": meta_error,
        "step_model_features": concise_features,
    }


def make_messages(
    payload: dict[str, Any],
    step_excerpt: str,
    language: str,
) -> list[dict[str, str]]:
    system = (
        "你是一名严谨的 CAD/机械结构数据标注专家。"
        "你会阅读 Onshape FeatureScript/OFS 特征树、ABC FEAT B-rep 清洗摘要、META 元数据和 STEP 最终几何摘要，为模型生成训练用文本描述。"
        "OFS 是真实建模操作和尺寸参数的主依据，STEP 只用于校验最终几何。"
        "FEAT 摘要只用于校验曲线/曲面类型、半径样本和空间分布，META 只用于对象命名和来源上下文。"
        "目标 command_sequence 需要尽量贴近 CAD vector schema 的可重建命令：Sketch、Extrude、Revolve、Fillet、Chamfer。"
        "其中 Revolve 是本项目在 Pointer-CAD 基础上的本地扩展能力。"
        "描述建模过程时优先使用确定性的动词和句式，避免把 OFS 已明确给出的操作写成推测。"
        "必须区分确定事实和推测；不得编造未提供的尺寸、数量、用途或功能。"
    )
    excerpt_block = ""
    if step_excerpt.strip():
        excerpt_block = f"""

STEP 原文片段，仅作为辅助，不要照抄实体编号：
<<<STEP
{step_excerpt}
STEP
"""
    else:
        excerpt_block = "\n\n未发送原始 STEP 原文；请只依据清洗后的 STEP 造型摘要输出。"

    user = f"""
请基于下面的 OFS 特征树和 STEP 造型摘要输出一个严格 JSON 对象，不要使用 Markdown。
语言：{language}

要求：
1. 给出 object_name：一个英文单词、小写、无空格，用作 entity_id 的语义前缀；含义应概括对象类别，例如 hinge、bracket、ring、plate。
2. 给出 one_sentence_caption：一个清晰简洁的英文单句，概括整体形状和关键结构特征，重点描述几何形态、对称性、主要拉伸/切除和显著元素，避免用途解释和无关细节。
3. 给出一个简短中文标题 short_caption。
4. 给出 final_structure_summary，用自然语言概括最终模型由哪些主体、附加结构和关键曲面/边界特征构成。
5. 给出 command_sequence，这是最重要的字段，必须按 OFS feature_sequence 的真实顺序组织成可复原建模历史。
6. command_sequence 的每一项代表一个 CAD 命令，而不是外观描述；每项必须包含 step_index、command_name、command_type、feature_id、inputs、parameters、result、command_text；如果输入中已有 resolved_inputs，必须原样保留。
7. 给出 key_shape_features 数组，每项是面向训练的短句，保留最终形状中最关键的结构特征，避免“有103个面”这种统计描述。
8. 给出 confidence，取 high/medium/low。

强约束：
- 不要在最终描述中输出边界盒范围、三角形数量、顶点数量、表面积、体积、欧拉数、面数、曲面数量、实体数量等统计信息。
- 不要写“包含25个面、7个球面、5个圆锥面”这类句子；只能写“包含球面、圆锥面、圆环面等曲面特征”。
- 对 component_names 必须忠实：直接列名称，不要改名，不要增减，也不要写“六个实体/三个实体”这类数量化表述。
- 不要输出“被忽略的信息”“未使用的信息”“单位不明确”“STEP片段截断”等与模型造型无关的元说明。
- 如果 ofs_feature_tree 存在，建模流程必须以 OFS 为准，不要再用“可能先...”替代真实操作。
- 如果 OFS 和 STEP 推断冲突，优先相信 OFS 的 feature_type、operationType 和参数。
- 如果 FEAT B-rep 摘要和 STEP 摘要冲突，只把 FEAT 当作最终几何校验信号，不要用 FEAT 曲面列表反推并新增 command_sequence 命令。
- metadata_summary.document_name 可以辅助 object_name/标题，但不得覆盖 OFS 中真实命令和参数。
- 尺寸数据必须保留原始表达式，例如 "0.2*in"、"180.0*deg"、"0.1*in"。
- 已在 command_sequence.parameters 中出现的明确参数不要再单独汇总成 precise_operation_data。
- 不要输出 caveats、limitations、notes、uncertainties 等尾部风险说明字段。
- command_sequence.command_text 应写成确定的建模命令，例如“创建草图...”“将草图拉伸为...”“对该实体执行倒角...”。输入缺失或 OFS/STEP 无法支撑的细节应省略，不要用推测词补足。
- 确定性描述不等于补全缺失信息。无法区分孔/凸台/凹槽/实体用途时，使用“圆柱面特征”“圆形轮廓”“重复实体”等中性表述，不要写“孔或凸台”“可能是...”。
- key_shape_features 只能写由 OFS 或 STEP 摘要直接支持的造型事实；不要断言阵列方向、阵列数量、对称性、用途或零件类别，除非输入参数明确给出。
- final_structure_summary 和 key_shape_features 尽量避免“可能”“或”“约”等词；如果必须表达不确定性，改用更宽泛但确定的形状类别。
- final_structure_summary、command_sequence、key_shape_features 中禁止使用“可能”“或许”“大概”“推测”“疑似”“类似”“孔或凸台”“块状或轴类”等模糊表达；把不确定的具体类别退回到更抽象的确定类别，例如“含圆柱面和平面的拉伸实体”。
- 不要把 STEP 的采样半径或曲面参数写成“约为”；可以直接写“STEP 摘要显示圆柱面半径样本包含 ...”，也可以省略尺寸。
- 输出前自检：最终 JSON 的所有字符串字段都不得包含“可能”“或许”“大概”“推测”“疑似”“约为”“孔或凸台”“块状或轴类”“方向明确”。如果某句话需要这些词才能成立，就删除这句话或改写为更宽泛的确定事实。
- 输出前自检：最终 JSON 顶层只允许包含 schema 中列出的键，不得额外加入 precise_operation_data、caveats、limitations、notes、uncertainties。
- command_sequence 必须尽量可用于还原模型结构：草图命令复制 OFS 中的 sketchPlane、sketch_entities、constraint_details；拉伸/旋转命令写明 bodyType、operationType、entities、axis、endBound、depth、angle、booleanScope 等输入中存在的原始参数；圆角/倒角/抽壳/拔模/阵列/镜像/布尔命令写明半径、厚度、角度、目标实体、作用范围或操作类型。
- feature_sequence 中的 inputs 是可重建命令的引用输入，scalar_parameters 是可重建命令的标量/枚举参数；生成 command_sequence 时优先逐项复制这些字段，不要把它们改写成自然语言。
- feature_sequence.pointercad_rebuild_template 是从本地 CAD vector translator 整理出的模板；如果存在，必须复制到对应 command_sequence.parameters.pointercad_vector_template，保留 supported_by_pointercad、support_level、cadmodel_class、token_order、pointer_inputs、operation_token、extent/radius/distance 等字段。
- 原始 Pointer-CAD 基础能力支持 Sketch、Extrude、Fillet、Chamfer；Revolve 是本项目新增的 local_extension。遇到 supported_by_pointercad=false 的命令时保留 OFS 命令本身，但不要编造 operation_token 或 vector token。
- Extrude 的 operationType 必须按模板映射：NEW=<|extrude_new|>，ADD=<|extrude_join|>，REMOVE/CUT=<|extrude_cut|>，INTERSECT=<|extrude_intersect|>；endBound/depth/depthBack 保留为 extent_type、extent_one、extent_two 的来源。
- Revolve 的 operationType 必须按模板映射：NEW=<|revolve_new|>，ADD=<|revolve_join|>，REMOVE/CUT=<|revolve_cut|>，INTERSECT=<|revolve_intersect|>；axis 是指针输入，angle/angleBack/revolveType 保留为 angle_one、angle_two、revolve_type 的来源。
- Fillet/Chamfer 的 entities 是边引用指针输入，radius/distance/width/angle/tangentPropagation 是重建参数；不要把这些参数只写进自然语言。
- 草图的 sketch_entities 是局部 2D 曲线定义，包含 line/arc/circle/point 的坐标、半径、角度等；必须放入对应草图命令的 parameters.sketch_entities。不要只输出“包含约束摘要”。
- sketch_entities 中的 entity_id 可在后处理中变成可读别名；raw_entity_id 是 Onshape 原始 geometry id，只用于调试和回溯，训练输出默认会移除。resolved_inputs 用于把 axis/entities 等 query 引用确定性解析回草图线、construction line 或其他几何对象。
- constraint_details 中的 LENGTH/DISTANCE/RADIUS 等表达式是草图尺寸约束，必须保留在 parameters.constraints 中；不要只保留 constraint_summary。
- entities、axis、sketchPlane、booleanScope、surfaceEntities、endBoundEntity 等几何引用放在 inputs 中；最终训练数据会被后处理成 canonical 引用，避免学习 Onshape 内部随机 id。如果 resolved_inputs 已给出解析结果，不要删除或改写。bodyType、operationType、endBound、depth、angle、radius、tangentPropagation、rho 等放在 parameters 中。
- command_sequence.parameters 只能使用 OFS 中已经出现的 expression/value/raw_value，不要换算、补全或改写单位；没有的参数就省略。
- command_text 用短命令句表达，例如“Extrude 1: extrude entities JJC,JJD as SOLID ADD with depth 0.1*in into scope JHD”。重点是重建命令，不是文学化描述。
- 为避免输出被截断，command_sequence 中不要重复写 evidence 和逐步 confidence；只在顶层给出 confidence。
- final_structure_summary 可以简短；不要让外观说明取代 command_sequence。
- object_name 必须是一个英文单词，后处理会将它作为短 entity_id 前缀；不要使用中文、空格、标点或多个单词。
- one_sentence_caption 必须是英文单句，不要使用 XML 标签；它只描述几何外形，不描述功能用途。
- 不要输出原始 OFS 的长 UUID 作为新的 entity_id；长 entity_id 会在后处理中统一替换为 object_name 前缀的短别名。

输出 JSON schema：
{{
  "object_name": "...",
  "one_sentence_caption": "...",
  "short_caption": "...",
  "final_structure_summary": "...",
  "command_sequence": [
    {{
      "step_index": 1,
      "command_name": "...",
      "command_type": "...",
      "feature_id": "...",
      "inputs": {{}},
      "resolved_inputs": {{}},
      "parameters": {{}},
      "result": "...",
      "command_text": "..."
    }}
  ],
  "key_shape_features": ["..."],
  "confidence": "high|medium|low"
}}

OFS 与 STEP 摘要：
{json.dumps(payload, ensure_ascii=False, indent=2)}
{excerpt_block}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


ALLOWED_DESCRIPTION_KEYS = (
    "object_name",
    "one_sentence_caption",
    "entity_id_prefix",
    "short_caption",
    "final_structure_summary",
    "command_sequence",
    "key_shape_features",
    "confidence",
)
ALLOWED_COMMAND_KEYS = (
    "step_index",
    "command_name",
    "command_type",
    "feature_id",
    "inputs",
    "resolved_inputs",
    "parameters",
    "result",
    "command_text",
)
FORBIDDEN_DESCRIPTION_KEYS = {
    "precise_operation_data",
    "caveats",
    "limitations",
    "notes",
    "uncertainties",
}
ALIASABLE_ID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}(?:[.\w-]*)?")
FORBIDDEN_UNCERTAIN_TERMS = (
    "可能",
    "或许",
    "大概",
    "推测",
    "疑似",
    "类似",
    "方向明确",
)
UNCERTAIN_REPLACEMENTS = {
    "孔或凸台": "圆柱面特征",
    "通孔或槽": "贯穿切除特征",
    "孔或槽": "切除特征",
    "孔或凹槽": "圆柱面特征",
    "凸台或孔": "圆柱面特征",
    "凸起或台阶": "局部拉伸特征",
    "凸起或凹槽": "局部拉伸特征",
    "圆形或椭圆形": "曲线轮廓",
    "装配体或多实体模型": "多实体模型",
    "块状或轴类": "含圆柱面和平面的实体",
    "类似一个": "呈",
    "类似": "呈",
    "约为": "为",
    "约": "",
}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；;])", text)
    return [part for part in parts if part]


def sanitize_description_text(text: str) -> str:
    cleaned = text
    for old, new in UNCERTAIN_REPLACEMENTS.items():
        cleaned = cleaned.replace(old, new)
    sentences = []
    for sentence in _split_sentences(cleaned):
        if any(term in sentence for term in FORBIDDEN_UNCERTAIN_TERMS):
            continue
        sentences.append(sentence)
    return "".join(sentences).strip()


def sanitize_description_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_description_text(value)
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := sanitize_description_value(item)) not in ("", None, [], {})
        ]
    if isinstance(value, dict):
        return {
            key: sanitized
            for key, item in value.items()
            if key not in FORBIDDEN_DESCRIPTION_KEYS
            if (sanitized := sanitize_description_value(item)) not in ("", None, [], {})
        }
    return value


def sanitize_command_sequence(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    commands = []
    for item in value:
        if not isinstance(item, dict):
            continue
        command = {
            key: sanitized
            for key in ALLOWED_COMMAND_KEYS
            if key in item
            if (sanitized := sanitize_description_value(item[key])) not in ("", None, [], {})
        }
        if command:
            commands.append(command)
    return commands


def sanitize_description_json(description: dict[str, Any]) -> dict[str, Any]:
    if "command_sequence" not in description and "construction_sequence" in description:
        description = dict(description)
        description["command_sequence"] = description["construction_sequence"]
    if "final_structure_summary" not in description:
        fallback_summary = description.get("geometry_description") or description.get("object_overview")
        if isinstance(fallback_summary, str):
            description = dict(description)
            description["final_structure_summary"] = fallback_summary
    sanitized = {}
    for key in ALLOWED_DESCRIPTION_KEYS:
        if key not in description:
            continue
        value = (
            sanitize_command_sequence(description[key])
            if key == "command_sequence"
            else sanitize_description_value(description[key])
        )
        if value not in ("", None, [], {}):
            sanitized[key] = value
    return sanitized


def deepseek_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def call_deepseek(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    json_mode: bool,
) -> tuple[str, dict[str, Any]]:
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}

    encoded = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = deepseek_url(api_base)

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            content = response_payload["choices"][0]["message"]["content"]
            return content, response_payload.get("usage", {})
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except Exception as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(str(last_error))


def load_manifest_records(manifest_path: Path, input_dir: Path) -> list[StepRecord]:
    if manifest_path.suffix.lower() == ".csv":
        return load_organized_manifest_records(manifest_path, input_dir)

    records: list[StepRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rel_step = item["step_path"]
            records.append(
                StepRecord(
                    sample_id=item.get("sample_id") or sample_id_from_relative_path(rel_step),
                    step_path=input_dir / rel_step,
                    relative_step_path=rel_step,
                    source_index=item.get("source_index", line_index),
                )
            )
    return records


def scan_step_records(
    input_dir: Path,
    recursive: bool,
    filename_pattern: str,
) -> list[StepRecord]:
    records = []
    for step_path in sorted(
        iter_step_files(input_dir, recursive, filename_pattern),
        key=lambda p: str(p.relative_to(input_dir)),
    ):
        rel_step = str(step_path.relative_to(input_dir))
        records.append(
            StepRecord(
                sample_id=sample_id_from_relative_path(rel_step),
                step_path=step_path,
                relative_step_path=rel_step,
            )
        )
    return records


def load_done_ids(descriptions_path: Path) -> set[str]:
    done: set[str] = set()
    if not descriptions_path.is_file():
        return done
    with descriptions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = item.get("sample_id")
            if sample_id:
                done.add(str(sample_id))
    return done


def load_api_key(env_name: str, key_file: Optional[str]) -> Optional[str]:
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value.strip()
    if not key_file:
        return None
    path = Path(key_file).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def write_text_description(text_dir: Path, sample_id: str, content: str) -> str:
    text_dir.mkdir(parents=True, exist_ok=True)
    path = text_dir / f"{sample_id}.txt"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return str(path)


def process_record(
    record: StepRecord,
    args: argparse.Namespace,
    api_key: Optional[str],
    ofs_index: Optional[dict[str, Path]] = None,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not record.step_path.is_file():
        return None, {
            "sample_id": record.sample_id,
            "step_path": record.relative_step_path,
            "stage": "locate_step",
            "error": f"STEP file not found: {record.step_path}",
        }

    try:
        step_stats = parse_step_statistics(record.step_path, args.max_step_chars)
    except Exception as exc:
        return None, {
            "sample_id": record.sample_id,
            "step_path": record.relative_step_path,
            "stage": "parse_step_text",
            "error": str(exc),
        }

    mesh_metrics, mesh_error = extract_mesh_metrics(
        record.step_path,
        triangle_face_tol=args.triangle_face_tol,
        angle_tol_rads=args.angle_tol_rads,
    )
    ofs_summary = None
    ofs_error = None
    ofs_path = resolve_ofs_path(record, ofs_index or {})
    if ofs_path is not None:
        ofs_summary, ofs_error = summarize_ofs_features(ofs_path, max_features=args.max_ofs_features)
    elif args.ofs_dir:
        ofs_error = "No matching OFS file found"

    feat_summary, feat_error = (None, None)
    meta_summary, meta_error = (None, None)
    if not args.skip_feat_meta:
        feat_path = record.feat_path or resolve_sibling_data_path(record, "_features_")
        meta_path = record.meta_path or resolve_sibling_data_path(record, "_metadata_")
        feat_summary, feat_error = summarize_feat_file(feat_path)
        meta_summary, meta_error = summarize_meta_file(meta_path)

    payload = build_prompt_payload(
        record,
        step_stats,
        mesh_metrics,
        mesh_error,
        ofs_summary,
        ofs_error,
        feat_summary,
        feat_error,
        meta_summary,
        meta_error,
    )
    messages = make_messages(payload, step_stats["step_excerpt"], args.language)

    if args.dry_run:
        return {
            "sample_id": record.sample_id,
            "step_path": record.relative_step_path,
            "status": "dry_run",
            "prompt_messages": messages,
            "local_payload": payload,
        }, None

    if not api_key:
        return None, {
            "sample_id": record.sample_id,
            "step_path": record.relative_step_path,
            "stage": "api_key",
            "error": (
                f"Missing API key. Set env var {args.api_key_env} "
                f"or create key file: {args.api_key_file}"
            ),
        }

    try:
        raw_description, usage = call_deepseek(
            messages,
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
            retries=args.retries,
            json_mode=args.json_mode,
        )
    except Exception as exc:
        return None, {
            "sample_id": record.sample_id,
            "step_path": record.relative_step_path,
            "stage": "deepseek_api",
            "error": str(exc),
        }

    parsed = parse_json_object(raw_description)
    if parsed is None:
        parsed = fallback_description_from_ofs(record, ofs_summary, step_stats)
    if parsed is not None:
        parsed = sanitize_description_json(parsed)
        parsed = enrich_description_with_ofs(parsed, ofs_summary)
        parsed = apply_readable_entity_aliases(parsed)
        if not args.include_raw_ids:
            parsed = canonicalize_description_for_training(parsed)
        raw_description = json.dumps(parsed, ensure_ascii=False, indent=2)
    text_path = None
    if args.write_text_files:
        text_path = write_text_description(args.output_dir / "texts", record.sample_id, raw_description)

    return {
        "sample_id": record.sample_id,
        "step_path": record.relative_step_path,
        "source_index": record.source_index,
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "local_mesh_metrics": mesh_metrics,
        "local_mesh_error": mesh_error,
        "ofs_summary": ofs_summary,
        "ofs_error": ofs_error,
        "feat_summary": feat_summary,
        "feat_error": feat_error,
        "metadata_summary": meta_summary,
        "metadata_error": meta_error,
        "step_statistics": {
            key: value
            for key, value in step_stats.items()
            if key != "step_excerpt"
        },
        "description_json": parsed,
        "description_raw": raw_description,
        "text_path": text_path,
        "usage": usage,
    }, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use DeepSeek API to describe STEP files with local CAD metrics."
    )
    parser.add_argument("--input-dir", required=True, type=str, help="Root directory of STEP files")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional generatedata/manifest.jsonl; preserves sample_id and step_path order",
    )
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument(
        "--ofs-dir",
        type=str,
        default=None,
        help="Optional ABC OFS root; matches *_step_NNN.step to *_featurescript_NNN.yml",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input-dir recursively when no manifest is given")
    parser.add_argument("--filename-pattern", default="*", help="fnmatch pattern when scanning STEP files")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N records before applying --limit")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N records")
    parser.add_argument("--resume", action="store_true", help="Skip sample_ids already in descriptions.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without calling the API")
    parser.add_argument("--write-text-files", action="store_true", help="Also write one .txt file per sample")

    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--api-key-file",
        default=str(DEFAULT_API_KEY_FILE),
        help="Local plaintext key file used only if --api-key-env is unset",
    )
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--json-mode", action="store_true", help="Send response_format=json_object")
    parser.add_argument("--language", default="简体中文")

    parser.add_argument("--max-ofs-features", type=int, default=120, help="Max OFS feature operations sent to DeepSeek")
    parser.add_argument(
        "--max-step-chars",
        type=int,
        default=0,
        help="Max raw STEP characters sent to DeepSeek; 0 sends only the cleaned STEP summary",
    )
    parser.add_argument(
        "--skip-feat-meta",
        action="store_true",
        help="Do not summarize sibling FEAT/META files from organized ABC datasets",
    )
    parser.add_argument(
        "--include-raw-ids",
        action="store_true",
        help="Keep raw Onshape geometry/query ids in description_json for debugging; default output is canonicalized for training",
    )
    parser.add_argument("--triangle-face-tol", type=float, default=0.01)
    parser.add_argument("--angle-tol-rads", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_dir = Path(args.input_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    if args.ofs_dir:
        args.ofs_dir = Path(args.ofs_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.is_dir():
        print(f"input-dir is not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    organized_manifest = args.input_dir / "manifest.csv"
    if args.manifest:
        records = load_manifest_records(Path(args.manifest).resolve(), args.input_dir)
    elif organized_manifest.is_file():
        records = load_manifest_records(organized_manifest, args.input_dir)
    else:
        records = scan_step_records(args.input_dir, args.recursive, args.filename_pattern)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    ofs_index = build_ofs_index(args.ofs_dir) if args.ofs_dir else {}

    descriptions_path = args.output_dir / "descriptions.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    done_ids = load_done_ids(descriptions_path) if args.resume else set()
    if done_ids:
        records = [record for record in records if record.sample_id not in done_ids]

    api_key = load_api_key(args.api_key_env, args.api_key_file)
    write_mode = "a" if args.resume else "w"
    n_ok = n_fail = 0
    with descriptions_path.open(write_mode, encoding="utf-8") as out_f, failures_path.open(
        write_mode, encoding="utf-8"
    ) as fail_f:
        for record in tqdm(records, desc="STEP descriptions"):
            result, failure = process_record(record, args, api_key, ofs_index)
            if result is not None:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                n_ok += 1
            if failure is not None:
                fail_f.write(json.dumps(failure, ensure_ascii=False) + "\n")
                fail_f.flush()
                n_fail += 1

    print(
        f"Done. ok={n_ok} failed={n_fail} "
        f"descriptions={descriptions_path} failures={failures_path}"
    )


if __name__ == "__main__":
    main()
