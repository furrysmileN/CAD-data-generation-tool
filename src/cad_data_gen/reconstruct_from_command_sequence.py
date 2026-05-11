#!/usr/bin/env python3
"""Reconstruct approximate STEP models from generated CAD command sequences.

This script is deterministic: it does not call an LLM. It consumes the JSON files
produced by describe_step_with_deepseek.py, extracts sketch geometry and command
parameters, and rebuilds an approximate CAD model with CadQuery.

The generated command sequence still lacks exact Onshape coordinate frames for
sketch planes, axes, body scopes, and selected edges. For that reason this is an
approximate reconstruction backend rather than a bit-exact Onshape replayer.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cadquery as cq


POINTERCAD_OPERATION_TOKENS = {
    "NEW": ("NewBodyFeatureOperation", "<|extrude_new|>"),
    "ADD": ("JoinFeatureOperation", "<|extrude_join|>"),
    "REMOVE": ("CutFeatureOperation", "<|extrude_cut|>"),
    "CUT": ("CutFeatureOperation", "<|extrude_cut|>"),
    "SUBTRACT": ("CutFeatureOperation", "<|extrude_cut|>"),
    "INTERSECT": ("IntersectFeatureOperation", "<|extrude_intersect|>"),
}
POINTERCAD_REVOLVE_OPERATION_TOKENS = {
    "NEW": ("NewBodyFeatureOperation", "<|revolve_new|>"),
    "ADD": ("JoinFeatureOperation", "<|revolve_join|>"),
    "REMOVE": ("CutFeatureOperation", "<|revolve_cut|>"),
    "CUT": ("CutFeatureOperation", "<|revolve_cut|>"),
    "SUBTRACT": ("CutFeatureOperation", "<|revolve_cut|>"),
    "INTERSECT": ("IntersectFeatureOperation", "<|revolve_intersect|>"),
}
POINTERCAD_EXTENT_TYPES = {
    "BLIND": "OneSideFeatureExtentType",
    "SYMMETRIC": "SymmetricFeatureExtentType",
    "TWO_SIDES": "TwoSidesFeatureExtentType",
    "TWO_SIDED": "TwoSidesFeatureExtentType",
    "THROUGH_ALL": "OneSideFeatureExtentType",
    "UP_TO_NEXT": "OneSideFeatureExtentType",
    "UP_TO_FACE": "OneSideFeatureExtentType",
}
POINTERCAD_RULES = {
    "base_pointercad_supported_feature_types": ["newSketch", "extrude", "fillet", "chamfer"],
    "local_extension_feature_types": ["revolve"],
    "supported_feature_types": ["newSketch", "extrude", "revolve", "fillet", "chamfer"],
    "standard_planes": ["Top", "Right", "Front"],
    "token_order": {
        "newSketch": [
            "<|sketch_start|>",
            "<|pointer_enable|>(sketchPlane)",
            "<|direction_x/y/z +/-|>",
            "<|profile_start|>",
            "<|loop_start|>",
            "<|curve_start|>...",
        ],
        "extrude": [
            "preceding sketch vector",
            "<|extrude_start|>",
            "extent_one",
            "extent_two",
            "<|extrude_new|>|<|extrude_join|>|<|extrude_cut|>|<|extrude_intersect|>",
        ],
        "revolve": [
            "preceding sketch vector",
            "<|revolve_start|>",
            "<|pointer_enable|>(axis)",
            "angle_one",
            "angle_two",
            "<|revolve_new|>|<|revolve_join|>|<|revolve_cut|>|<|revolve_intersect|>",
        ],
        "fillet": ["<|fillet_start|>", "radius", "<|pointer_enable|>(edge list)"],
        "chamfer": ["<|chamfer_start|>", "distance", "<|pointer_enable|>(edge list)"],
    },
    "curve_encoding": {
        "line": "curve_start + start point + pointer flag; the next curve determines the end point",
        "circle": "curve_start + center point + pointer flag + radius",
        "arc": "curve_start + start point + pointer flag + sweep angle + clockwise/counter_clockwise",
    },
}
LENGTH_UNIT_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "cm": 1e-2,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
}
EXPRESSION_RE = re.compile(
    r"^\s*([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?)\s*(?:\*?\s*([A-Za-z]+))?\s*$"
)


@dataclass
class Segment:
    points: list[tuple[float, float]]
    entity_id: str = ""

    @property
    def start(self) -> tuple[float, float]:
        return self.points[0]

    @property
    def end(self) -> tuple[float, float]:
        return self.points[-1]

    def reversed(self) -> "Segment":
        return Segment(list(reversed(self.points)), self.entity_id)


@dataclass
class SketchData:
    step_index: int
    profiles: list[list[tuple[float, float]]]
    vector: dict[str, Any]
    entities: list[dict[str, Any]]


@dataclass
class DescriptionRecord:
    source_path: Path
    output_stem: str
    description: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate STEP reconstruction from generated command_sequence JSON."
    )
    parser.add_argument("input", type=str, help="Input .txt/.json file or directory")
    parser.add_argument("--output", type=str, default=None, help="Output STEP path for a single input")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generatedata/reconstructed_steps",
        help="Output directory when input is a directory",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directory inputs")
    parser.add_argument("--scale", type=float, default=1000.0, help="Scale model units; default m to mm")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="2D endpoint matching tolerance")
    parser.add_argument("--arc-segments", type=int, default=24, help="Polyline segments per full circle")
    parser.add_argument("--min-profile-area", type=float, default=1e-12, help="Discard smaller profiles")
    parser.add_argument("--apply-finish", action="store_true", help="Best-effort global fillet/chamfer")
    parser.add_argument(
        "--write-vector-json",
        action="store_true",
        help="Also write Pointer-CAD-style vector translation JSON for each input",
    )
    parser.add_argument(
        "--vector-output-dir",
        type=str,
        default=None,
        help="Directory for --write-vector-json; defaults to <output-dir>/pointercad_vectors",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def description_from_json_object(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and "description_json" in data and isinstance(data["description_json"], dict):
        return data["description_json"]
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def load_description(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return description_from_json_object(data)


def parse_expression(value: Any, *, kind: str = "length", default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return default
    match = EXPRESSION_RE.match(value)
    if not match:
        return default
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    if kind == "angle":
        if unit in {"deg", "degree", "degrees"}:
            return number
        if unit in {"rad", "radian", "radians"}:
            return math.degrees(number)
        return number
    if unit:
        return number * LENGTH_UNIT_TO_M.get(unit, 1.0)
    return number


def clean_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := clean_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := clean_empty(item)) not in (None, "", [], {})
        ]
    return value


def command_params(command: dict[str, Any]) -> dict[str, Any]:
    params = command.get("parameters")
    return params if isinstance(params, dict) else {}


def command_inputs(command: dict[str, Any]) -> dict[str, Any]:
    inputs = command.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def command_resolved_inputs(command: dict[str, Any]) -> dict[str, Any]:
    resolved_inputs = command.get("resolved_inputs")
    return resolved_inputs if isinstance(resolved_inputs, dict) else {}


def pointercad_template(command: dict[str, Any]) -> dict[str, Any]:
    params = command_params(command)
    template = params.get("pointercad_vector_template")
    return template if isinstance(template, dict) else {}


def scalar_or_template(params: dict[str, Any], template: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = params.get(key)
        if value not in (None, "", [], {}):
            return value
    for key in keys:
        value = template.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def operation_from_params(
    params: dict[str, Any],
    template: dict[str, Any],
    operation_tokens: Optional[dict[str, tuple[str, str]]] = None,
) -> str:
    operation_tokens = operation_tokens or POINTERCAD_OPERATION_TOKENS
    operation = str(params.get("operationType") or "").upper()
    if operation:
        return operation
    cad_operation = str(template.get("cadmodel_operation") or "")
    for candidate, (cad_name, _) in operation_tokens.items():
        if cad_operation == cad_name:
            return "REMOVE" if candidate == "CUT" else candidate
    token = str(template.get("operation_token") or "")
    for candidate, (_, candidate_token) in operation_tokens.items():
        if token == candidate_token:
            return "REMOVE" if candidate == "CUT" else candidate
    return "ADD"


def extent_from_params(params: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    end_bound = str(params.get("endBound") or "").upper()
    extent_type = template.get("extent_type") or POINTERCAD_EXTENT_TYPES.get(end_bound)
    extent_one = scalar_or_template(params, template, "depth", "extent_one", default="0.01")
    extent_two = scalar_or_template(params, template, "depthBack", "extent_two", default=0)
    if end_bound == "SYMMETRIC" and extent_two in (None, "", [], {}, 0):
        extent_two = extent_one
    return {
        "extent_type": extent_type,
        "extent_one": extent_one,
        "extent_two": extent_two,
    }


def revolve_extent_from_params(params: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    revolve_type = scalar_or_template(params, template, "revolveType", "revolve_type")
    revolve_type_key = str(revolve_type or "").upper()
    angle_one = scalar_or_template(params, template, "angle", "angle_one", default="360.0*deg")
    angle_two = scalar_or_template(params, template, "angleBack", "angle_two", default=0)
    if revolve_type_key in {"FULL", "FULL_REVOLVE", "FULLFEATUREEXTENTTYPE"}:
        angle_one, angle_two = "360.0*deg", 0
    elif revolve_type_key == "SYMMETRIC" and angle_two in (None, "", [], {}, 0):
        angle_two = angle_one
    return {
        "revolve_type": revolve_type,
        "angle_one": angle_one,
        "angle_two": angle_two,
    }


def sketch_vector_from_command(command: dict[str, Any]) -> dict[str, Any]:
    params = command_params(command)
    inputs = command_inputs(command)
    template = pointercad_template(command)
    entities = params.get("sketch_entities") if isinstance(params.get("sketch_entities"), list) else []
    constraints = params.get("constraints") if isinstance(params.get("constraints"), list) else []
    curves = []
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("is_construction"):
            continue
        kind = entity.get("kind")
        curve: dict[str, Any] = {
            "token": "<|curve_start|>",
            "curve_type": kind,
            "entity_id": entity.get("entity_id"),
            "pointer_token": "<|pointer_disable|>",
        }
        if kind == "line":
            curve["start"] = entity.get("start")
            curve["end"] = entity.get("end")
        elif kind == "circle":
            curve["center"] = entity.get("center")
            curve["radius"] = entity.get("radius")
        elif kind == "arc":
            curve["center"] = entity.get("center")
            curve["radius"] = entity.get("radius")
            curve["start_angle"] = entity.get("start_angle")
            curve["end_angle"] = entity.get("end_angle")
            curve["direction_token"] = "<|clockwise|>" if entity.get("clockwise") else "<|counter_clockwise|>"
        else:
            curve["raw_geometry"] = entity
        curves.append(clean_empty(curve))

    return clean_empty(
        {
            "step_index": command.get("step_index"),
            "command_type": "newSketch",
            "supported_by_pointercad": template.get("supported_by_pointercad", True),
            "support_level": template.get("support_level") or "base_pointercad",
            "cadmodel_class": template.get("cadmodel_class") or "Sketch",
            "token_order": template.get("token_order") or POINTERCAD_RULES["token_order"]["newSketch"],
            "tokens": ["<|sketch_start|>", "<|pointer_enable|>", "<|profile_start|>", "<|loop_start|>"],
            "pointer_inputs": template.get("pointer_inputs") or {"sketchPlane": inputs.get("sketchPlane")},
            "curves": curves,
            "constraints": constraints,
            "rules": {"curve_encoding": POINTERCAD_RULES["curve_encoding"]},
        }
    )


def extrude_vector_from_command(command: dict[str, Any], sketch_vector: Optional[dict[str, Any]]) -> dict[str, Any]:
    params = command_params(command)
    inputs = command_inputs(command)
    template = pointercad_template(command)
    operation = operation_from_params(params, template)
    cad_operation, operation_token = POINTERCAD_OPERATION_TOKENS.get(
        operation,
        (template.get("cadmodel_operation"), template.get("operation_token")),
    )
    extent = extent_from_params(params, template)
    return clean_empty(
        {
            "step_index": command.get("step_index"),
            "command_type": "extrude",
            "supported_by_pointercad": template.get("supported_by_pointercad", operation_token is not None),
            "support_level": template.get("support_level") or ("base_pointercad" if operation_token is not None else None),
            "cadmodel_class": template.get("cadmodel_class") or "Extrude",
            "token_order": template.get("token_order") or POINTERCAD_RULES["token_order"]["extrude"],
            "tokens": ["<|extrude_start|>", operation_token],
            "operation": operation,
            "cadmodel_operation": cad_operation,
            "operation_token": operation_token,
            "extent_type": extent["extent_type"],
            "extent_one": extent["extent_one"],
            "extent_two": extent["extent_two"],
            "pointer_inputs": template.get("pointer_inputs")
            or {
                key: value
                for key, value in {
                    "entities": inputs.get("entities"),
                    "booleanScope": inputs.get("booleanScope"),
                }.items()
                if value not in (None, [], {})
            },
            "sketch_step_index": sketch_vector.get("step_index") if isinstance(sketch_vector, dict) else None,
        }
    )


def revolve_vector_from_command(command: dict[str, Any], sketch_vector: Optional[dict[str, Any]]) -> dict[str, Any]:
    params = command_params(command)
    inputs = command_inputs(command)
    template = pointercad_template(command)
    operation = operation_from_params(params, template, POINTERCAD_REVOLVE_OPERATION_TOKENS)
    cad_operation, operation_token = POINTERCAD_REVOLVE_OPERATION_TOKENS.get(
        operation,
        (template.get("cadmodel_operation"), template.get("operation_token")),
    )
    extent = revolve_extent_from_params(params, template)
    pointer_inputs = template.get("pointer_inputs")
    if not isinstance(pointer_inputs, dict):
        pointer_inputs = {
            key: value
            for key, value in {
                "entities": inputs.get("entities"),
                "axis": inputs.get("axis"),
                "booleanScope": inputs.get("booleanScope"),
            }.items()
            if value not in (None, [], {})
        }
    return clean_empty(
        {
            "step_index": command.get("step_index"),
            "command_type": "revolve",
            "supported_by_pointercad": operation_token is not None,
            "support_level": template.get("support_level") or ("local_extension" if operation_token is not None else None),
            "cadmodel_class": template.get("cadmodel_class") or "Revolve",
            "token_order": template.get("token_order") or POINTERCAD_RULES["token_order"]["revolve"],
            "tokens": ["<|revolve_start|>", "<|pointer_enable|>", operation_token],
            "operation": operation,
            "cadmodel_operation": cad_operation,
            "operation_token": operation_token,
            "revolve_type": extent["revolve_type"],
            "angle_one": extent["angle_one"],
            "angle_two": extent["angle_two"],
            "pointer_inputs": pointer_inputs,
            "axis": pointer_inputs.get("axis"),
            "sketch_step_index": sketch_vector.get("step_index") if isinstance(sketch_vector, dict) else None,
        }
    )


def finish_vector_from_command(command: dict[str, Any]) -> dict[str, Any]:
    params = command_params(command)
    inputs = command_inputs(command)
    template = pointercad_template(command)
    command_type = str(command.get("command_type") or "")
    if command_type == "fillet":
        token = "<|fillet_start|>"
        value_key = "radius"
        value = scalar_or_template(params, template, "radius")
    else:
        token = "<|chamfer_start|>"
        value_key = "distance"
        value = scalar_or_template(params, template, "distance", "width", "width1", "offset")
    return clean_empty(
        {
            "step_index": command.get("step_index"),
            "command_type": command_type,
            "supported_by_pointercad": template.get("supported_by_pointercad", command_type in {"fillet", "chamfer"}),
            "support_level": template.get("support_level") or ("base_pointercad" if command_type in {"fillet", "chamfer"} else None),
            "cadmodel_class": template.get("cadmodel_class") or command_type.capitalize(),
            "token_order": template.get("token_order") or POINTERCAD_RULES["token_order"].get(command_type),
            "tokens": [token, "<|pointer_enable|>"],
            value_key: value,
            "pointer_inputs": template.get("pointer_inputs") or {"entities": inputs.get("entities")},
            "tangent_chain": scalar_or_template(params, template, "tangentPropagation", "tangent_chain"),
        }
    )


def unsupported_vector_from_command(command: dict[str, Any]) -> dict[str, Any]:
    template = pointercad_template(command)
    return clean_empty(
        {
            "step_index": command.get("step_index"),
            "command_type": command.get("command_type"),
            "supported_by_pointercad": False,
            "unsupported_reason": template.get("unsupported_reason")
            or f"Pointer-CAD translator has no local rule for {command.get('command_type')}",
            "raw_inputs": command_inputs(command),
            "raw_parameters": {
                key: value
                for key, value in command_params(command).items()
                if key != "pointercad_vector_template"
            },
        }
    )


def as_point(value: Any, scale: float) -> Optional[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    if not all(isinstance(item, (int, float)) for item in value[:2]):
        return None
    return (float(value[0]) * scale, float(value[1]) * scale)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for current, nxt in zip(points, points[1:] + points[:1]):
        area += current[0] * nxt[1] - nxt[0] * current[1]
    return 0.5 * area


def dedupe_consecutive(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or distance(deduped[-1], point) > tolerance:
            deduped.append(point)
    if len(deduped) > 1 and distance(deduped[0], deduped[-1]) <= tolerance:
        deduped.pop()
    return deduped


def arc_points(entity: dict[str, Any], scale: float, arc_segments: int) -> list[tuple[float, float]]:
    center = as_point(entity.get("center"), scale)
    radius = entity.get("radius")
    start_angle = entity.get("start_angle")
    end_angle = entity.get("end_angle")
    if center is None or not isinstance(radius, (int, float)):
        return []
    radius = float(radius) * scale
    if not isinstance(start_angle, (int, float)) or not isinstance(end_angle, (int, float)):
        steps = max(16, arc_segments)
        return [
            (
                center[0] + radius * math.cos(2.0 * math.pi * i / steps),
                center[1] + radius * math.sin(2.0 * math.pi * i / steps),
            )
            for i in range(steps)
        ]

    start = float(start_angle)
    end = float(end_angle)
    clockwise = bool(entity.get("clockwise", False))
    if clockwise and end > start:
        end -= 2.0 * math.pi
    elif not clockwise and end < start:
        end += 2.0 * math.pi
    span = abs(end - start)
    steps = max(2, int(math.ceil(arc_segments * span / (2.0 * math.pi))))
    return [
        (
            center[0] + radius * math.cos(start + (end - start) * i / steps),
            center[1] + radius * math.sin(start + (end - start) * i / steps),
        )
        for i in range(steps + 1)
    ]


def entity_to_segments(entity: dict[str, Any], scale: float, arc_segments: int) -> list[Segment]:
    if entity.get("is_construction"):
        return []
    kind = entity.get("kind")
    entity_id = str(entity.get("entity_id") or "")
    if kind == "line":
        start = as_point(entity.get("start"), scale)
        end = as_point(entity.get("end"), scale)
        if start is None or end is None or distance(start, end) <= 0:
            return []
        return [Segment([start, end], entity_id)]
    if kind == "arc":
        points = arc_points(entity, scale, arc_segments)
        return [Segment(points, entity_id)] if len(points) >= 2 else []
    if kind == "circle":
        points = arc_points(entity, scale, arc_segments)
        return [Segment(points, entity_id)] if len(points) >= 3 else []
    return []


def chain_segments(
    segments: list[Segment],
    tolerance: float,
    min_profile_area: float,
) -> list[list[tuple[float, float]]]:
    unused = segments[:]
    profiles: list[list[tuple[float, float]]] = []
    while unused:
        chain = unused.pop(0).points[:]
        changed = True
        while changed and unused:
            changed = False
            for index, segment in enumerate(unused):
                candidates = (segment, segment.reversed())
                for candidate in candidates:
                    if distance(chain[-1], candidate.start) <= tolerance:
                        chain.extend(candidate.points[1:])
                        unused.pop(index)
                        changed = True
                        break
                    if distance(chain[0], candidate.end) <= tolerance:
                        chain = candidate.points[:-1] + chain
                        unused.pop(index)
                        changed = True
                        break
                if changed:
                    break
        chain = dedupe_consecutive(chain, tolerance)
        if len(chain) >= 3 and distance(chain[0], chain[-1]) <= tolerance:
            chain = chain[:-1]
        if len(chain) >= 3 and abs(signed_area(chain)) >= min_profile_area:
            profiles.append(chain)
    return profiles


def fallback_bbox_profile(segments: list[Segment], padding_ratio: float = 0.02) -> list[tuple[float, float]]:
    points = [point for segment in segments for point in segment.points]
    if not points:
        return []
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    pad = max(max_x - min_x, max_y - min_y, 1.0) * padding_ratio
    return [
        (min_x - pad, min_y - pad),
        (max_x + pad, min_y - pad),
        (max_x + pad, max_y + pad),
        (min_x - pad, max_y + pad),
    ]


def sketch_to_profiles(
    command: dict[str, Any],
    *,
    scale: float,
    tolerance: float,
    arc_segments: int,
    min_profile_area: float,
) -> SketchData:
    params = command.get("parameters") if isinstance(command.get("parameters"), dict) else {}
    entities = params.get("sketch_entities") if isinstance(params, dict) else []
    if not isinstance(entities, list):
        entities = []
    segments = [
        segment
        for entity in entities
        if isinstance(entity, dict)
        for segment in entity_to_segments(entity, scale, arc_segments)
    ]
    profiles = chain_segments(segments, tolerance, min_profile_area)
    if not profiles and segments:
        profiles = [fallback_bbox_profile(segments)]
    return SketchData(int(command.get("step_index", 0)), profiles, sketch_vector_from_command(command), entities)


def make_workplane_from_profiles(profiles: list[list[tuple[float, float]]]) -> Optional[cq.Workplane]:
    workplane: Optional[cq.Workplane] = None
    for profile in profiles:
        clean = dedupe_consecutive(profile, 1e-9)
        if len(clean) < 3:
            continue
        wp = cq.Workplane("XY").polyline(clean).close()
        workplane = wp if workplane is None else workplane.add(wp)
    return workplane


def make_workplane_from_profile(profile: list[tuple[float, float]]) -> Optional[cq.Workplane]:
    clean = dedupe_consecutive(profile, 1e-9)
    if len(clean) < 3:
        return None
    return cq.Workplane("XY").polyline(clean).close()


def combine_solids(solids: list[cq.Workplane]) -> Optional[cq.Workplane]:
    result: Optional[cq.Workplane] = None
    for solid in solids:
        result = solid if result is None else result.union(solid)
    return result


def apply_operation(
    current: Optional[cq.Workplane],
    tool: cq.Workplane,
    operation: str,
) -> cq.Workplane:
    operation = operation.upper()
    if current is None or operation == "NEW":
        return tool
    if operation in {"ADD", "UNION"}:
        return current.union(tool)
    if operation in {"REMOVE", "SUBTRACT", "CUT"}:
        return current.cut(tool)
    if operation == "INTERSECT":
        return current.intersect(tool)
    return current.union(tool)


def build_extrude(
    sketch: SketchData,
    params: dict[str, Any],
    template: Optional[dict[str, Any]] = None,
    *,
    scale: float,
) -> Optional[cq.Workplane]:
    template = template or {}
    extent = extent_from_params(params, template)
    depth = parse_expression(extent.get("extent_one"), kind="length", default=0.01) * scale
    back_depth = parse_expression(extent.get("extent_two"), kind="length", default=0.0) * scale
    if params.get("endBound") == "THROUGH_ALL":
        depth = max(abs(depth), 100.0)
    if params.get("endBound") == "SYMMETRIC" or extent.get("extent_type") == "SymmetricFeatureExtentType":
        depth = abs(depth) + abs(back_depth)
    if params.get("oppositeDirection"):
        depth = -abs(depth)
    solids = []
    for profile in sketch.profiles:
        wp = make_workplane_from_profile(profile)
        if wp is None:
            continue
        try:
            solids.append(wp.extrude(depth, combine=True))
        except Exception:
            continue
    return combine_solids(solids)


def _axis_refs(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value if item not in (None, "", [], {})}
    if value not in (None, "", [], {}):
        return {str(value)}
    return set()


def resolve_revolve_axis(
    sketch: SketchData,
    inputs: dict[str, Any],
    *,
    scale: float,
    resolved_axis: Optional[Any] = None,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    axis_items = resolved_axis if isinstance(resolved_axis, list) else [resolved_axis]
    for item in axis_items:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "line":
            continue
        start = as_point(item.get("start"), scale)
        end = as_point(item.get("end"), scale)
        if start is None or end is None or distance(start, end) <= 0:
            continue
        raw_id = item.get("raw_entity_id") or item.get("query_ref") or item.get("entity_id")
        return start, end, f"resolved axis line {raw_id}"

    refs = _axis_refs(inputs.get("axis"))
    candidate_lines = []
    for entity in sketch.entities:
        if not isinstance(entity, dict) or entity.get("kind") != "line":
            continue
        start = as_point(entity.get("start"), scale)
        end = as_point(entity.get("end"), scale)
        if start is None or end is None or distance(start, end) <= 0:
            continue
        entity_id = str(entity.get("entity_id") or "")
        raw_entity_id = str(entity.get("raw_entity_id") or "")
        candidate_ids = {item for item in (entity_id, raw_entity_id) if item}
        candidate_lines.append((entity_id or raw_entity_id, start, end, bool(entity.get("is_construction"))))
        if refs & candidate_ids:
            matched_id = next(iter(refs & candidate_ids))
            return start, end, f"matched sketch entity {matched_id}"

    construction_lines = [item for item in candidate_lines if item[3]]
    if construction_lines:
        entity_id, start, end, _ = max(
            construction_lines,
            key=lambda item: distance(item[1], item[2]),
        )
        return start, end, f"fallback construction line {entity_id}"

    all_points = [point for profile in sketch.profiles for point in profile]
    if all_points:
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
        return (0.0, min_y), (0.0, max_y), "fallback vertical profile axis x=0"
    return (0.0, 0.0), (0.0, 1.0), "fallback unit y-axis"


def build_revolve(
    sketch: SketchData,
    params: dict[str, Any],
    inputs: Optional[dict[str, Any]] = None,
    resolved_inputs: Optional[dict[str, Any]] = None,
    template: Optional[dict[str, Any]] = None,
    *,
    scale: float,
) -> Optional[cq.Workplane]:
    inputs = inputs or {}
    resolved_inputs = resolved_inputs or {}
    template = template or {}
    extent = revolve_extent_from_params(params, template)
    angle = parse_expression(extent.get("angle_one"), kind="angle", default=360.0)
    angle_back = parse_expression(extent.get("angle_two"), kind="angle", default=0.0)
    if str(extent.get("revolve_type", "")).upper() == "FULL":
        angle, angle_back = 360.0, 0.0
    elif str(extent.get("revolve_type", "")).upper() == "SYMMETRIC":
        angle = abs(angle) + abs(angle_back or angle)
    if params.get("oppositeDirection"):
        angle = -abs(angle)
    axis_start, axis_end, _ = resolve_revolve_axis(
        sketch,
        inputs,
        scale=scale,
        resolved_axis=resolved_inputs.get("axis"),
    )
    solids = []
    for profile in sketch.profiles:
        wp = make_workplane_from_profile(profile)
        if wp is None:
            continue
        try:
            solids.append(wp.revolve(angleDegrees=angle, axisStart=axis_start, axisEnd=axis_end))
        except Exception:
            continue
    return combine_solids(solids)


def apply_finish_operation(
    current: Optional[cq.Workplane],
    command: dict[str, Any],
    *,
    scale: float,
) -> Optional[cq.Workplane]:
    if current is None:
        return current
    params = command.get("parameters") if isinstance(command.get("parameters"), dict) else {}
    template = pointercad_template(command)
    command_type = command.get("command_type")
    try:
        if command_type == "fillet":
            radius = parse_expression(
                scalar_or_template(params, template, "radius"),
                kind="length",
                default=0.0,
            ) * scale
            return current.edges().fillet(radius) if radius > 0 else current
        if command_type == "chamfer":
            distance_value = parse_expression(
                scalar_or_template(params, template, "distance", "width", "width1", "offset", "radius"),
                kind="length",
                default=0.0,
            ) * scale
            return current.edges().chamfer(distance_value) if distance_value > 0 else current
    except Exception:
        return current
    return current


def reconstruct(description: dict[str, Any], args: argparse.Namespace) -> tuple[Optional[cq.Workplane], list[str], list[dict[str, Any]]]:
    commands = description.get("command_sequence")
    if not isinstance(commands, list):
        raise ValueError("description has no command_sequence array")

    current: Optional[cq.Workplane] = None
    last_sketch: Optional[SketchData] = None
    logs: list[str] = []
    vector_sequence: list[dict[str, Any]] = []
    for command in commands:
        if not isinstance(command, dict):
            continue
        command_type = command.get("command_type")
        params = command.get("parameters") if isinstance(command.get("parameters"), dict) else {}
        template = pointercad_template(command)
        step = command.get("step_index")
        if command_type == "newSketch":
            last_sketch = sketch_to_profiles(
                command,
                scale=args.scale,
                tolerance=args.tolerance * args.scale,
                arc_segments=args.arc_segments,
                min_profile_area=args.min_profile_area * args.scale * args.scale,
            )
            vector_sequence.append(last_sketch.vector)
            logs.append(f"step {step}: sketch profiles={len(last_sketch.profiles)}")
            continue
        if command_type in {"extrude", "revolve"}:
            resolved_inputs = command_resolved_inputs(command)
            if command_type == "extrude":
                vector_sequence.append(extrude_vector_from_command(command, last_sketch.vector if last_sketch else None))
            else:
                vector_sequence.append(revolve_vector_from_command(command, last_sketch.vector if last_sketch else None))
            if last_sketch is None or not last_sketch.profiles:
                logs.append(f"step {step}: skipped {command_type}, no usable sketch")
                continue
            tool = (
                build_extrude(last_sketch, params, template, scale=args.scale)
                if command_type == "extrude"
                else build_revolve(
                    last_sketch,
                    params,
                    command_inputs(command),
                    resolved_inputs,
                    template,
                    scale=args.scale,
                )
            )
            if tool is None:
                logs.append(f"step {step}: failed {command_type}")
                continue
            operation = operation_from_params(
                params,
                template,
                POINTERCAD_REVOLVE_OPERATION_TOKENS if command_type == "revolve" else POINTERCAD_OPERATION_TOKENS,
            )
            current = apply_operation(current, tool, operation)
            axis_note = " axis=resolved" if command_type == "revolve" and resolved_inputs.get("axis") else ""
            logs.append(f"step {step}: applied {command_type} {operation}{axis_note}")
            continue
        if command_type in {"fillet", "chamfer"}:
            vector_sequence.append(finish_vector_from_command(command))
            if args.apply_finish:
                before = current
                current = apply_finish_operation(current, command, scale=args.scale)
                logs.append(f"step {step}: finish {'applied' if current is not before else 'skipped'} {command_type}")
            else:
                logs.append(f"step {step}: translated {command_type}, finish not applied")
            continue
        vector_sequence.append(unsupported_vector_from_command(command))
        logs.append(f"step {step}: skipped unsupported {command_type}")
    return current, logs, vector_sequence


def input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = ["**/*.txt", "**/*.json", "**/*.jsonl"] if recursive else ["*.txt", "*.json", "*.jsonl"]
    files = sorted({item for pattern in patterns for item in path.glob(pattern)})
    return [item for item in files if item.is_file()]


def description_records_from_file(path: Path) -> list[DescriptionRecord]:
    if path.suffix.lower() != ".jsonl":
        return [DescriptionRecord(path, path.stem, load_description(path))]

    records: list[DescriptionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            description = description_from_json_object(data)
            if not isinstance(description.get("command_sequence"), list):
                continue
            stem = (
                str(data.get("sample_id"))
                if isinstance(data, dict) and data.get("sample_id") not in (None, "", [], {})
                else f"{path.stem}_{line_index:06d}"
            )
            records.append(DescriptionRecord(path, stem, description))
    return records


def input_records(path: Path, recursive: bool) -> list[DescriptionRecord]:
    records: list[DescriptionRecord] = []
    for file_path in input_files(path, recursive):
        records.extend(description_records_from_file(file_path))
    return records


def output_path_for(record: DescriptionRecord, args: argparse.Namespace, multiple: bool) -> Path:
    if not multiple and args.output:
        return Path(args.output)
    output_dir = Path(args.output_dir)
    return output_dir / f"{record.output_stem}.step"


def vector_output_path_for(record: DescriptionRecord, args: argparse.Namespace) -> Path:
    output_dir = Path(args.vector_output_dir) if args.vector_output_dir else Path(args.output_dir) / "pointercad_vectors"
    return output_dir / f"{record.output_stem}.pointercad.json"


def write_vector_json(
    record: DescriptionRecord,
    output_path: Path,
    description: dict[str, Any],
    logs: list[str],
    vector_sequence: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_description": str(record.source_path),
        "output_stem": record.output_stem,
        "object_name": description.get("object_name"),
        "pointercad_rules": POINTERCAD_RULES,
        "vector_sequence": vector_sequence,
        "logs": logs,
    }
    output_path.write_text(json.dumps(clean_empty(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = input_records(Path(args.input), args.recursive)
    if not records:
        raise SystemExit(f"No reconstructable descriptions found: {args.input}")

    multiple = len(records) > 1
    for record in records:
        description = record.description
        result, logs, vector_sequence = reconstruct(description, args)
        out_path = output_path_for(record, args, multiple)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.write_vector_json:
            vector_path = vector_output_path_for(record, args)
            write_vector_json(record, vector_path, description, logs, vector_sequence)
            if args.verbose:
                print(f"[vector] {record.source_path}:{record.output_stem} -> {vector_path}")
        if result is None:
            print(f"[fail] {record.source_path}:{record.output_stem}: no solid reconstructed")
            if args.verbose:
                print("\n".join(logs))
            continue
        cq.exporters.export(result, str(out_path))
        print(f"[ok] {record.source_path}:{record.output_stem} -> {out_path}")
        if args.verbose:
            print("\n".join(logs))


if __name__ == "__main__":
    main()
