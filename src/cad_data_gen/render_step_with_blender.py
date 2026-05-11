#!/usr/bin/env python3
"""
Blender-side STEP renderer.

Run only from Blender, for example:
  blender -b --python cad_data_gen/src/cad_data_gen/render_step_with_blender.py -- --config config.json

The caller prepares a JSON config. Blender is responsible only for importing the
STEP scene, normalizing it with the caller-provided bbox transform, optionally
adding foreground occluders, and rendering the requested camera fronts.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render STEP views in Blender.")
    parser.add_argument("--config", required=True, help="JSON render config path")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _objects_before() -> set[str]:
    return {obj.name for obj in bpy.context.scene.objects}


def _new_objects(before: set[str]) -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.name not in before]


def import_step(step_path: str) -> list[bpy.types.Object]:
    before = _objects_before()
    errors: list[str] = []
    operators = [
        ("bpy.ops.wm.step_import", lambda: bpy.ops.wm.step_import(filepath=step_path)),
        (
            "bpy.ops.import_scene.step",
            lambda: bpy.ops.import_scene.step(filepath=step_path),
        ),
        (
            "bpy.ops.import_mesh.step",
            lambda: bpy.ops.import_mesh.step(filepath=step_path),
        ),
    ]
    for name, op in operators:
        try:
            op()
            imported = _new_objects(before)
            mesh_objects = [obj for obj in imported if obj.type == "MESH"]
            if mesh_objects:
                return mesh_objects
            errors.append(f"{name}: imported no mesh objects")
        except Exception as exc:  # Blender import operators vary by add-on.
            errors.append(f"{name}: {exc}")
    joined = "; ".join(errors)
    raise RuntimeError(
        "No usable STEP import operator found in Blender. Install/enable a STEP "
        f"import add-on. Tried: {joined}"
    )


def convert_step_to_stl_with_freecad(step_path: str) -> str:
    out_file = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    out_path = out_file.name
    out_file.close()
    script_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    script_file.write(
        "\n".join(
            [
                "import sys",
                "import FreeCAD",
                "import Import",
                "import Mesh",
                "step_path = sys.argv[-2]",
                "out_path = sys.argv[-1]",
                "doc = FreeCAD.newDocument('step_to_stl')",
                "Import.insert(step_path, doc.Name)",
                "doc.recompute()",
                "objects = [obj for obj in doc.Objects if hasattr(obj, 'Shape')]",
                "if not objects:",
                "    raise RuntimeError('FreeCAD imported no shape objects')",
                "Mesh.export(objects, out_path)",
                "FreeCAD.closeDocument(doc.Name)",
            ]
        )
    )
    script_file.close()
    try:
        result = subprocess.run(
            ["freecadcmd", script_file.name, step_path, out_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("freecadcmd not found; install FreeCAD for STEP fallback") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit code {exc.returncode}"
        raise RuntimeError(f"FreeCAD STEP conversion failed: {detail}") from exc
    finally:
        Path(script_file.name).unlink(missing_ok=True)
    if not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
        raise RuntimeError(f"FreeCAD did not produce a valid STL: {out_path}")
    return out_path


def import_stl(stl_path: str) -> list[bpy.types.Object]:
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_stl")
    except Exception:
        pass
    before = _objects_before()
    try:
        bpy.ops.import_mesh.stl(filepath=stl_path)
    except Exception as exc:
        raise RuntimeError(f"Blender STL import failed: {exc}") from exc
    imported = _new_objects(before)
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Blender STL import created no mesh objects")
    return mesh_objects


def import_step_or_fallback(step_path: str) -> list[bpy.types.Object]:
    try:
        return import_step(step_path)
    except Exception as direct_error:
        stl_path = convert_step_to_stl_with_freecad(step_path)
        try:
            return import_stl(stl_path)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Direct STEP import failed ({direct_error}); "
                f"FreeCAD STL fallback failed ({fallback_error})"
            ) from fallback_error


def _set_node_input(node: bpy.types.Node, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def make_material(
    name: str,
    color: list[float],
    metallic: float = 0.0,
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        _set_node_input(bsdf, ("Base Color",), tuple(color))
        _set_node_input(bsdf, ("Metallic",), float(metallic))
        _set_node_input(bsdf, ("Roughness",), float(roughness))
    return material


def make_visualization_material(
    name: str,
    color: list[float],
    config: dict[str, Any],
) -> bpy.types.Material:
    return make_material(
        name,
        color,
        metallic=float(config.get("metallic", 0.9)),
        roughness=float(config.get("roughness", 0.7)),
    )


def make_occluder_material(name: str, color: list[float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    _set_node_input(emission, ("Color",), tuple(color))
    _set_node_input(emission, ("Strength",), 1.0)
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def normalize_objects(
    objects: list[bpy.types.Object],
    raw_center: list[float],
    scale: float,
) -> None:
    center = Vector((float(raw_center[0]), float(raw_center[1]), float(raw_center[2])))
    transform = (
        Matrix.Translation(Vector((0.5, 0.5, 0.5)))
        @ Matrix.Diagonal((float(scale), float(scale), float(scale), 1.0))
        @ Matrix.Translation(-center)
    )
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def add_occluder_cube(config: dict[str, Any]) -> None:
    occluder = config.get("occluder")
    if not occluder:
        return
    center = occluder["center_unit"]
    size = occluder["size_unit"]
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(float(center[0]), float(center[1]), float(center[2])),
    )
    cube = bpy.context.object
    cube.name = "OccluderRedBox"
    cube.dimensions = (float(size[0]), float(size[1]), float(size[2]))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    color = occluder.get("color", [0.86, 0.08, 0.08, 1.0])
    cube.data.materials.append(make_material("OccluderRed", color))


def view_basis(front: list[float]) -> tuple[Vector, Vector, Vector]:
    direction = Vector(front).normalized()
    up = Vector((0.0, 1.0, 0.0))
    right = up.cross(direction)
    if right.length < 1e-8:
        up = Vector((0.0, 0.0, 1.0))
        right = up.cross(direction)
    right.normalize()
    true_up = direction.cross(right)
    true_up.normalize()
    return right, true_up, direction


def _foreground_view_config(
    foreground: dict[str, Any],
    view_index: int,
) -> dict[str, Any]:
    views = foreground.get("views") or []
    if view_index < len(views):
        return dict(views[view_index])
    return {
        "offset_xy": foreground.get("offset_xy", [0.0, 0.0]),
        "size_xy": foreground.get("size_xy", [0.28, 0.28]),
        "angle_degrees": foreground.get("angle_degrees", 0.0),
        "shape": foreground.get("shape", "rectangle"),
    }


def _shape_unit_points(shape: str, segments: int = 32) -> list[tuple[float, float]]:
    if shape == "triangle":
        return [(0.0, 0.58), (-0.5, -0.29), (0.5, -0.29)]
    if shape == "ellipse":
        return [
            (0.5 * math.cos(2.0 * math.pi * i / segments), 0.5 * math.sin(2.0 * math.pi * i / segments))
            for i in range(segments)
        ]
    if shape == "hexagon":
        return [
            (0.5 * math.cos(2.0 * math.pi * i / 6.0), 0.5 * math.sin(2.0 * math.pi * i / 6.0))
            for i in range(6)
        ]
    return [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]


def add_foreground_occluder_for_camera(
    camera: bpy.types.Object,
    front: list[float],
    config: dict[str, Any],
    view_index: int,
) -> bpy.types.Object | None:
    foreground = config.get("foreground_occluder")
    if not foreground:
        return None

    view_config = _foreground_view_config(foreground, view_index)
    offset_xy = view_config.get("offset_xy", [0.0, 0.0])
    size_xy = view_config.get("size_xy", [0.28, 0.28])
    angle = math.radians(float(view_config.get("angle_degrees", 0.0)))
    shape = str(view_config.get("shape", foreground.get("shape", "rectangle")))
    color = foreground.get("color", [0.12, 0.15, 0.18, 1.0])
    depth = float(view_config.get("depth", foreground.get("depth", 0.45)))

    right, true_up, direction = view_basis(front)
    lookat = Vector((0.5, 0.5, 0.5))
    camera_to_model = lookat - camera.location
    distance = camera_to_model.length
    if distance < 1e-6:
        return None

    depth = min(max(depth, 0.05), 0.95)
    ortho_height = float(camera.data.ortho_scale)
    aspect = float(config["img_size"]) / float(config["img_size"])
    ortho_width = ortho_height * aspect

    center = camera.location + direction * distance * depth
    center += right * (float(offset_xy[0]) * ortho_width)
    center += true_up * (float(offset_xy[1]) * ortho_height)

    width = max(1e-4, float(size_xy[0]) * ortho_width)
    height = max(1e-4, float(size_xy[1]) * ortho_height)
    rect_right = right * math.cos(angle) + true_up * math.sin(angle)
    rect_up = -right * math.sin(angle) + true_up * math.cos(angle)

    verts = [
        center + rect_right * (local_x * width) + rect_up * (local_y * height)
        for local_x, local_y in _shape_unit_points(shape)
    ]
    mesh = bpy.data.meshes.new(f"ForegroundOccluderMesh_{view_index:03d}")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(range(len(verts)))])
    mesh.update()
    obj = bpy.data.objects.new(f"ForegroundOccluder_{view_index:03d}", mesh)
    bpy.context.collection.objects.link(obj)
    obj.visible_shadow = False
    obj.visible_diffuse = True
    obj.visible_glossy = True
    material = bpy.data.materials.get("OccluderForegroundMaterial")
    if material is None:
        material = make_occluder_material("OccluderForegroundMaterial", color)
    obj.data.materials.append(material)
    return obj


def setup_lighting() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (1.0, 1.0, 1.0)

    light_data = bpy.data.lights.new("KeyArea", type="AREA")
    light = bpy.data.objects.new("KeyArea", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (2.5, -3.0, 4.0)
    light.data.energy = 550.0
    light.data.size = 4.0

    fill_data = bpy.data.lights.new("FillArea", type="AREA")
    fill = bpy.data.objects.new("FillArea", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-3.0, 2.0, 3.0)
    fill.data.energy = 80.0
    fill.data.size = 5.0


def setup_visualization_lighting(config: dict[str, Any]) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    ambient = tuple(config.get("ambient_color", [0.4, 0.4, 0.4, 1.0]))
    world.color = ambient[:3]
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        _set_node_input(background, ("Color",), ambient)
        _set_node_input(background, ("Strength",), float(config.get("ambient_strength", 1.0)))

    rotation = (
        math.radians(45.0),
        math.radians(30.0),
        math.radians(-45.0),
    )
    bpy.ops.object.light_add(type="SUN", rotation=rotation)
    sun = bpy.context.object
    sun.name = "VisualizationSun"
    sun.data.energy = float(config.get("sun_strength", 10.0))
    if hasattr(sun.data, "angle"):
        sun.data.angle = float(config.get("sun_soft_size", 0.05))


def projected_span(front: list[float]) -> float:
    right, true_up, _ = view_basis(front)
    corners = [
        Vector((x, y, z)) - Vector((0.5, 0.5, 0.5))
        for x in (0.0, 1.0)
        for y in (0.0, 1.0)
        for z in (0.0, 1.0)
    ]
    xs = [corner.dot(right) for corner in corners]
    ys = [corner.dot(true_up) for corner in corners]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def setup_camera(
    front: list[float],
    camera_distance: float,
    ortho_fill: float = 0.76,
) -> bpy.types.Object:
    lookat = Vector((0.5, 0.5, 0.5))
    direction = Vector(front).normalized()
    distance = abs(float(camera_distance))
    if distance < 1.5:
        distance = 2.4
    location = lookat - direction * distance

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = (lookat - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = projected_span(front) / float(ortho_fill)
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    bpy.context.scene.camera = cam
    return cam


def setup_render(config: dict[str, Any]) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(config["img_size"])
    scene.render.resolution_y = int(config["img_size"])
    scene.render.resolution_percentage = 100
    render_style = config.get("render_style", "default")
    scene.render.film_transparent = bool(
        config.get("film_transparent", render_style == "visualization")
    )
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if scene.render.film_transparent else "RGB"
    if render_style == "visualization":
        desired_engine = "CYCLES"
        fallback_engines = ("CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH")
    else:
        desired_engine = config.get("engine", "BLENDER_EEVEE_NEXT")
        fallback_engines = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH")
    for engine in (desired_engine, *fallback_engines):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(config.get("samples", 64))
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass
    for look in ("Medium High Contrast", "None"):
        try:
            scene.view_settings.look = look
            break
        except TypeError:
            continue
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def render_views(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_distance = float(config.get("camera_distance", -0.9))
    ortho_fill = float(config.get("ortho_fill", 0.76))
    for i, front in enumerate(config["fronts"]):
        old_camera = bpy.context.scene.camera
        if old_camera is not None:
            bpy.data.objects.remove(old_camera, do_unlink=True)
        camera = setup_camera(front, camera_distance, ortho_fill=ortho_fill)
        foreground_obj = add_foreground_occluder_for_camera(camera, front, config, i)
        try:
            bpy.context.scene.render.filepath = str(output_dir / f"view_{i:03d}.png")
            bpy.ops.render.render(write_still=True)
        finally:
            if foreground_obj is not None:
                mesh = foreground_obj.data
                bpy.data.objects.remove(foreground_obj, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    clear_scene()
    if config.get("mesh_path"):
        objects = import_stl(config["mesh_path"])
    else:
        objects = import_step_or_fallback(config["step_path"])
    normalize_objects(objects, config["raw_center"], float(config["scale"]))

    render_style = config.get("render_style", "default")
    color = config.get("model_color", [0.82, 0.88, 0.94, 1.0])
    if render_style == "visualization":
        color = config.get("model_color", [67.0 / 255.0, 147.0 / 255.0, 233.0 / 255.0, 1.0])
        material = make_visualization_material("ModelMaterial", color, config)
    else:
        material = make_material("ModelMaterial", color)
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    if not config.get("foreground_occluder"):
        add_occluder_cube(config)
    if render_style == "visualization":
        setup_visualization_lighting(config)
    else:
        setup_lighting()
    setup_render(config)
    render_views(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[render_step_with_blender] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
