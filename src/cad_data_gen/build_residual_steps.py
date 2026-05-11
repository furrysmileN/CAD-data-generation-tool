import os
import json
import traceback
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import numpy as np
import trimesh

from cad_data_gen.pointcloud import mesh_to_point_cloud


def _exec_prefix_worker(code_prefix, queue):
    local_env = {}
    try:
        exec(code_prefix, {}, local_env)
        if "r" not in local_env:
            queue.put({"ok": False, "error": "Variable `r` was not produced by prefix."})
            return
        compound = local_env["r"].val()
        vertices, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
        queue.put({"ok": True, "mesh": mesh})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def run_prefix_to_mesh(code_prefix, timeout_sec=3):
    queue = Queue()
    process = Process(target=_exec_prefix_worker, args=(code_prefix, queue))
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join()
        return None, "timeout"
    if queue.empty():
        return None, "no_output"
    result = queue.get()
    if not result["ok"]:
        return None, result.get("error", "unknown_error")
    return result["mesh"], None


def split_operations(py_string):
    lines = py_string.splitlines()
    kept = []
    for line in lines:
        if line.strip().startswith("#"):
            continue
        kept.append(line)
    return kept


def build_steps_for_sample(
    root_dir,
    item,
    out_point_dir,
    n_target_points,
    n_current_points,
):
    mesh_path = os.path.join(root_dir, item["mesh_path"])
    py_path = os.path.join(root_dir, item["py_path"])
    file_stem = os.path.splitext(os.path.basename(item["mesh_path"]))[0]

    with open(py_path, "r") as f:
        full_code = f.read()

    operations = split_operations(full_code)
    if len(operations) < 2:
        return []

    target_mesh = trimesh.load(mesh_path)
    target_points = mesh_to_point_cloud(target_mesh, n_target_points)
    target_points_path = os.path.join(out_point_dir, f"{file_stem}_target.npy")
    np.save(target_points_path, target_points.astype(np.float32))

    records = []
    prefix_lines = []
    for step_idx in range(1, len(operations)):
        prefix_lines.append(operations[step_idx - 1])
        code_prefix = "\n".join(prefix_lines)
        next_operation = operations[step_idx]

        current_mesh, error = run_prefix_to_mesh(code_prefix)
        current_path_rel = None
        if current_mesh is not None and len(current_mesh.faces) > 2:
            current_points = mesh_to_point_cloud(current_mesh, n_current_points)
            current_name = f"{file_stem}_step_{step_idx:03d}.npy"
            current_abs = os.path.join(out_point_dir, current_name)
            np.save(current_abs, current_points.astype(np.float32))
            current_path_rel = os.path.relpath(current_abs, root_dir)

        records.append(
            {
                "sample_id": f"{file_stem}_step_{step_idx:03d}",
                "mesh_path": item["mesh_path"],
                "py_path": item["py_path"],
                "target_pcd_path": os.path.relpath(target_points_path, root_dir),
                "current_pcd_path": current_path_rel,
                "previous_code": code_prefix,
                "next_operation": next_operation,
                "prefix_exec_error": error,
            }
        )
    return records


def run(
    root_dir,
    split,
    output_dir,
    n_target_points,
    n_current_points,
):
    pkl_path = os.path.join(root_dir, f"{split}.pkl")
    with open(pkl_path, "rb") as f:
        annotations = __import__("pickle").load(f)

    split_out_dir = os.path.join(output_dir, split)
    out_point_dir = os.path.join(split_out_dir, "points")
    os.makedirs(out_point_dir, exist_ok=True)
    jsonl_path = os.path.join(split_out_dir, f"{split}.jsonl")

    all_records = []
    for item in annotations:
        all_records.extend(
            build_steps_for_sample(
                root_dir=root_dir,
                item=item,
                out_point_dir=out_point_dir,
                n_target_points=n_target_points,
                n_current_points=n_current_points,
            )
        )

    with open(jsonl_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Saved {len(all_records)} residual-step records to {jsonl_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="./data/cad-recode-v1.5")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="./data/cad-recode-v1.5/residual_steps")
    parser.add_argument("--n-target-points", type=int, default=1024)
    parser.add_argument("--n-current-points", type=int, default=1024)
    args = parser.parse_args()

    run(
        root_dir=args.root_dir,
        split=args.split,
        output_dir=args.output_dir,
        n_target_points=args.n_target_points,
        n_current_points=args.n_current_points,
    )
