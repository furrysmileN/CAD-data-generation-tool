# Core CAD Data Generation Pipeline

The occlusion-generation code path is preserved in this folder.

## Entry Points

- `src/cad_data_gen/build_occlusion_assets.py`
  - Main occlusion variant generator.
  - Produces derived point clouds, rendered images, masks, labels, audit logs, and optional STEP cutouts.
- `src/cad_data_gen/build_step_assets.py`
  - Shared STEP asset helpers used by occlusion generation.
  - Provides STEP loading, point sampling, unit-cube normalization, camera view fronts, CPU PNG rendering, and the Blender invocation wrapper.
- `src/cad_data_gen/render_step_with_blender.py`
  - Blender-side renderer called by `build_step_assets.py` when `--render-backend blender-step` is used.

## Local Dependency Chain

```text
build_occlusion_assets.py
  -> build_step_assets.py
       -> render_step_with_blender.py  (only for blender-step rendering)
       -> CadQuery / trimesh / PIL
  -> CadQuery / trimesh / PIL
```

Legacy paths under `cadrille/data/` are compatibility wrappers that forward to these package modules.
