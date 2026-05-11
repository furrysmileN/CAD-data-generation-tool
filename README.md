# CAD 数据生成工具

这个文件夹是一个独立的数据生成小项目。它的核心用途是把 STEP CAD 文件处理成点云、多视角渲染图、遮挡增强样本、文本描述和重建结果。

当前已经包含一个可直接试跑的小样本集：`sample/steps/`，里面有 10 个 STEP 文件,来源于ABCdataset的前十个数据样本。

## 目录结构

```text
cad_data_gen/
├── README.md
├── CORE_PIPELINE.md
├── requirements.txt
├── sample/
│   ├── sample_manifest.jsonl
│   └── steps/                         # 10 个示例 STEP 文件
├── runs/                              # 默认输出目录，建议不要提交到仓库
└── src/cad_data_gen/
    ├── build_step_assets.py
    ├── build_occlusion_assets.py
    ├── render_step_with_blender.py
    ├── describe_step_with_deepseek.py
    ├── reconstruct_from_command_sequence.py
    ├── build_residual_steps.py
    ├── cadrecode2mesh.py
    ├── step_assets.py
    ├── pointcloud.py
    └── __init__.py
```

## 每个代码文件的作用

`src/cad_data_gen/build_step_assets.py`

基础 STEP 资产生成入口。读取 STEP/STP 文件，使用 CadQuery 转三角网格，采样点云，归一化到单位立方体，并生成多视角 PNG。默认有一个软件渲染器，也可以用 `--render-backend blender-step` 调用 Blender 做更好看的渲染。

`src/cad_data_gen/build_occlusion_assets.py`

遮挡增强主入口。基于 `build_step_assets.py` 生成的 `manifest.jsonl`，为每个样本生成遮挡变体，输出点云、渲染图、mask、label、audit 和 summary。支持 `cutout`、`occluder`、`mixed` 三种模式；当前推荐的前景遮挡渲染链路是 `--mode occluder --render-backend blender-step`。

`src/cad_data_gen/render_step_with_blender.py`

Blender 内部执行的渲染脚本，不建议直接用普通 Python 运行。`build_step_assets.py` 会把渲染配置写成 JSON，然后通过 `blender -b --python ...` 调用它。它负责场景清理、导入 STL/STEP、材质、灯光、相机、前景遮挡物和逐视角 PNG 输出。

`src/cad_data_gen/describe_step_with_deepseek.py`

STEP 文本描述生成脚本。它先在本地提取 STEP 文件的几何统计、实体类型和部分 STEP 文本片段，再调用 DeepSeek API 生成中文 CAD 描述。需要单独配置 DeepSeek API Key。

`src/cad_data_gen/reconstruct_from_command_sequence.py`

根据 `describe_step_with_deepseek.py` 生成的命令序列，使用 CadQuery 尝试重建近似 CAD 模型并导出 STEP。它是近似重建，不是精确的 Onshape 历史回放。

`src/cad_data_gen/build_residual_steps.py`

面向 CAD-Recode 数据的 residual-step 构建脚本。它逐步执行 CadQuery 程序前缀，生成当前步骤点云、目标点云和下一步操作标签。

`src/cad_data_gen/cadrecode2mesh.py`

把 CAD-Recode 里的 CadQuery Python 程序批量转成 STL mesh，并生成 `train.pkl` / `val.pkl` 标注。

`src/cad_data_gen/step_assets.py`

STEP 文件枚举、sample id 生成、CadQuery 读取 STEP 的轻量工具模块。主要供描述生成脚本复用。

`src/cad_data_gen/pointcloud.py`

点云采样工具，目前提供 `mesh_to_point_cloud()`。

`CORE_PIPELINE.md`

核心遮挡链路说明，记录 `build_occlusion_assets.py -> build_step_assets.py -> render_step_with_blender.py` 的调用关系。

## 环境要求

建议使用 Linux + Python 3.10 或 3.11。核心 Python 依赖在 `requirements.txt` 中：

```bash
pip install -r requirements.txt
```

主要 Python 包：

- `cadquery`：读取 STEP、布尔 cutout、CAD 重建。
- `trimesh`：mesh 处理和点云采样。
- `numpy`：数组计算。
- `pillow`：PNG 和 mask 处理。
- `tqdm`：进度条。
- `PyYAML`：解析 Onshape / FeatureScript 相关 YAML 元数据。

如果要用 Blender 高质量渲染，还需要系统安装：

- `blender`：命令行可执行文件需要能通过 `blender` 找到，或用 `--blender-bin` 指定路径。
- `freecadcmd`：可选 fallback，仅当 Blender 侧需要直接转换 STEP 时使用；当前主流程会优先在父 Python 进程用 CadQuery 转临时 STL。
- `xvfb`：无显示器服务器上运行 Blender 时可能需要。

Ubuntu 示例：

```bash
sudo apt-get update
sudo apt-get install -y blender xvfb
```

CadQuery 在有些机器上用 conda 更稳定：

```bash
conda create -n cad_data_gen python=3.10 -y
conda activate cad_data_gen
conda install -c conda-forge cadquery -y
pip install -r requirements.txt
```

## 快速试跑

所有命令都在 `cad_data_gen/` 目录下执行。

```bash
cd cad_data_gen
export PYTHONPATH="$PWD/src"
```

1. 生成 10 个示例 STEP 的基础资产：

```bash
python -m cad_data_gen.build_step_assets \
  --input-dir sample/steps \
  --output-dir runs/sample_first10_assets \
  --recursive \
  --limit 10 \
  --num-points 2048 \
  --num-views 6 \
  --img-size 256 \
  --render-backend blender-step \
  --blender-style visualization \
  --num-processes 1
```

输出内容：

- `runs/sample_first10_assets/manifest.jsonl`
- `runs/sample_first10_assets/failures.jsonl`
- `runs/sample_first10_assets/summary.json`
- `runs/sample_first10_assets/points/*.npz`
- `runs/sample_first10_assets/images/<sample_id>/view_*.png`

2. 基于基础资产生成遮挡增强样本：

```bash
python -m cad_data_gen.build_occlusion_assets \
  --manifest runs/sample_first10_assets/manifest.jsonl \
  --source-assets-dir runs/sample_first10_assets \
  --step-root sample/steps \
  --output-dir runs/sample_first10_occlusion \
  --variants-per-sample 1 \
  --target-dims point_cloud,image \
  --mode occluder \
  --num-views 6 \
  --img-size 256 \
  --render-backend blender-step \
  --blender-style visualization \
  --num-processes 1 \
  --limit 10 \
  --seed 0
```

输出内容：

- `runs/sample_first10_occlusion/manifest.jsonl`
- `runs/sample_first10_occlusion/failures.jsonl`
- `runs/sample_first10_occlusion/audit.jsonl`
- `runs/sample_first10_occlusion/summary.json`
- `runs/sample_first10_occlusion/points/*.npz`
- `runs/sample_first10_occlusion/images/<sample_id>__occ_000/view_*.png`
- `runs/sample_first10_occlusion/masks/<sample_id>__occ_000/view_*_mask.png`
- `runs/sample_first10_occlusion/labels/<sample_id>__occ_000.json`

本仓库当前已验证过这 10 个样本：`ok=10 failed=0`。

## 常用参数说明

`build_step_assets.py`

- `--input-dir`：STEP/STP 输入目录。
- `--output-dir`：基础资产输出目录。
- `--recursive`：递归查找 STEP。
- `--num-points`：每个样本点云数量。
- `--num-views`：渲染视角数量，支持常用的 1/2/4/6，也支持自动生成其他数量。
- `--render-backend trimesh`：使用内置软件渲染，依赖少，效果较简单。
- `--render-backend blender-step`：调用 Blender 渲染，效果更好。
- `--blender-style visualization`：使用更接近可视化展示的材质、灯光和透明背景设置。

`build_occlusion_assets.py`

- `--manifest`：基础资产清单，通常来自 `build_step_assets.py`。
- `--source-assets-dir`：基础资产根目录。
- `--step-root`：原始 STEP 根目录。
- `--target-dims`：生成哪些维度，常用 `point_cloud,image`，也可包含 `step`。
- `--mode cutout`：删除几何/点云区域，并可视化 cutout。
- `--mode occluder`：不改原始模型几何，在图像中添加前景遮挡物，并生成 mask。
- `--mode mixed`：随机混合 `cutout` 和 `occluder`。
- `--variants-per-sample`：每个源样本生成多少个遮挡变体。
- `--min-removed-ratio` / `--max-removed-ratio`：cutout 采样时控制点云删除比例。
- `--foreground-occluder-size-min` / `--foreground-occluder-size-max`：前景遮挡物大小范围。

## DeepSeek 文本描述

如果需要运行 `describe_step_with_deepseek.py`，需要 DeepSeek API Key。推荐放在：

```text
cad_data_gen/.secrets/deepseek_api_key
```

也可以查看脚本 `--help` 使用命令行参数传入。`.secrets/` 已加入 `.gitignore`，不要把密钥提交到公开仓库。

## 发布建议

建议发布时保留：

- `README.md`
- `CORE_PIPELINE.md`
- `requirements.txt`
- `sample/`
- `src/`
- `.gitignore`

建议不要提交：

- `runs/`
- `.secrets/`
- `__pycache__/`
- `*.pyc`
- `recovered_patches_from_transcript/`

如果用户没有 Blender，也可以先把命令里的 `--render-backend blender-step --blender-style visualization` 改成：

```bash
--render-backend trimesh
```

这样可以不依赖 Blender 完成基础功能试跑，但图片效果会更简单。
