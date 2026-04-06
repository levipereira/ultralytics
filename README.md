# YOLO model export (ONNX / TensorRT)

This repository is **for exporting trained YOLO weights** (for example `.pt`) to deployment formats only — in particular **ONNX** aimed at **NVIDIA TensorRT** and integrations such as **DeepStream**. It is not a replacement for full training or inference documentation; the scope here is the **export** pipeline.

The codebase derives from the Ultralytics project (YOLOv8 / YOLO11 / YOLO26, etc.); the intended use in this fork is **export only**.

## Installation

Use a dedicated Conda environment with Python 3.12:

```bash
conda create -n yolo_export_trt python=3.12
conda activate yolo_export_trt
```

From the repository root (with the code cloned), install the package with the export extra:

```bash
pip install ".[export]"
```

The `[export]` extra pulls ONNX-related dependencies (`onnx`, `onnxslim`, `onnxscript`, `onnx-graphsurgeon`, and others) needed for `format=onnx`, `onnx_trt`, and related tooling.

Install **PyTorch** for your GPU/CPU (see [pytorch.org](https://pytorch.org/get-started/locally/)) before or after the step above, depending on your platform.

## Export commands

### CLI (`yolo`)

Example — ONNX with TensorRT-oriented packaging (post-processing depends on the head type; see below):

```bash
yolo export model=path/to/model.pt format=onnx_trt dynamic=True topk_all=300
```

Standard ONNX export (without the `onnx_trt` pipeline):

```bash
yolo export model=path/to/model.pt format=onnx opset=20 simplify=True
```

### `onnx_trt.py` (repository root)

Thin wrapper around `YOLO(...).export(format="onnx_trt", ...)` with fixed `dynamic=True`:

```bash
python onnx_trt.py -w path/to/model.pt --topk_all 300
```

Equivalent `yolo export` arguments include `topk_all=`, `iou_thres=`, `conf_thres=`, `class_agnostic`, `pooler_scale`, `sampling_ratio`, and `mask_resolution` (see `default.yaml` and export docs).

#### `onnx_trt.py` arguments (important)

These map directly to the exporter; choose them to match how you will build the TensorRT engine and parse outputs.

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `-w` / `--weights` | str | *(required)* | Path to the `.pt` checkpoint. |
| `--topk_all` | int | `100` | Maximum number of detections per image (slot count `K` in ONNX outputs). For end-to-end (NMS-free) heads, this sets the model’s internal top‑`K`; for YOLOv8-style exports it is also the EfficientNMS `max_output_boxes`. |
| `--iou_thres` | float | `0.45` | IoU threshold for **EfficientNMS_TRT** (non–end-to-end detection and segmentation paths). Ignored for end-to-end detection heads where NMS is already inside the network. |
| `--conf_thres` | float | `0.25` | Score threshold for **EfficientNMS_TRT** (non–end-to-end paths). Ignored for end-to-end detection heads. |
| `--class_agnostic` | flag | off | If set, uses class-agnostic NMS in the TRT plugin path. Requires TensorRT **8.6+** for the corresponding plugin variant. |
| `--pooler_scale` | float | `0.25` | **Segmentation only**: `spatial_scale` for ROIAlign in the export graph. |
| `--sampling_ratio` | int | `0` | **Segmentation only**: ROIAlign sampling ratio (`0` = default). |
| `--mask_resolution` | int | `160` | **Segmentation only**: height/width of mask crops before upsample (output mask vector length is `mask_resolution²`). |

**Note — NMS-free / end-to-end models (e.g. YOLO26, YOLOv10, YOLOv11 with built-in top‑k):** only **`--topk_all`** (and `-w`) meaningfully change the exported detection graph. `--iou_thres`, `--conf_thres`, and `--class_agnostic` apply to exports that insert **EfficientNMS_TRT** (YOLOv8-style raw outputs). Segmentation models still use **`pooler_scale`**, **`sampling_ratio`**, and **`mask_resolution`** together with the NMS-related args.

### `export_yolo26.py` (repository root, optional)

Alternative ONNX export path aimed at **DeepStream** (adapted head forward, simplified outputs). Example:

```bash
python export_yolo26.py -w path/to/model.pt -s 640 --opset 17
```

This calls `torch.onnx.export` on a model prepared for that flow; the graph may differ from Ultralytics `format=onnx_trt`.

## How exports are produced

### `format=onnx_trt` (and `onnx_trt.py`)

- **Output files** (by default next to the `.pt`):  
  - `<weights-stem>-trt.onnx` — ONNX graph for TensorRT;  
  - `<weights-stem>-trt.txt` — label file (one class name per line).

- **End-to-end heads (NMS / top-k inside the model)** — e.g. **YOLO26**, **YOLOv10**, **YOLOv11** with e2e outputs: the network already returns post-processed detections (top-k). Export **does not** insert the **EfficientNMS_TRT** plugin; the ONNX graph uses the standard ONNX domain and runs on TensorRT without that custom plugin.

- **YOLOv8-style heads (non-e2e)** — raw boxes and scores: export wraps the output with **EfficientNMS_TRT** in the ONNX graph for NVIDIA TensorRT plugins.

- **Segmentation** — separate path (masks, ROI, etc.); TRT plugins follow the exporter implementation.

- Optional post-processing: ONNX simplification (`onnxsim` when available) and graph **cleanup** with **ONNX GraphSurgeon** when installed.

### ONNX output tensors (`onnx_trt` detection)

The exported model has a single input named **`images`**: `NCHW`, typically `[batch, 3, H, W]` (dynamic batch and spatial sizes when `dynamic=True`).

**Detection** models expose **four** ONNX outputs (names fixed in the graph):

| Output name | Typical shape | Description |
| ----------- | -------------- | ----------- |
| `num_dets` | `[batch, 1]` | Integer count of valid detections per image (exact semantics depend on the wrapper / plugin). |
| `det_boxes` | `[batch, K, 4]` | `K = topk_all`. Four box values per slot. For **end-to-end** heads, values follow the model head (e.g. `x1,y1,x2,y2`). For **EfficientNMS_TRT**, coordinates follow the **TensorRT EfficientNMS** convention (center-style `xywh` feeding the plugin in the exporter). |
| `det_scores` | `[batch, K]` | Confidence or class score for each slot. |
| `det_classes` | `[batch, K]` | Class index per slot (floating-point in the traced graph; cast as needed). |

Pad or filter using `num_dets` and scores as in your TensorRT / deployment sample.

**Segmentation** adds a fifth output:

| Output name | Typical shape | Description |
| ----------- | -------------- | ----------- |
| `det_masks` | `[batch, K, mask_resolution²]` | Per-detection mask coefficients or rasterized mask strip (as produced by the ROIAlign + matmul path in the exporter). |

Use the same `K` and `mask_resolution` you passed at export (`--topk_all`, `--mask_resolution`).

### `format=onnx`

Standard ONNX for ONNX Runtime, OpenCV DNN, etc., without the `onnx_trt`-specific packaging.

## License

See the `LICENSE` file in this repository (AGPL-3.0 in the upstream project). Review the terms before commercial use.
