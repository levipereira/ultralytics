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

Thin wrapper around `YOLO(...).export(format="onnx_trt", ...)`:

```bash
python onnx_trt.py -w path/to/model.pt --topk_all 300
```

Useful flags include `--iou_thres`, `--conf_thres`, `--class_agnostic`, and for segmentation `--mask_resolution`, etc. (see the script).

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

### `format=onnx`

Standard ONNX for ONNX Runtime, OpenCV DNN, etc., without the `onnx_trt`-specific packaging.

## License

See the `LICENSE` file in this repository (AGPL-3.0 in the upstream project). Review the terms before commercial use.
