# YOLO model export (ONNX / TensorRT)

This repository is **for exporting trained YOLO weights** (for example `.pt`) to deployment formats only — in particular **ONNX** aimed at **NVIDIA TensorRT**, **ONNX Runtime**, and integrations such as **DeepStream**. It is not a replacement for full training or inference documentation; the scope here is the **export** pipeline.

The codebase derives from the Ultralytics project (YOLOv8 / YOLO11 / YOLO26, etc.); the intended use in this fork is **export only**.

## All supported export formats (`yolo export format=…`)

These entries mirror `export_formats()` in `ultralytics/engine/exporter.py`. **GPU** indicates whether the exporter can use CUDA for that format; some formats still produce CPU-compatible artifacts.

| Format | `format=` | Output suffix / path | CPU | GPU | Notable CLI args (subset) |
|--------|-----------|----------------------|-----|-----|---------------------------|
| PyTorch | *(already `.pt`)* | `.pt` | ✓ | ✓ | — |
| TorchScript | `torchscript` | `.torchscript` | ✓ | ✓ | `batch`, `optimize`, `half`, `nms`, `dynamic` |
| **ONNX** | **`onnx`** | **`.onnx`** | ✓ | ✓ | **`batch`, `dynamic`, `half`, `opset`, `simplify`, `nms`, `etnms`, `onnx_output`, `input_tensor_name`** |
| OpenVINO | `openvino` | `_openvino_model` dir | ✓ | — | `batch`, `dynamic`, `half`, `int8`, `nms`, `fraction` |
| TensorRT | `engine` | `.engine` | — | ✓ | `batch`, `dynamic`, `half`, `int8`, `simplify`, `nms`, `fraction` |
| CoreML | `coreml` | `.mlpackage` | ✓ | — | `batch`, `dynamic`, `half`, `int8`, `nms` |
| TensorFlow SavedModel | `saved_model` | `_saved_model` | ✓ | ✓ | `batch`, `int8`, `keras`, `nms` |
| TensorFlow GraphDef | `pb` | `.pb` | ✓ | ✓ | `batch` |
| TensorFlow Lite | `tflite` | `.tflite` | ✓ | — | `batch`, `half`, `int8`, `nms`, `fraction` |
| Edge TPU | `edgetpu` | `_edgetpu.tflite` | ✓ | — | — |
| TensorFlow.js | `tfjs` | `_web_model` | ✓ | — | `batch`, `half`, `int8`, `nms` |
| PaddlePaddle | `paddle` | `_paddle_model` | ✓ | ✓ | `batch` |
| MNN | `mnn` | `.mnn` | ✓ | ✓ | `batch`, `half`, `int8` |
| NCNN | `ncnn` | `_ncnn_model` | ✓ | ✓ | `batch`, `half` |
| IMX | `imx` | `_imx_model` | ✓ | ✓ | `int8`, `fraction`, `nms` |
| RKNN | `rknn` | `_rknn_model` | — | — | `batch`, `name` |
| ExecuTorch | `executorch` | `_executorch_model` | ✓ | — | `batch` |
| Axelera AI | `axelera` | `_axelera_model` | — | — | `batch`, `int8`, `fraction`, `data` |

**ONNX-specific options** (`onnx_output`, `input_tensor_name`, `etnms`, etc.) apply **only** when `format=onnx`; other formats ignore them (with a warning).

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

The `[export]` extra pulls ONNX-related dependencies (`onnx`, `onnxslim`, `onnxscript`, `onnx-graphsurgeon`, and others) needed for `format=onnx` and all `onnx_output` variants below.

Install **PyTorch** for your GPU/CPU (see [pytorch.org](https://pytorch.org/get-started/locally/)) before or after the step above, depending on your platform.

---

## ONNX export overview (`format=onnx`)

All paths use **`yolo export ... format=onnx`**. The layout is selected with **`onnx_output`** (see `ultralytics/cfg/default.yaml`).

| `onnx_output` | Typical use | Graph input name | Artifact names |
|---------------|-------------|------------------|----------------|
| **`default`** | ONNX Runtime, OpenCV DNN, generic tooling | `input_tensor_name` (default **`images`**) | `<stem>.onnx` |
| **`ds_yolo`** | DeepStream-Yolo utils (`models/DeepStream-Yolo/utils`) | `input_tensor_name` (default **`images`**; set `input_tensor_name=input` for older samples) | `<stem>_ds_yolo.onnx`, `<stem>_ds_yolo.txt` |
| **`enms`** | TensorRT with **EfficientNMS_TRT** plugin | **`images`** via `input_tensor_name` | `<stem>_enms.onnx`, `<stem>_enms.txt` |
| **`onnx_nms`** | TensorRT **INMSLayer** / portable **`onnx::NonMaxSuppression`** | **`images`** via `input_tensor_name` | `<stem>_onnx_nms.onnx`, `<stem>_onnx_nms.txt` |
| **`emulated_enms`** | Packed four outputs on **end-to-end** detection heads only | **`images`** via `input_tensor_name` | `<stem>_emulated_enms.onnx`, `<stem>_emulated_enms.txt` |

- Non-`default` modes also write a **`.txt`** label file next to the ONNX.
- **`etnms=True`** is a legacy alias for **`onnx_output=enms`** (do not combine with conflicting `onnx_output`).

### Implicit `end2end=False`

For **`onnx_output`** ∈ **`enms`**, **`onnx_nms`**, or **`ds_yolo`**, the exporter sets **`end2end=False`** automatically (raw two-stage / DeepStream-style graphs). You do not need to pass `end2end=False` on the CLI.

### `dynamic=True`, `imgsz`, and `batch`

When **`dynamic=True`**, **`imgsz`** and **`batch`** are **trace / example dimensions** only: they shape the dummy tensor used during export. **Runtime** batch and height/width are **not** fixed to those values; the ONNX graph exposes dynamic axes where applicable. With **`dynamic=False`**, the traced shapes match `imgsz` and `batch` more tightly.

### ONNX input tensor name

| Parameter | Default | Meaning |
|-----------|---------|---------|
| **`input_tensor_name`** | `images` | ONNX graph **input** tensor name. Override if your runtime or DeepStream config expects another name (e.g. `input_tensor_name=input`). |

The **output** tensor name for **`ds_yolo`** remains **`output`** (DeepStream-Yolo convention).

---

## Parameter matrix (ONNX export)

Cross-cutting parameters (all modes unless noted):

| Parameter | Applies when | Role |
|-----------|----------------|------|
| **`model`** | always | Path to `.pt` weights |
| **`format=onnx`** | always | ONNX export |
| **`onnx_output`** | always | `default` \| `ds_yolo` \| `enms` \| `onnx_nms` \| `emulated_enms` |
| **`imgsz`** | always | Export / trace image size (see **dynamic** note above) |
| **`batch`** | always | Trace minibatch size (see **dynamic** note above) |
| **`dynamic`** | always | If `True`, dynamic axes where supported; `imgsz`/`batch` are example dims |
| **`opset`** | optional | ONNX opset (default: auto from ONNX + torch) |
| **`simplify`** | optional | Graph slimming; may be forced **`False`** for `enms` / `onnx_nms` to keep custom/NMS nodes |
| **`half`** | optional | FP16 export where supported |
| **`input_tensor_name`** | optional | ONNX input name (default `images`) |
| **`end2end`** | varies | **Forced `False`** (with `model.end2end=False`) for `enms`, `onnx_nms`, `ds_yolo`. For **`emulated_enms`**, the checkpoint must already be an **e2e detection head** (YOLOv10 / YOLO11 / YOLO26-style); the exporter validates this and does not use the same two-stage path as `enms`. |

Mode-specific parameters:

| Parameter | `default` | `ds_yolo` | `enms` | `onnx_nms` | `emulated_enms` |
|-----------|-----------|-----------|--------|------------|-----------------|
| **`topk_all`** | — | — | max boxes / plugin slots | maps to ONNX **max_output_boxes_per_class** | internal top‑K / slots |
| **`iou_thres`** | — | — | EfficientNMS / TRT | ONNX **NonMaxSuppression** IoU | packed e2e path |
| **`conf_thres`** | — | — | score threshold | ONNX **score_threshold** | packed e2e path |
| **`class_agnostic`** | — | — | `enms` TRT path | — | — |
| **`nms`** | optional fused NMS | forced off if set | incompatible → off | incompatible → off | — |
| **Task** | detect / seg / pose / obb / classify | **detect only** | **detect only** | **detect only** | **detect only**, **e2e head** |
| **Segmentation extras** (`pooler_scale`, …) | `default` seg + packed TRT seg if used | — | packed seg with plugins | — | — |

**Mapping (`enms` / `onnx_nms`):** `topk_all` → max detections per class (plugin / ONNX NMS input); `iou_thres` → IoU threshold; `conf_thres` → score threshold.

---

## CLI examples

**Standard Ultralytics ONNX** (`output0`, full upstream flags):

```bash
yolo export model=yolo26n.pt format=onnx onnx_output=default imgsz=640 dynamic=True simplify=True
```

**DeepStream-Yolo** (single `output`; no NMS in graph):

```bash
yolo export model=yolo26n.pt format=onnx onnx_output=ds_yolo imgsz=640 dynamic=True input_tensor_name=input
```

**EfficientNMS_TRT** (TensorRT plugin):

```bash
yolo export model=yolo26n.pt format=onnx onnx_output=enms imgsz=640 dynamic=True topk_all=100 iou_thres=0.45 conf_thres=0.25
```

**Standard ONNX NonMaxSuppression** (TensorRT INMSLayer–friendly, portable NMS op):

```bash
yolo export model=yolo26n.pt format=onnx onnx_output=onnx_nms imgsz=640 dynamic=True topk_all=100 iou_thres=0.45 conf_thres=0.25 input_tensor_name=images
```

**Packed four outputs on e2e heads** (YOLOv10 / YOLO11 / YOLO26-style):

```bash
yolo export model=yolo26n.pt format=onnx onnx_output=emulated_enms imgsz=640 dynamic=True topk_all=300
```

### `onnx_trt.py` (repository root)

Wrapper around **`onnx_output=enms`** with `dynamic=True`. For e2e weights and packed export, use **`onnx_output=emulated_enms`** or **`onnx_nms`** via CLI/Python.

```bash
python onnx_trt.py -w path/to/model.pt --topk_all 300
```

Arguments mirror **`topk_all`**, **`iou_thres`**, **`conf_thres`**, **`class_agnostic`**, and segmentation-related pooler args where applicable (see `default.yaml`).

### `export_yolo26.py` (repository root, optional)

Alternative script for a DeepStream-oriented graph; may differ from `yolo export format=onnx onnx_output=ds_yolo`.

```bash
python export_yolo26.py -w path/to/model.pt -s 640 --opset 17
```

---

## Output layouts (reference)

### `onnx_output=default`

- **Files:** `<stem>.onnx`
- **Outputs:** `output0` (and `output1` for segmentation), same as upstream Ultralytics.

### `onnx_output` ∈ { `enms`, `onnx_nms` }

- **Files:** `<stem>_<mode>.onnx`, `<stem>_<mode>.txt`
- **Outputs:** `num_dets`, `det_boxes`, `det_scores`, `det_classes` (detection).
- **Input:** NCHW tensor named per **`input_tensor_name`** (default **`images`**).

### `onnx_output=ds_yolo`

- **Files:** `<stem>_ds_yolo.onnx`, `<stem>_ds_yolo.txt`
- **I/O:** Input name = **`input_tensor_name`**; output name **`output`** (fixed). No NMS in the graph.

### `onnx_output=emulated_enms`

- **Files:** `<stem>_emulated_enms.onnx`, `<stem>_emulated_enms.txt`
- **Outputs:** same four tensors as `enms`/`onnx_nms` naming; **only** valid for **end-to-end** detection heads (exporter validates).
- Use a realistic **`imgsz`** (e.g. 640); very small sizes can break top‑K / packed export.

### Segmentation

Use **`onnx_output=default`** for standard two-head ONNX. The specialized non-default modes above target **detection** deployment layouts unless documented otherwise for a specific packed segmentation path.

---

## License

See the `LICENSE` file in this repository (AGPL-3.0 in the upstream project). Review the terms before commercial use.
