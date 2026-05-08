# ONNX export for TensorRT (YOLO)

This repository is **focused on ONNX export for NVIDIA TensorRT** (and stacks that consume the same graphs, e.g. **DeepStream** with TensorRT). The maintained path is **`yolo export format=onnx`** plus **`onnx_output`** variants aligned with TensorRT plugins, standard **`onnx::NonMaxSuppression`**, or DeepStream-Yolo layouts.

**Not using TensorRT?** Use the **official Ultralytics export** instead of this fork: [Ultralytics — Export](https://docs.ultralytics.com/modes/export/). Upstream supports ONNX and many other targets with the standard maintenance and docs; this repo adds TensorRT-oriented ONNX graphs and related tooling.

The codebase derives from Ultralytics (YOLOv8 / YOLO11 / YOLO26, etc.). Scope here is **ONNX export** for deployment pipelines that build **`.engine`** or otherwise rely on these ONNX layouts.

## Installation

Use a dedicated Conda environment with Python 3.12:

```bash
conda create -n yolo_onnx_export python=3.12
conda activate yolo_onnx_export
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
| **`default`** | Same ONNX layout as upstream Ultralytics (baseline before TensorRT or generic ONNX) | `input_tensor_name` (default **`images`**) | `<stem>.onnx` |
| **`ds_yolo`** | [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo.git) | `input_tensor_name` (default **`images`**; set `input_tensor_name=input` for older samples) | `<stem>_ds_yolo.onnx`, `<stem>_ds_yolo.txt` |
| **`enms`** | TensorRT with **EfficientNMS_TRT** plugin | **`images`** via `input_tensor_name` | `<stem>_enms.onnx`, `<stem>_enms.txt` |
| **`onnx_nms`** | **`onnx::NonMaxSuppression`** → TensorRT **`INMSLayer`** ([API](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_n_m_s_layer.html)); graph **outputs** are still packed like `enms` (see below) | **`images`** via `input_tensor_name` | `<stem>_onnx_nms.onnx`, `<stem>_onnx_nms.txt` |
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

### TensorRT FP16 profiling — why latency differs across `onnx_output` variants

All `onnx_output` modes can be built to **FP16** TensorRT engines, but **end-to-end latency is not dominated by precision**: it is dominated by **how NMS appears in the ONNX graph** and how TensorRT maps it (fused many-op tail vs **EfficientNMS_TRT** vs **`NonMaxSuppression` / INMSLayer**). In representative **YOLO26n** runs, **default** and **`ds_yolo`** stayed near **~1.7 ms** mean inference (backbone-heavy; **`ds_yolo`** has **no NMS inside the engine**), while **`enms`** and **`onnx_nms`** landed near **~4.2 ms** because **one or few NMS-related layers** consumed a large share of per-layer time (e.g. **`EfficientNMS_TRT`** on the order of half of profiled layer time in one profile). **Choose the export for deployment constraints** (plugin API, DeepStream parser, portable ONNX NMS), not assuming “NMS in the graph” is automatically fastest.

**Full report (tables, artifacts, takeaways):** [docs/en/guides/tensorrt-fp16-onnx-nms-variant-profiling.md](docs/en/guides/tensorrt-fp16-onnx-nms-variant-profiling.md)

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

**Mapping (`enms` / `onnx_nms`):** `topk_all` → max boxes per class (**EfficientNMS** slots / ONNX **`MaxOutputBoxesPerClass`** feeding **`INMSLayer`**); `iou_thres` → IoU threshold; `conf_thres` → score threshold.

---

## CLI examples

**Upstream-style ONNX** (`onnx_output=default`, `output0`; matches stock Ultralytics ONNX):

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

## Inputs and outputs (tensor names and shapes)

Symbols used below:

| Symbol | Meaning |
|--------|---------|
| **B** | Batch size (often dynamic axis `batch` when `dynamic=True`) |
| **H**, **W** | Input height/width (match `imgsz`; often dynamic when `dynamic=True` in modes that export spatial axes) |
| **nc** | Number of classes |
| **nm** | Number of mask coefficients (segmentation proto channels) |
| **A** | Total anchor count (depends on `imgsz` and model strides; e.g. common YOLO layouts at 640² land near **8400**) |
| **reg_max** | DFL bins (often **16** on YOLOv8-style heads) |
| **K** | `topk_all` — max detection slots in packed / NMS ONNX outputs |
| **R** | `mask_resolution` — side length for packed mask grid (**see `default.yaml`**, often 56–160 depending on config) |

**Input (all modes):** one NCHW tensor named **`input_tensor_name`** (default **`images`**), shape **`(B, 3, H, W)`**. With `dynamic=True`, the default/enms/onnx_nms exporter marks **`batch`**, and usually **`height` / `width`** on the input (except where noted).

### ONNX graph outputs vs TensorRT engine I/O (what matters in production)

| | **ONNX** (`.onnx`) | **TensorRT** (`.engine`) |
|---|-------------------|---------------------------|
| **Use** | Netron, ONNX Runtime, cross-framework checks | **Deployment:** `trtexec`, TensorRT runtime, C++/Python bindings |
| **What you read** | `graph.input` / `graph.output` tensor **names** and **shapes** in the model | **Engine I/O tensor names** and **shapes** per **optimization profile** (dynamic) or fixed |
| **Names** | Defined by export | With the **ONNX parser** (`trt.OnnxParser` / `trtexec --onnx=...`), output names are **typically the same** as ONNX—use them as binding / `setTensorAddress` targets (TensorRT 8.x “bindings”; TensorRT 10.x `IEngine` / `IExecutionContext` tensor names). **Always verify** with `polygraphy inspect model` or a short TRT script on your build. |
| **Shapes** | Symbolic or static in the protobuf | At runtime, resolve **min/opt/max** for each dynamic axis in the **profile** you build with—shapes match ONNX **semantics**, not always the trace-only `imgsz`. |

**Important:** For **`onnx_nms`**, TensorRT may place an **`INMSLayer`** **inside** the network for the ONNX **`NonMaxSuppression`** node. The layer’s native outputs (**`SelectedIndices`**, **`NumOutputBoxes`**) are **not** the engine’s public I/O unless you export a graph that exposes only those—this fork **does not**. The **engine still exposes the same packed tensors** as ONNX: **`num_dets`**, **`det_boxes`**, **`det_scores`**, **`det_classes`**. Your app should allocate buffers for **those** names.

**Summary — public outputs (ONNX file = TensorRT bindings, same names):**

| `onnx_output` | Output tensor names (ONNX & typical TRT engine) | Notes |
|---------------|-----------------------------------------------|--------|
| **`default`** | `output0` [; `output1` if segment] | No TRT-specific NMS layer unless you add a plugin later. |
| **`enms`** | `num_dets`, `det_boxes`, `det_scores`, `det_classes` | **EfficientNMS_TRT** plugin inside the graph. |
| **`onnx_nms`** | **Same four as `enms`** | **`INMSLayer`** implements NMS **internally**; **packed `det_*` remain** the engine outputs. |
| **`ds_yolo`** | `output` | Single tensor; batch-only dynamic in this export path. |
| **`emulated_enms`** | Same four as `enms` [+ `det_masks` if segmentation packed] | E2E head + packed layout. |

Dtypes: ONNX is usually **FP32** in the file; a **FP16** or **INT8** TensorRT engine may still expose the same names with **reduced precision** tensors—set precision when building the engine.

---

### `onnx_output=default`

**File:** `<stem>.onnx`

| Tensor | Shape | Dtype (typical) | Notes |
|--------|-------|-----------------|-------|
| **`images`** (or custom) | `(B, 3, H, W)` | float32 | Input |
| **`output0`** | `(B, P, A)` | float32 | **Detect:** `P = 4·reg_max + nc` on DFL heads (e.g. `reg_max=16`, `nc=80` → **P=144**). **Segment:** `P = 4·reg_max + nc + nm` (proto coeffs + classes). **Pose / OBB:** `P` follows the head (extra keypoint or rotated-box channels). |
| **`output1`** | `(B, nm, H_p, W_p)` | float32 | **Segmentation only:** mask prototype grid (`H_p`, `W_p` are mask spatial dims, e.g. 160×160 in many configs). |

*Example (detection, illustrative):* `output0` might trace as `(1, 144, 8400)` for a 640×640 COCO-style DFL model — **inspect your ONNX** for exact `P` and `A`.

**Optional `nms=True`:** the graph wraps NMS; `output0` becomes a **post-NMS** tensor (shape differs, typically `(B, max_det, …)`). Not the usual TensorRT-pre ONNX path in this fork.

---

### `onnx_output=enms`

**Files:** `<stem>_enms.onnx`, `<stem>_enms.txt`

Detection only. NMS is implemented with the TensorRT **EfficientNMS_TRT** plugin (custom op), not `INMSLayer`.

| Tensor | Shape | Dtype (typical) | Notes |
|--------|-------|-----------------|-------|
| **`images`** (or custom) | `(B, 3, H, W)` | float32 | Input |
| **`num_dets`** | `(B, 1)` | int32 | Count per image (packed layout) |
| **`det_boxes`** | `(B, K, 4)` | float32 | **K = `topk_all`**. **`xyxy`** from the EfficientNMS plugin (corner coordinates). |
| **`det_scores`** | `(B, K)` | float32 | |
| **`det_classes`** | `(B, K)` | float32 | Class id per slot (often float in the graph) |

With `dynamic=True`, batch is dynamic on these tensors.

---

### `onnx_output=onnx_nms` (ONNX `NonMaxSuppression` → TensorRT `INMSLayer`)

**Files:** `<stem>_onnx_nms.onnx`, `<stem>_onnx_nms.txt`

Detection only. The graph uses the standard ONNX **`NonMaxSuppression`** operator (`center_point_box` / xywh-style boxes in this path). When you build a **TensorRT** engine from this ONNX, that node is mapped to NVIDIA’s **`nvinfer1::INMSLayer`**. The **layer semantics** match the [TensorRT 8.6.1 `INMSLayer` reference](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_n_m_s_layer.html):

**Inputs to the NMS layer (conceptual; inside the ONNX graph after the backbone):**

| Role | Shape (per NVIDIA INMSLayer doc) | Dtype |
|------|----------------------------------|-------|
| Boxes | `[batchSize, numInputBoundingBoxes, 4]` *or* per-class variant `[batchSize, numInputBoundingBoxes, numClasses, 4]` | float |
| Scores | `[batchSize, numInputBoundingBoxes, numClasses]` | float |
| MaxOutputBoxesPerClass | scalar (0-D) | int32 |
| IoUThreshold | scalar (optional) | float |
| ScoreThreshold | scalar (optional) | float |

**Native outputs of `INMSLayer` / ONNX `NonMaxSuppression` (before packing):**

| Output | Shape | Dtype | Meaning |
|--------|-------|-------|---------|
| **SelectedIndices** | `[NumOutputBoxes, 3]` | int32 | Each row is `(batchIndex, classIndex, boxIndex)` |
| **NumOutputBoxes** | scalar (0-D) | int32 | How many rows are valid in **SelectedIndices** |

**What this fork exposes as ONNX graph outputs:** the same **packed** tensors as **`enms`** so downstream code can stay aligned with the EfficientNMS layout—gather/scatter after NMS fills **`num_dets`**, **`det_boxes`**, **`det_scores`**, **`det_classes`**. Open the `.onnx` in Netron: you will see the **`NonMaxSuppression`** node and its **`selected_indices`**-style output wired into those packs.

**`trtexec` / random inputs:** With **`--loadInputs`** omitted, TensorRT uses random data; scores rarely pass **`conf_thres`**, so **`NonMaxSuppression`** often returns **zero** selected rows. Graphs that packed those results with **empty** intermediate tensors used to trigger TensorRT runtime errors (e.g. `IShuffleLayer` / `Expand`, reshape volume 0→1). The pack after NMS pads to **`max_det`** with sentinels and masks so **batch size 1** engines stay valid even with zero selections.

| Tensor (graph output) | Shape | Dtype (typical) | Notes |
|--------|-------|-----------------|-------|
| **`images`** (or custom) | `(B, 3, H, W)` | float32 | Input |
| **`num_dets`** | `(B, 1)` | int32 | Packed count (not the raw `NumOutputBoxes` scalar alone) |
| **`det_boxes`** | `(B, K, 4)` | float32 | **K = `topk_all`**. **`xyxy`** absolute corners (same convention as **`enms`** / EfficientNMS plugin output), not raw head **center-xywh**—the exporter applies **`xywh2xyxy`** after gather so eval pipelines match. |
| **`det_scores`** | `(B, K)` | float32 | |
| **`det_classes`** | `(B, K)` | float32 | |

With `dynamic=True`, batch is dynamic on these tensors.

---

### `onnx_output=ds_yolo`

**Files:** `<stem>_ds_yolo.onnx`, `<stem>_ds_yolo.txt`

Detection only. Single output; **no NMS** in the graph (DeepStream-Yolo consumes raw candidates).

| Tensor | Shape | Dtype (typical) | Notes |
|--------|-------|-----------------|-------|
| **`images`** (or custom) | `(B, 3, H, W)` | float32 | Input |
| **`output`** | `(B, A, 6)` | float32 | Per anchor: **4** box values + **1** score + **1** class (concatenated after `argmax` over classes). **A** matches total anchors (same order as upstream head concat). |

In this export path, **`dynamic=True`** sets dynamic axes on **batch only** for input/output (see `exporter.py`); height/width follow the traced **`imgsz`** unless you change the export code.

---

### `onnx_output=emulated_enms`

**Files:** `<stem>_emulated_enms.onnx`, `<stem>_emulated_enms.txt`

Requires an **end-to-end** detection head (YOLOv10 / YOLO11 / YOLO26-style). **Detection** uses the same four tensors as `enms` / `onnx_nms` (names and **K** slots). **Segmentation** packed export adds **`det_masks`**.

| Tensor | Shape | Dtype (typical) | Notes |
|--------|-------|-----------------|-------|
| **`images`** (or custom) | `(B, 3, H, W)` | float32 | Input |
| **`num_dets`** | `(B, 1)` | int32 | |
| **`det_boxes`** | `(B, K, 4)` | float32 | **K = `topk_all`** |
| **`det_scores`** | `(B, K)` | float32 | |
| **`det_classes`** | `(B, K)` | float32 | |
| **`det_masks`** | `(B, K, R²)` | float32 | **Segmentation only:** flattened **R×R** mask per slot (**R = `mask_resolution`**). |

Non–e2e two-stage models use the **`enms`** path (EfficientNMS wrapper), not `emulated_enms`.

---

### Segmentation summary

- **Standard two-head ONNX (recommended for generic ONNX):** **`onnx_output=default`** — `output0` + `output1` as in the first table.
- **Packed TRT-style seg** (plugins / ROIAlign): only via the **`emulated_enms`** / **`enms`** segmentation paths in the exporter, with **`det_masks`** and extra TRT plugin requirements — see `default.yaml` (`mask_resolution`, `pooler_scale`, …).

---

## License

See the `LICENSE` file in this repository (AGPL-3.0 in the upstream project). Review the terms before commercial use.
