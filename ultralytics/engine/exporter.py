# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export a YOLO PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit.

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolo26n.pt
TorchScript             | `torchscript`             | yolo26n.torchscript
ONNX                    | `onnx`                    | yolo26n.onnx
OpenVINO                | `openvino`                | yolo26n_openvino_model/
TensorRT                | `engine`                  | yolo26n.engine
CoreML                  | `coreml`                  | yolo26n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolo26n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolo26n.pb
TensorFlow Lite         | `tflite`                  | yolo26n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolo26n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolo26n_web_model/
PaddlePaddle            | `paddle`                  | yolo26n_paddle_model/
MNN                     | `mnn`                     | yolo26n.mnn
NCNN                    | `ncnn`                    | yolo26n_ncnn_model/
IMX                     | `imx`                     | yolo26n_imx_model/
RKNN                    | `rknn`                    | yolo26n_rknn_model/
ExecuTorch              | `executorch`              | yolo26n_executorch_model/
Axelera AI              | `axelera`                 | yolo26n_axelera_model/

Requirements:
    $ pip install "ultralytics[export]"

Python:
    from ultralytics import YOLO
    model = YOLO('yolo26n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolo26n.pt format=onnx

Inference:
    $ yolo predict model=yolo26n.pt                 # PyTorch
                         yolo26n.torchscript        # TorchScript
                         yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                         yolo26n_openvino_model     # OpenVINO
                         yolo26n.engine             # TensorRT
                         yolo26n.mlpackage          # CoreML (macOS-only)
                         yolo26n_saved_model        # TensorFlow SavedModel
                         yolo26n.pb                 # TensorFlow GraphDef
                         yolo26n.tflite             # TensorFlow Lite
                         yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolo26n_paddle_model       # PaddlePaddle
                         yolo26n.mnn                # MNN
                         yolo26n_ncnn_model         # NCNN
                         yolo26n_imx_model          # IMX
                         yolo26n_rknn_model         # RKNN
                         yolo26n_executorch_model   # ExecuTorch
                         yolo26n_axelera_model      # Axelera AI

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolo26n_web_model public/yolo26n_web_model
    $ npm start
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics import __version__
from ultralytics.cfg import TASK2CALIBRATIONDATA, TASK2DATA, get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.nn.modules import C2f, Classify, Detect, RTDETRDecoder, Segment26, v10Detect
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, WorldModel
from ultralytics.utils import (
    ARM64,
    DEFAULT_CFG,
    IS_DEBIAN_BOOKWORM,
    IS_DEBIAN_TRIXIE,
    IS_DOCKER,
    IS_RASPBERRYPI,
    IS_UBUNTU,
    LINUX,
    LOGGER,
    MACOS,
    MACOS_VERSION,
    RKNN_CHIPS,
    SETTINGS,
    TORCH_VERSION,
    WINDOWS,
    YAML,
    callbacks,
    colorstr,
    get_default_args,
    is_dgx,
    is_jetson,
)
from ultralytics.utils.checks import (
    IS_PYTHON_MINIMUM_3_9,
    check_apt_requirements,
    check_executorch_requirements,
    check_imgsz,
    check_requirements,
    check_tensorrt,
    check_version,
    is_intel,
    is_sudo_available,
)
from ultralytics.utils.files import file_size
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.patches import arange_patch
from ultralytics.utils.torch_utils import (
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_3,
    TORCH_2_4,
    TORCH_2_8,
    TORCH_2_9,
    select_device,
)


def export_formats():
    """Return a dictionary of Ultralytics YOLO export formats."""
    x = [
        ["PyTorch", "-", ".pt", True, True, []],
        ["TorchScript", "torchscript", ".torchscript", True, True, ["batch", "optimize", "half", "nms", "dynamic"]],
        [
            "ONNX",
            "onnx",
            ".onnx",
            True,
            True,
            [
                "batch",
                "dynamic",
                "half",
                "opset",
                "simplify",
                "nms",
                "etnms",
                "onnx_output",
                "input_tensor_name",
            ],
        ],
        [
            "OpenVINO",
            "openvino",
            "_openvino_model",
            True,
            False,
            ["batch", "dynamic", "half", "int8", "nms", "fraction"],
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "dynamic", "half", "int8", "simplify", "nms", "fraction"],
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "half", "int8", "nms"]],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True, ["batch", "int8", "keras", "nms"]],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"]],
        ["TensorFlow Lite", "tflite", ".tflite", True, False, ["batch", "half", "int8", "nms", "fraction"]],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False, []],
        ["TensorFlow.js", "tfjs", "_web_model", True, False, ["batch", "half", "int8", "nms"]],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"]],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "half", "int8"]],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "half"]],
        ["IMX", "imx", "_imx_model", True, True, ["int8", "fraction", "nms"]],
        ["RKNN", "rknn", "_rknn_model", False, False, ["batch", "name"]],
        ["ExecuTorch", "executorch", "_executorch_model", True, False, ["batch"]],
        ["Axelera AI", "axelera", "_axelera_model", False, False, ["batch", "int8", "fraction", "data"]],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments"], zip(*x)))


ONNX_OUTPUT_MODES = frozenset({"default", "ds_yolo", "enms", "emulated_enms", "onnx_nms"})


def is_v10_style_e2e_export(model) -> bool:
    """Return True for YOLOv10-style / fused end-to-end detection heads (top-k inside the graph)."""
    v10 = False
    for m in model.modules():
        if isinstance(m, v10Detect):
            v10 = True
            break
    if not v10 and isinstance(model, DetectionModel):
        head = model.model[-1]
        if type(head) is Detect and getattr(head, "end2end", False):
            v10 = True
    return v10


def validate_args(format, passed_args, valid_args):
    """Validate arguments based on the export format.

    Args:
        format (str): The export format.
        passed_args (SimpleNamespace): The arguments used during export.
        valid_args (list): List of valid arguments for the format.

    Raises:
        AssertionError: If an unsupported argument is used, or if the format lacks supported argument listings.
    """
    export_args = ["half", "int8", "dynamic", "keras", "nms", "etnms", "batch", "fraction"]

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}  # exporter defaults
    default_args = get_cfg(DEFAULT_CFG, custom)
    for arg in export_args:
        not_default = getattr(passed_args, arg, None) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)  # exported file/dir or tuple of (file/dir, *)
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.0, "0.0 MB output model size"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}")
            raise e

    return outer_func


class Exporter:
    """A class for exporting YOLO models to various formats.

    This class provides functionality to export YOLO models to different formats including ONNX, TensorRT, CoreML,
    TensorFlow, and others. It handles format validation, device selection, model preparation, and the actual export
    process for each supported format.

    Attributes:
        args (SimpleNamespace): Configuration arguments for the exporter.
        callbacks (dict): Dictionary of callback functions for different export events.
        im (torch.Tensor): Input tensor for model inference during export.
        model (torch.nn.Module): The YOLO model to be exported.
        file (Path): Path to the model file being exported.
        output_shape (tuple): Shape of the model output tensor(s).
        pretty_name (str): Formatted model name for display purposes.
        metadata (dict): Model metadata including description, author, version, etc.
        device (torch.device): Device on which the model is loaded.
        imgsz (list): Input image size for the model.

    Methods:
        __call__: Main export method that handles the export process.
        get_int8_calibration_dataloader: Build dataloader for INT8 calibration.
        export_torchscript: Export model to TorchScript format.
        export_onnx: Export model to ONNX format.
        export_openvino: Export model to OpenVINO format.
        export_paddle: Export model to PaddlePaddle format.
        export_mnn: Export model to MNN format.
        export_ncnn: Export model to NCNN format.
        export_coreml: Export model to CoreML format.
        export_engine: Export model to TensorRT format.
        export_saved_model: Export model to TensorFlow SavedModel format.
        export_pb: Export model to TensorFlow GraphDef format.
        export_tflite: Export model to TensorFlow Lite format.
        export_edgetpu: Export model to Edge TPU format.
        export_tfjs: Export model to TensorFlow.js format.
        export_rknn: Export model to RKNN format.
        export_imx: Export model to IMX format.
        export_executorch: Export model to ExecuTorch format.
        export_axelera: Export model to Axelera format.

    Examples:
        Export a YOLO26 model to ONNX format
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolo26n.pt")  # exports to yolo26n.onnx

        Export with specific arguments
        >>> args = {"format": "onnx", "dynamic": True, "half": True}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolo26n.pt")
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the Exporter class.

        Args:
            cfg (str | Path | dict | SimpleNamespace, optional): Configuration file path or configuration object.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def _task_model(self):
        """Return the Ultralytics task model (unwraps nn.Sequential used for etnms ONNX)."""
        m = self.model
        if isinstance(m, nn.Sequential) and len(m) and hasattr(m[0], "task"):
            return m[0]
        return m

    def _onnx_output_paths(self, tag: str) -> tuple[str, str]:
        """Return ``({stem}_{tag}.onnx``, ``{stem}_{tag}.txt``) beside the source weights."""
        stem, parent = self.file.stem, self.file.parent
        base = parent / f"{stem}_{tag}"
        return str(base.with_suffix(".onnx")), str(base.with_suffix(".txt"))

    def _onnx_input_tensor_name(self) -> str:
        """Return ONNX graph input tensor name (default ``images``)."""
        name = getattr(self.args, "input_tensor_name", None) or "images"
        if not isinstance(name, str):
            name = str(name)
        name = name.strip()
        return name if name else "images"

    def _configure_onnx_output(self, model, fmt: str) -> None:
        """Normalize ``onnx_output``, apply ``etnms`` legacy mapping, and set packed-ONNX export flags."""
        self._onnx_packed_export = False
        self._onnx_ds_yolo_export = False
        if fmt != "onnx":
            oo_other = getattr(self.args, "onnx_output", "default") or "default"
            if oo_other not in (None, "default"):
                LOGGER.warning(f"ignoring onnx_output={oo_other!r} for format={fmt!r}.")
            self.args.onnx_output = "default"
            self._onnx_ds_yolo_export = False
            return

        oo = getattr(self.args, "onnx_output", "default") or "default"
        if not isinstance(oo, str):
            oo = str(oo)
        oo = oo.strip().lower() or "default"

        if getattr(self.args, "etnms", False):
            if oo not in ("default", "enms"):
                raise ValueError("etnms=True conflicts with onnx_output; use onnx_output=enms or omit etnms.")
            oo = "enms"

        if oo not in ONNX_OUTPUT_MODES:
            raise ValueError(f"Invalid onnx_output={oo!r}. Valid modes: {sorted(ONNX_OUTPUT_MODES)}.")

        if oo == "default":
            self.args.onnx_output = "default"
            return

        if oo == "enms":
            if not isinstance(model, DetectionModel) or model.task != "detect":
                raise ValueError("onnx_output='enms' is only supported for detection models.")
            if self.args.nms:
                LOGGER.warning("onnx_output=enms is incompatible with nms=True; setting nms=False.")
                self.args.nms = False
            model.end2end = False
            self.args.end2end = False
            assert TORCH_1_13, f"onnx_output=enms requires torch>=1.13 (found torch=={TORCH_VERSION})"
            self.args.conf = self.args.conf or 0.25
            self.args.onnx_output = "enms"
            return

        if oo == "onnx_nms":
            if not isinstance(model, DetectionModel) or model.task != "detect":
                raise ValueError("onnx_output='onnx_nms' is only supported for detection models.")
            if self.args.nms:
                LOGGER.warning("onnx_output=onnx_nms is incompatible with nms=True; setting nms=False.")
                self.args.nms = False
            model.end2end = False
            self.args.end2end = False
            assert TORCH_1_13, f"onnx_output=onnx_nms requires torch>=1.13 (found torch=={TORCH_VERSION})"
            self.args.conf = self.args.conf or 0.25
            self.args.onnx_output = "onnx_nms"
            return

        if oo == "ds_yolo":
            if model.task != "detect":
                raise ValueError("onnx_output='ds_yolo' is only supported for detection models (DeepStream-Yolo utils).")
            if self.args.nms:
                LOGGER.warning("onnx_output=ds_yolo is incompatible with nms=True; setting nms=False.")
                self.args.nms = False
            model.end2end = False
            self.args.end2end = False
            self._onnx_ds_yolo_export = True
            self.args.onnx_output = "ds_yolo"
            return

        if oo == "emulated_enms":
            if model.task != "detect":
                raise ValueError("onnx_output='emulated_enms' is only supported for detection models.")
            if not is_v10_style_e2e_export(model):
                raise ValueError(
                    "onnx_output='emulated_enms' requires an end-to-end detection head (YOLOv10 / YOLO11 / YOLO26-style). "
                    "Use onnx_output='enms' (EfficientNMS_TRT), onnx_output='onnx_nms' (onnx::NonMaxSuppression), "
                    "onnx_output='ds_yolo' for DeepStream-Yolo, or onnx_output='default' for standard Ultralytics ONNX."
                )
            self._onnx_packed_export = True
            self.args.onnx_output = "emulated_enms"
            return

    def __call__(self, model=None) -> str:
        """Export a model and return the final exported path as a string.

        Returns:
            (str): Path to the exported file or directory (the last export artifact).
        """
        t = time.time()
        fmt = self.args.format.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # available export formats
        if fmt not in fmts:
            import difflib

            # Get the closest match if format is invalid
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 60% similarity required to match
            if not matches:
                msg = "Model is already in PyTorch format." if fmt == "pt" else f"Invalid export format='{fmt}'."
                raise ValueError(f"{msg} Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        is_tf_format = fmt in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}

        # Device
        self.dla = None
        if fmt == "engine" and self.args.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.args.device = "0"
        if fmt == "engine" and "dla" in str(self.args.device):  # convert int/list to str first
            device_str = str(self.args.device)
            self.dla = device_str.rsplit(":", 1)[-1]
            self.args.device = "0"  # update device to "0"
            assert self.dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        if fmt == "imx" and self.args.device is None and torch.cuda.is_available():
            LOGGER.warning("Exporting on CPU while CUDA is available, setting device=0 for faster export on GPU.")
            self.args.device = "0"  # update device to "0"
        self.device = select_device("cpu" if self.args.device is None else self.args.device)

        # Argument compatibility checks
        fmt_keys = dict(zip(fmts_dict["Argument"], fmts_dict["Arguments"]))[fmt]
        validate_args(fmt, self.args, fmt_keys)
        if fmt == "axelera":
            if model.task == "segment" and any(isinstance(m, Segment26) for m in model.modules()):
                raise ValueError("Axelera export does not currently support YOLO26 segmentation models.")
            if not self.args.int8:
                LOGGER.warning("Setting int8=True for Axelera mixed-precision export.")
                self.args.int8 = True
            if not self.args.data:
                self.args.data = TASK2CALIBRATIONDATA.get(model.task)
        if fmt == "imx":
            if not self.args.int8:
                LOGGER.warning("IMX export requires int8=True, setting int8=True.")
                self.args.int8 = True
            if not self.args.nms and model.task in {"detect", "pose", "segment"}:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.args.nms = True
            if model.task not in {"detect", "pose", "classify", "segment"}:
                raise ValueError(
                    "IMX export only supported for detection, pose estimation, classification, and segmentation models."
                )
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if fmt in {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu"}:
                # Disable end2end branch for certain export formats as they does not support topk
                model.end2end = False
                LOGGER.warning(f"{fmt.upper()} export does not support end2end models, disabling end2end branch.")
            if fmt == "engine" and self.args.int8:
                # TensorRT<=10.3.0 with int8 has known end2end build issues
                # https://github.com/ultralytics/ultralytics/issues/23841
                try:
                    import tensorrt as trt

                    if check_version(trt.__version__, "<=10.3.0", hard=True):
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT<=10.3.0 with int8 has known end2end build issues, disabling end2end branch."
                        )
                except ImportError:
                    pass
        if self.args.half and self.args.int8:
            LOGGER.warning("half=True and int8=True are mutually exclusive, setting half=False.")
            self.args.half = False
        if self.args.half and fmt == "torchscript" and self.device.type == "cpu":
            LOGGER.warning(
                "half=True only compatible with GPU export for TorchScript, i.e. use device=0, setting half=False."
            )
            self.args.half = False
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.args.optimize:
            assert fmt != "ncnn", "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
            assert self.device.type == "cpu", "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
        if fmt == "rknn":
            if not self.args.name:
                LOGGER.warning(
                    "Rockchip RKNN export requires a missing 'name' arg for processor type. "
                    "Using default name='rk3588'."
                )
                self.args.name = "rk3588"
            self.args.name = self.args.name.lower()
            assert self.args.name in RKNN_CHIPS, (
                f"Invalid processor name '{self.args.name}' for Rockchip RKNN export. Valid names are {RKNN_CHIPS}."
            )
        if self.args.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert fmt != "tflite" or not ARM64 or not LINUX, "TFLite export with NMS unsupported on ARM64 Linux"
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert fmt != "onnx" or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.args.nms = False
            self.args.conf = self.args.conf or 0.25  # set conf default value for nms export
        self._configure_onnx_output(model, fmt)
        # TensorRT / CoreML builders often need an explicit max batch in the optimization profile; ONNX trace
        # dimensions (batch, imgsz) are only example shapes when dynamic=True — do not warn for ONNX (confusing).
        if fmt in {"engine", "coreml"} and self.args.dynamic and self.args.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' with format={fmt!r} and batch=1: if your deployment tool needs a higher max batch "
                f"in its optimization profile, re-export with a larger trace batch (e.g. batch=16). ONNX export does "
                f"not require this; trace batch/imgsz are example dims only."
            )
        if fmt == "edgetpu":
            if not LINUX or ARM64:
                raise SystemError(
                    "Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler"
                )
            elif self.args.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.args.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # openvino int8 export error: https://github.com/ultralytics/ultralytics/pull/18445
        if self.args.int8 and not self.args.data:
            self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
        if fmt == "tfjs" and ARM64 and LINUX:
            raise SystemError("TF.js exports are not currently supported on ARM64 Linux")
        # Recommend OpenVINO if export and Intel CPU
        if SETTINGS.get("openvino_msg"):
            if is_intel():
                LOGGER.info(
                    "💡 ProTip: Export to OpenVINO format for best performance on Intel hardware."
                    " Learn more at https://docs.ultralytics.com/integrations/openvino/"
                )
            SETTINGS["openvino_msg"] = False

        # Input
        im = torch.zeros(self.args.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        if fmt == "imx":
            from ultralytics.utils.export.imx import FXModel

            model = FXModel(model, self.imgsz)
        if fmt in {"tflite", "edgetpu"}:
            from ultralytics.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        if fmt == "executorch":
            from ultralytics.utils.export.executorch import executorch_wrapper

            model = executorch_wrapper(model)
        for m in model.modules():
            if isinstance(m, Classify):
                m.export = True
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.args.dynamic
                m.export = True
                m.format = self.args.format
                # Clamp max_det to anchor count for small image sizes (required for TensorRT compatibility)
                anchors = sum(int(self.imgsz[0] / s) * int(self.imgsz[1] / s) for s in model.stride.tolist())
                m.max_det = min(self.args.max_det, anchors)
                m.agnostic_nms = self.args.agnostic_nms
                m.xyxy = self.args.nms and fmt != "coreml"
                m.shape = None  # reset cached shape for new export input size
                if hasattr(model, "pe") and hasattr(m, "fuse"):  # for YOLOE models
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        _oo = getattr(self.args, "onnx_output", "default") if fmt == "onnx" else "default"
        if fmt == "onnx" and _oo == "enms":
            model = nn.Sequential(
                model,
                ONNX_EfficientNMS_TRT(
                    class_agnostic=self.args.class_agnostic,
                    max_obj=self.args.topk_all,
                    iou_thres=self.args.iou_thres,
                    score_thres=self.args.conf_thres,
                    max_wh=None,
                    device=self.device,
                    n_classes=len(model.names),
                ),
            )
        if fmt == "onnx" and _oo == "onnx_nms":
            from ultralytics.utils.export.onnx_nonmaxsuppression import ONNX_NMS

            model = nn.Sequential(
                model,
                ONNX_NMS(
                    max_obj=self.args.topk_all,
                    iou_thres=self.args.iou_thres,
                    score_thres=self.args.conf_thres,
                    device=self.device,
                    n_classes=len(model.names),
                ),
            )
        _onnx_fused_post = fmt == "onnx" and _oo in {"enms", "onnx_nms", "ds_yolo", "emulated_enms"}
        y = None
        for _ in range(2):  # dry runs
            y = (
                NMSModel(model, self.args)(im)
                if self.args.nms and fmt not in {"coreml", "imx"} and not _onnx_fused_post
                else model(im)
            )
        if self.args.half and fmt in {"onnx", "torchscript"} and self.device.type != "cpu":
            im, model = im.half(), model.half()  # to FP16

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        _meta_model = (
            model[0]
            if (fmt == "onnx" and _oo in {"enms", "onnx_nms"} and isinstance(model, nn.Sequential))
            else model
        )
        self.pretty_name = Path(_meta_model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = (
            _meta_model.args["data"]
            if hasattr(_meta_model, "args") and isinstance(_meta_model.args, dict)
            else ""
        )
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": int(max(_meta_model.stride)),
            "task": _meta_model.task,
            "batch": self.args.batch,
            "imgsz": self.imgsz,
            "names": _meta_model.names,
            "args": {k: v for k, v in self.args if k in fmt_keys},
            "channels": _meta_model.yaml.get("channels", 3),
            "end2end": getattr(_meta_model, "end2end", False),
        }  # model metadata
        if self.dla is not None:
            self.metadata["dla"] = self.dla  # make sure `AutoBackend` uses correct dla device if it has one
        if _meta_model.task == "pose":
            self.metadata["kpt_shape"] = _meta_model.model[-1].kpt_shape
            if hasattr(_meta_model, "kpt_names"):
                self.metadata["kpt_names"] = _meta_model.kpt_names

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")

        # Export
        if is_tf_format:
            self.args.int8 |= fmt == "edgetpu"
            f, keras_model = self.export_saved_model()
            if fmt in {"pb", "tfjs"}:  # pb prerequisite to tfjs
                f = self.export_pb(keras_model=keras_model)
            if fmt == "tflite":
                f = self.export_tflite()
            if fmt == "edgetpu":
                f = self.export_edgetpu(tflite_model=Path(f) / f"{self.file.stem}_full_integer_quant.tflite")
            if fmt == "tfjs":
                f = self.export_tfjs()
        else:
            f = getattr(self, f"export_{fmt}")()

        # Finish
        if f:
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            q = "int8" if self.args.int8 else "half" if self.args.half else ""  # quantization
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
                f"\nPredict:         yolo predict task={self._task_model().task} model={f} imgsz={imgsz} {q}"
                f"\nValidate:        yolo val task={self._task_model().task} model={f} imgsz={imgsz} data={data} {q} {s}"
                f"\nVisualize:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # path to final export artifact

    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader for calibration of INT8 models."""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.args.data}'")
        tm = self._task_model()
        data = (check_cls_dataset if tm.task == "classify" else check_det_dataset)(self.args.data)
        dataset = YOLODataset(
            data[self.args.split or "val"],
            data=data,
            fraction=self.args.fraction,
            task=tm.task,
            imgsz=max(self.imgsz),
            augment=False,
            batch_size=self.args.batch,
        )
        if hasattr(dataset.transforms.transforms[0], "new_shape"):
            dataset.transforms.transforms[0].new_shape = self.imgsz  # LetterBox with non-square imgsz
        n = len(dataset)
        if n < 1:
            raise ValueError(f"The calibration dataset must have at least 1 image, but found {n} images.")
        batch = min(self.args.batch, n)
        if n < self.args.batch:
            LOGGER.warning(
                f"{prefix} calibration dataset has only {n} images, reducing calibration batch size to {batch}."
            )
        if self.args.format == "axelera" and n < 100:
            LOGGER.warning(f"{prefix} >100 images required for Axelera calibration, found {n} images.")
        elif self.args.format != "axelera" and n < 300:
            LOGGER.warning(f"{prefix} >300 images recommended for INT8 calibration, found {n} images.")
        return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)  # required for batch loading

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """Export YOLO model to TorchScript format."""
        from ultralytics.utils.export.torchscript import torch2torchscript

        return torch2torchscript(
            NMSModel(self.model, self.args) if self.args.nms else self.model,
            self.im,
            self.file,
            optimize=self.args.optimize,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """Export YOLO model to ONNX format."""
        if getattr(self, "_onnx_ds_yolo_export", False):
            return self._export_onnx_ds_yolo(prefix)
        if getattr(self, "_onnx_packed_export", False):
            return self._export_onnx_packed_trt_layout(prefix)

        requirements = ["onnx>=1.12.0,<2.0.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx

        from ultralytics.utils.export.engine import best_onnx_opset, torch2onnx

        _oo = getattr(self.args, "onnx_output", "default")

        opset = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.args.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

        in_name = self._onnx_input_tensor_name()
        if self.args.dynamic:
            LOGGER.info(
                f"{prefix} dynamic=True: imgsz={self.imgsz} and batch={self.args.batch} are trace/example dimensions only; "
                f"runtime shapes are not fixed (ONNX input name '{in_name}')."
            )

        _four_out = _oo in {"enms", "onnx_nms"}
        if _oo != "default":
            f, _ = self._onnx_output_paths(_oo)
        else:
            f = str(self.file.with_suffix(".onnx"))
        base = self.model[0] if _four_out else self.model
        if _four_out:
            output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        else:
            output_names = ["output0", "output1"] if base.task == "segment" else ["output0"]
        dynamic = self.args.dynamic
        if dynamic:
            dynamic = {in_name: {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if _four_out:
                dynamic.update(
                    {
                        "num_dets": {0: "batch"},
                        "det_boxes": {0: "batch"},
                        "det_scores": {0: "batch"},
                        "det_classes": {0: "batch"},
                    }
                )
            elif isinstance(base, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(base, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.args.nms:  # only batch size is dynamic with NMS
                dynamic["output0"].pop(2)
        if self.args.nms and base.task == "obb":
            self.args.opset = opset  # for NMSModel
            self.args.simplify = True  # fix OBB runtime error related to topk

        with arange_patch(dynamic=bool(dynamic), half=self.args.half, fmt=self.args.format):
            torch2onnx(
                NMSModel(self.model, self.args) if self.args.nms else self.model,
                self.im,
                f,
                opset=opset,
                input_names=[in_name],
                output_names=output_names,
                dynamic=dynamic or None,
                do_constant_folding=True,
            )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # IR version
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        # FP16 conversion for CPU export (GPU exports are already FP16 from model.half() during tracing)
        if self.args.half and self.args.format == "onnx" and self.device.type == "cpu":
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} converting to FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

        onnx.save(model_onnx, f)
        return f

    @try_export
    def _export_onnx_ds_yolo(self, prefix=colorstr("ONNX:")):
        """Export ONNX in DeepStream-Yolo layout (``models/DeepStream-Yolo/utils``): ``input`` / ``output``, no NMS."""
        requirements = ["onnx>=1.12.0,<2.0.0"]
        if self.args.simplify:
            requirements += ["onnxslim>=0.1.71", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)
        import onnx

        from ultralytics.utils.export.deepstream_yolo import (
            DeepStreamOutput,
            deepstream_tal_dist2bbox_patch,
            prepare_detection_model_deepstream,
        )
        from ultralytics.utils.export.engine import best_onnx_opset

        f, label_file = self._onnx_output_paths("ds_yolo")
        if len(self.model.names.keys()) > 0:
            with open(label_file, "w", encoding="utf-8") as f_lbl:
                for name in self.model.names.values():
                    f_lbl.write(f"{name}\n")
            LOGGER.info(f"{prefix} wrote labels to '{label_file}'.")

        opset_version = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} DeepStream-Yolo ONNX export, onnx {onnx.__version__} opset {opset_version}...")

        model = deepcopy(self.model)
        prepare_detection_model_deepstream(model)
        model = nn.Sequential(model, DeepStreamOutput())
        model.eval()
        im_x = self.im.cpu()
        model = model.cpu()
        if self.args.half:
            im_x, model = im_x.half(), model.half()

        in_name = self._onnx_input_tensor_name()
        if self.args.dynamic:
            LOGGER.info(
                f"{prefix} dynamic=True: imgsz={self.imgsz} and batch={self.args.batch} are trace/example dimensions only "
                f"(ONNX input '{in_name}')."
            )
        dynamic_axes = {in_name: {0: "batch"}, "output": {0: "batch"}} if self.args.dynamic else None
        onnx_kwargs = {"dynamo": False} if TORCH_2_4 else {}
        with deepstream_tal_dist2bbox_patch():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Exporting aten::index operator of advanced indexing",
                    category=UserWarning,
                )
                torch.onnx.export(
                    model,
                    im_x,
                    f,
                    verbose=False,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=[in_name],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    **onnx_kwargs,
                )

        model_onnx = onnx.load(f)
        if self.args.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)
            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        if getattr(model_onnx, "ir_version", 0) > 10:
            model_onnx.ir_version = 10
        onnx.save(model_onnx, f)
        return f

    @try_export
    def _export_onnx_packed_trt_layout(self, prefix=colorstr("ONNX:")):
        """Export ONNX with packed TRT-style four outputs for ``onnx_output=emulated_enms`` (end-to-end heads only).

        Uses EfficientNMS_TRT only for heads that output raw boxes+scores (e.g. YOLOv8). End-to-end heads (YOLOv10 / 11 /
        26-style) already include top-k in the graph; those exports omit the extra EfficientNMS stage in the v10 path.
        """
        requirements = ["onnx>=1.12.0,<2.0.0"]
        if self.args.simplify:
            requirements += ["onnxsim>=0.4.33", "onnxruntime" + ("-gpu" if torch.cuda.is_available() else "")]
        check_requirements(requirements)

        import onnx  # noqa
        from ultralytics.utils.export.engine import best_onnx_opset, patch_onnx_helper_for_graphsurgeon

        oo_tag = getattr(self.args, "onnx_output", "emulated_enms")
        f, label_file = self._onnx_output_paths(oo_tag)
        labels = len(self.model.names)
        is_det_model = True
        v10detect = is_v10_style_e2e_export(self.model)

        if len(self.model.names.keys()) > 0:
            with open(label_file, "w", encoding="utf-8") as f_trt:
                for name in self.model.names.values():
                    f_trt.write(f"{name}\n")
            LOGGER.info(f"{prefix} wrote labels to '{label_file}'.")

        opset_version = self.args.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type)
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")

        batch_size = "batch"
        dynamic = self.args.dynamic
        in_name = self._onnx_input_tensor_name()
        if dynamic:
            LOGGER.info(
                f"{prefix} dynamic=True: imgsz={self.imgsz} and batch={self.args.batch} are trace/example dimensions only "
                f"(ONNX input '{in_name}')."
            )
        dynamic_axes = {in_name: {0: "batch", 2: "height", 3: "width"}}  # variable length axes
        output_axes = {
            "num_dets": {0: "batch"},
            "det_boxes": {0: "batch"},
            "det_scores": {0: "batch"},
            "det_classes": {0: "batch"},
        }

        d = {
            "stride": int(max(self.model.stride)),
            "names": self.model.names,
            "model type": "Segmentation" if isinstance(self.model, SegmentationModel) else "Detection",
            "train input": f"{tuple(self.im.shape[1:])} - CHW",
            "TRT Compatibility": "8.6 or above" if self.args.class_agnostic else "8.5 or above",
        }
        if not v10detect:
            d["TRT Plugins"] = (
                "TRT_EfficientNMSX, ROIAlign" if isinstance(self.model, SegmentationModel) else "TRT_EfficientNMS"
            )

        if not isinstance(self.model, SegmentationModel):
            is_det_model = True
            output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]
            shapes = [
                batch_size,
                1,
                batch_size,
                self.args.topk_all,
                4,
                batch_size,
                self.args.topk_all,
                batch_size,
                self.args.topk_all,
            ]

        else:
            is_det_model = False
            output_axes["det_masks"] = {0: "batch"}
            output_names = ["num_dets", "det_boxes", "det_scores", "det_classes", "det_masks"]
            shapes = [
                batch_size,
                1,
                batch_size,
                self.args.topk_all,
                4,
                batch_size,
                self.args.topk_all,
                batch_size,
                self.args.topk_all,
                batch_size,
                self.args.topk_all,
                self.args.mask_resolution * self.args.mask_resolution,
            ]

        dynamic_axes.update(output_axes)

        if v10detect:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            self.model.float()
            self.model.fuse()
            for m in self.model.modules():
                if isinstance(m, v10Detect) or (type(m) is Detect and m.end2end):
                    m.max_det = self.args.topk_all

        if v10detect:
            self.model = nn.Sequential(
                self.model,
                End2End_TRT(
                    self.model,
                    self.args.class_agnostic,
                    self.args.topk_all,
                    self.args.iou_thres,
                    self.args.conf_thres,
                    self.args.mask_resolution,
                    self.args.pooler_scale,
                    self.args.sampling_ratio,
                    None,
                    self.args.device,
                    labels,
                    is_det_model,
                    v10detect,
                ),
            )
        else:
            self.model = End2End_TRT(
                self.model,
                self.args.class_agnostic,
                self.args.topk_all,
                self.args.iou_thres,
                self.args.conf_thres,
                self.args.mask_resolution,
                self.args.pooler_scale,
                self.args.sampling_ratio,
                None,
                self.args.device,
                labels,
                is_det_model,
                v10detect,
            )

        self.model.eval()
        # PyTorch 2.4+ defaults dynamo=True; dynamic_axes then triggers a warning and is ignored for shape hints.
        # Match torch2onnx (utils/export/engine.py): use the legacy TorchScript exporter with dynamic_axes.
        onnx_kwargs = {"dynamo": False} if TORCH_2_4 else {}
        # Legacy exporter warns on aten::index → Gather; safe for typical YOLO inference (non-negative indices).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Exporting aten::index operator of advanced indexing",
                category=UserWarning,
            )
            torch.onnx.export(
                self.model.cpu() if dynamic else self.model,
                self.im.cpu() if dynamic else self.im,
                f,
                verbose=False,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=[in_name],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                **onnx_kwargs,
            )

        model_onnx = onnx.load(f)
        onnx.checker.check_model(model_onnx)

        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        for i in model_onnx.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

        check_requirements("onnxsim")
        try:
            import onnxsim

            LOGGER.info(f"\n{prefix} Starting to simplify ONNX...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
        except Exception as e:
            LOGGER.info(f"\n{prefix} Simplifier failure: {e}")

        check_requirements("onnx_graphsurgeon")
        LOGGER.info(f"\n{prefix} Starting to cleanup ONNX using onnx_graphsurgeon...")
        try:
            patch_onnx_helper_for_graphsurgeon()
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(model_onnx)
            graph = graph.cleanup().toposort()
            model_onnx = gs.export_onnx(graph)
        except Exception as e:
            LOGGER.info(f"\n{prefix} Cleanup failure: {e}")
        onnx.save(model_onnx, f)
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """Export YOLO model to OpenVINO format."""
        from ultralytics.utils.export import torch2openvino

        # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023"
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"

        def serialize(ov_model, file):
            """Set RT info, serialize, and save metadata YAML."""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.args.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            ov.save_model(ov_model, file, compress_to_fp16=self.args.half)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

        calibration_dataset, ignored_scope = None, None
        if self.args.int8:
            check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
            check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
            import nncf

            calibration_dataset = nncf.Dataset(self.get_int8_calibration_dataloader(prefix), self._transform_fn)
            if isinstance(self.model.model[-1], Detect):
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(  # ignore operations
                    patterns=[
                        f".*{head_module_name}/.*/Add",
                        f".*{head_module_name}/.*/Sub*",
                        f".*{head_module_name}/.*/Mul*",
                        f".*{head_module_name}/.*/Div*",
                    ],
                    types=["Sigmoid"],
                )

        ov_model = torch2openvino(
            model=NMSModel(self.model, self.args) if self.args.nms else self.model,
            im=self.im,
            dynamic=self.args.dynamic,
            half=self.args.half,
            int8=self.args.int8,
            calibration_dataset=calibration_dataset,
            ignored_scope=ignored_scope,
            prefix=prefix,
        )

        suffix = f"_{'int8_' if self.args.int8 else ''}openvino_model{os.sep}"
        f = str(self.file).replace(self.file.suffix, suffix)
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """Export YOLO model to PaddlePaddle format."""
        from ultralytics.utils.export.paddle import torch2paddle

        return torch2paddle(self.model, self.im, self.file, self.metadata, prefix)

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
        from ultralytics.utils.export.mnn import onnx2mnn

        f_onnx = self.export_onnx()
        return onnx2mnn(
            f_onnx, self.file, half=self.args.half, int8=self.args.int8, metadata=self.metadata, prefix=prefix
        )

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
        from ultralytics.utils.export.ncnn import torch2ncnn

        return torch2ncnn(
            self.model,
            self.im,
            self.file,
            half=self.args.half,
            metadata=self.metadata,
            device=self.device,
            prefix=prefix,
        )

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """Export YOLO model to CoreML format."""
        mlmodel = self.args.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
        from ultralytics.utils.export.coreml import IOSDetectModel, pipeline_coreml, torch2coreml

        # latest numpy 2.4.0rc1 breaks coremltools exports
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        if self.args.batch > 1:
            assert self.args.dynamic, (
                "batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'."
            )
        if self.args.dynamic:
            assert not self.args.nms, (
                "'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them."
            )
            assert self.model.task != "classify", "'dynamic=True' is not supported for CoreML classification models."
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        if self.model.task == "detect":
            model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel) if self.args.nms else self.model
        else:
            if self.args.nms:
                LOGGER.warning(f"{prefix} 'nms=True' is only available for Detect models like 'yolo26n.pt'.")
                # TODO CoreML Segment and Pose model pipelining
            model = self.model

        if self.args.dynamic:
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.args.batch, default=1),
                    self.im.shape[1],
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[0] * 2, default=self.imgsz[0]),
                    ct.RangeDim(lower_bound=32, upper_bound=self.imgsz[1] * 2, default=self.imgsz[1]),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

        ct_model = torch2coreml(
            model=model,
            inputs=inputs,
            im=self.im,
            classifier_names=list(self.model.names.values()) if self.model.task == "classify" else None,
            mlmodel=mlmodel,
            half=self.args.half,
            int8=self.args.int8,
            metadata=self.metadata,
            prefix=prefix,
        )

        if self.args.nms and self.model.task == "detect":
            ct_model = pipeline_coreml(
                ct_model,
                self.output_shape,
                weights_dir=None if mlmodel else ct_model.weights_dir,
                metadata=self.metadata,
                mlmodel=mlmodel,
                iou=self.args.iou,
                conf=self.args.conf,
                agnostic_nms=self.args.agnostic_nms,
                prefix=prefix,
            )

        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f

    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016

        # Force re-install TensorRT on CUDA 13 ARM devices to 10.15.x versions for RT-DETR exports
        # https://github.com/ultralytics/ultralytics/issues/22873
        if is_jetson(jetpack=7) or is_dgx():
            check_tensorrt("10.15")

        try:
            import tensorrt as trt
        except ImportError:
            check_tensorrt()
            import tensorrt as trt
        check_version(trt.__version__, ">=7.0.0", hard=True)
        check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")

        from ultralytics.utils.export.engine import onnx2engine

        # Setup and checks
        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        onnx2engine(
            f_onnx,
            f,
            self.args.workspace,
            self.args.half,
            self.args.int8,
            self.args.dynamic,
            self.im.shape,
            dla=self.dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.args.int8 else None,
            metadata=self.metadata,
            verbose=self.args.verbose,
            prefix=prefix,
        )

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel:")):
        """Export YOLO model to TensorFlow SavedModel format."""
        cuda = torch.cuda.is_available()
        try:
            import tensorflow as tf
        except ImportError:
            check_requirements("tensorflow>=2.0.0,<=2.19.0")
            import tensorflow as tf
        check_requirements(
            (
                "tf_keras<=2.19.0",  # required by 'onnx2tf' package
                "sng4onnx>=1.0.1",  # required by 'onnx2tf' package
                "onnx_graphsurgeon>=0.3.26",  # required by 'onnx2tf' package
                "ai-edge-litert>=1.2.0" + (",<1.4.0" if MACOS else ""),  # required by 'onnx2tf' package
                "onnx>=1.12.0,<2.0.0",
                "onnx2tf>=1.26.3,<1.29.0",  # pin to avoid h5py build issues on aarch64
                "onnxslim>=0.1.71",
                "onnxruntime-gpu" if cuda else "onnxruntime",
                "protobuf>=5",
            ),
            cmds="--extra-index-url https://pypi.ngc.nvidia.com",  # onnx_graphsurgeon only on NVIDIA
        )

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        check_version(
            tf.__version__,
            ">=2.0.0",
            name="tensorflow",
            verbose=True,
            msg="https://github.com/ultralytics/ultralytics/issues/5161",
        )
        from ultralytics.utils.export.tensorflow import onnx2saved_model

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder

        # Export to TF
        images = None
        if self.args.int8 and self.args.data:
            images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
            images = (
                torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz)
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.float32)
            )

        # Export to ONNX
        if isinstance(self.model.model[-1], RTDETRDecoder):
            self.args.opset = self.args.opset or 19
            assert 16 <= self.args.opset <= 19, "RTDETR export requires opset>=16;<=19"
        self.args.simplify = True
        f_onnx = self.export_onnx()  # ensure ONNX is available
        keras_model = onnx2saved_model(
            f_onnx,
            f,
            int8=self.args.int8,
            images=images,
            disable_group_convolution=self.args.format in {"tfjs", "edgetpu"},
            prefix=prefix,
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            file.unlink() if "quant_with_int16_act.tflite" in str(file) else self._add_tflite_metadata(file)

        return str(f), keras_model  # or keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef:")):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
        from ultralytics.utils.export.tensorflow import keras2pb

        f = self.file.with_suffix(".pb")
        keras2pb(keras_model, f, prefix)
        return f

    @try_export
    def export_tflite(self, prefix=colorstr("TensorFlow Lite:")):
        """Export YOLO model to TensorFlow Lite format."""
        # BUG https://github.com/ultralytics/ultralytics/issues/13436
        import tensorflow as tf

        LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
        saved_model = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if self.args.int8:
            f = saved_model / f"{self.file.stem}_int8.tflite"  # fp32 in/out
        elif self.args.half:
            f = saved_model / f"{self.file.stem}_float16.tflite"  # fp32 in/out
        else:
            f = saved_model / f"{self.file.stem}_float32.tflite"
        return str(f)

    @try_export
    def export_axelera(self, prefix=colorstr("Axelera:")):
        """Export YOLO model to Axelera format."""
        assert LINUX and not (ARM64 and IS_DOCKER), (
            "export is only supported on Linux and is not supported on ARM64 Docker."
        )
        assert TORCH_2_8, "export requires torch>=2.8.0."

        from ultralytics.utils.export.axelera import torch2axelera

        return torch2axelera(
            model=self.model,
            file=self.file,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix),
            transform_fn=self._transform_fn,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_executorch(self, prefix=colorstr("ExecuTorch:")):
        """Export YOLO model to ExecuTorch *.pte format."""
        assert TORCH_2_9, f"ExecuTorch requires torch>=2.9.0 but torch=={TORCH_VERSION} is installed"
        check_executorch_requirements()
        from ultralytics.utils.export.executorch import torch2executorch

        return torch2executorch(self.model, self.file, self.im, metadata=self.metadata, prefix=prefix)

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU:")):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
        cmd = "edgetpu_compiler --version"
        help_url = "https://coral.ai/docs/edgetpu/compiler/"
        assert LINUX, f"export only supported on Linux. See {help_url}"
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True).returncode != 0:
            LOGGER.info(f"\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}")
            sudo = "sudo " if is_sudo_available() else ""
            for c in (
                f"{sudo}mkdir -p /etc/apt/keyrings",
                f"curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | {sudo}gpg --no-tty --dearmor -o /etc/apt/keyrings/google.gpg",
                f'echo "deb [signed-by=/etc/apt/keyrings/google.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | {sudo}tee /etc/apt/sources.list.d/coral-edgetpu.list',
            ):
                subprocess.run(c, shell=True, check=True)
            check_apt_requirements(["edgetpu-compiler"])

        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().rsplit(maxsplit=1)[-1]
        from ultralytics.utils.export.tensorflow import tflite2edgetpu

        LOGGER.info(f"\n{prefix} starting export with Edge TPU compiler {ver}...")
        tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        f = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model
        self._add_tflite_metadata(f)
        return f

    @try_export
    def export_tfjs(self, prefix=colorstr("TensorFlow.js:")):
        """Export YOLO model to TensorFlow.js format."""
        check_requirements("tensorflowjs")
        from ultralytics.utils.export.tensorflow import pb2tfjs

        f = str(self.file).replace(self.file.suffix, "_web_model")  # js dir
        f_pb = str(self.file.with_suffix(".pb"))  # *.pb path
        pb2tfjs(pb_file=f_pb, output_dir=f, half=self.args.half, int8=self.args.int8, prefix=prefix)
        # Add metadata
        YAML.save(Path(f) / "metadata.yaml", self.metadata)  # add metadata.yaml
        return f

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """Export YOLO model to RKNN format."""
        from ultralytics.utils.export.rknn import onnx2rknn

        self.args.opset = min(self.args.opset or 19, 19)  # rknn-toolkit expects opset<=19
        f_onnx = self.export_onnx()
        return onnx2rknn(f_onnx, name=self.args.name, metadata=self.metadata, prefix=prefix)

    @try_export
    def export_imx(self, prefix=colorstr("IMX:")):
        """Export YOLO model to IMX format."""
        assert LINUX, (
            "Export only supported on Linux."
            "See https://developer.aitrios.sony-semicon.com/en/docs/raspberry-pi-ai-camera/imx500-converter?version=3.17.3&progLang="
        )
        assert IS_PYTHON_MINIMUM_3_9, "IMX export is only supported on Python 3.9 or above."

        if getattr(self.model, "end2end", False):
            raise ValueError("IMX export is not supported for end2end models.")
        check_requirements(
            (
                "model-compression-toolkit>=2.4.1",
                "edge-mdt-cl<1.1.0",
                "edge-mdt-tpc>=1.2.0",
                "pydantic<=2.11.7",
            )
        )

        check_requirements("imx500-converter[pt]>=3.17.3")
        from ultralytics.utils.export.imx import torch2imx

        # Install Java>=17
        try:
            java_output = subprocess.run(["java", "--version"], check=True, capture_output=True).stdout.decode()
            version_match = re.search(r"(?:openjdk|java) (\d+)", java_output)
            java_version = int(version_match.group(1)) if version_match else 0
            assert java_version >= 17, "Java version too old"
        except (FileNotFoundError, subprocess.CalledProcessError, AssertionError):
            if IS_UBUNTU or IS_DEBIAN_TRIXIE:
                LOGGER.info(f"\n{prefix} installing Java 21 for Ubuntu...")
                check_apt_requirements(["openjdk-21-jre"])
            elif IS_RASPBERRYPI or IS_DEBIAN_BOOKWORM:
                LOGGER.info(f"\n{prefix} installing Java 17 for Raspberry Pi or Debian ...")
                check_apt_requirements(["openjdk-17-jre"])

        return torch2imx(
            self.model,
            self.file,
            self.args.conf,
            self.args.iou,
            self.args.max_det,
            metadata=self.metadata,
            dataset=self.get_int8_calibration_dataloader(prefix),
            prefix=prefix,
        )

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://ai.google.dev/edge/litert/models/metadata."""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

    @staticmethod
    def _transform_fn(data_item) -> np.ndarray:
        """The transformation function for Axelera/OpenVINO quantization preprocessing."""
        data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
        assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
        im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
        return im[None] if im.ndim == 3 else im

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


class NMSModel(torch.nn.Module):
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""

    def __init__(self, model, args):
        """Initialize the NMSModel.

        Args:
            model (torch.nn.Module): The model to wrap with NMS postprocessing.
            args (SimpleNamespace): The export arguments.
        """
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
        self.is_tf = self.args.format in frozenset({"saved_model", "tflite", "tfjs"})

    def forward(self, x):
        """Perform inference with NMS post-processing. Supports Detect, Segment, OBB and Pose.

        Args:
            x (torch.Tensor): The preprocessed tensor with shape (B, C, H, W).

        Returns:
            (torch.Tensor | tuple): Tensor of shape (B, max_det, 4 + 2 + extra_shape) where B is the batch size, or a
                tuple of (detections, proto) for segmentation models.
        """
        from functools import partial

        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # extras from Segment, OBB, Pose
        if self.args.dynamic and self.args.batch > 1:  # batch size needs to always be same due to loop unroll
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)  # in case num_anchors < max_det
        # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
            if self.args.format == "tflite":  # TFLite is already normalized
                nmsbox *= multiplier
            else:
                nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
            if not self.args.agnostic_nms:  # class-wise NMS
                end = 2 if self.obb else 4
                # fully explicit expansion otherwise reshape error
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    TorchNMS.fast_nms,
                    use_triu=not (
                        self.is_tf
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.int8)  # OpenVINO int8 error with triu
                    ),
                    iou_func=batch_probiou,
                    exit_early=False,
                )
                if self.obb
                else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # Zero-pad to max_det size to avoid reshape error
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]

# TensorRT 8.5 compatible
class TRT_EfficientNMS_85(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25
                 ):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes
    


class TRT_EfficientNMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25,
                 class_agnostic=0):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   class_agnostic_i=class_agnostic,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class TRT_EfficientNMSX_85(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0,num_boxes,(batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25
                 ):
        out = g.op("TRT::EfficientNMSX_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=5)
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices
    

class TRT_EfficientNMSX(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=0,
    ):

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0,num_boxes,(batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25,
                 class_agnostic=0):
        out = g.op("TRT::EfficientNMSX_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   class_agnostic_i=class_agnostic,
                   score_threshold_f=score_threshold,
                   outputs=5)
        nums, boxes, scores, classes, det_indices = out
        return nums, boxes, scores, classes, det_indices
    

class TRT_ROIAlign(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode= 1,
        mode=1,  # 1- avg pooling  / 0 - max pooling
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode=1,
        mode=1,
        output_height=160,
        output_width=160,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        return g.op(
            "TRT::ROIAlign_TRT",
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_i=coordinate_transformation_mode,
            mode_i=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )

class ONNX_EfficientNMS_TRT(torch.nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes
        

    def forward(self, x):
        if isinstance(x, list):
            x = x[1]
        x = x.permute(0, 2, 1)
        bboxes = x[..., :4].unsqueeze(2)  # [B, N, 1, 4]
        scores = x[..., 4:]               # [B, N, nc]
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold, self.class_agnostic)
        else:
            num_det, det_boxes, det_scores, det_classes = TRT_EfficientNMS_85.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class ONNX_EfficientNMSX_TRT(torch.nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes

    def forward(self, x):
        if isinstance(x, list):
            x = x[1]
        x = x.permute(0, 2, 1)
        bboxes = x[..., :4].unsqueeze(2)  # [B, N, 1, 4]
        scores = x[..., 4:]               # [B, N, nc]
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold, self.class_agnostic)
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes, det_indices



class End2End_TRT(torch.nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, class_agnostic=False, max_obj=100, iou_thres=0.45, score_thres=0.25, mask_resolution=56, pooler_scale=0.25, sampling_ratio=0, max_wh=None, device=None, n_classes=80, is_det_model=True, v10detect=False):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.v10detect = v10detect

        if is_det_model and not self.v10detect:
            self.model.model[-1].end2end = False
            self.patch_model = ONNX_EfficientNMS_TRT 
            self.end2end = self.patch_model(class_agnostic, max_obj, iou_thres, score_thres, max_wh, device, n_classes)
            self.end2end.eval()
        elif not is_det_model and not self.v10detect:
            self.model.model[-1].end2end = False
            self.patch_model = ONNX_End2End_MASK_TRT 
            self.end2end = self.patch_model(class_agnostic, max_obj, iou_thres, score_thres, mask_resolution, pooler_scale, sampling_ratio, max_wh, device, n_classes) 
            self.end2end.eval()  
        elif self.v10detect:
            self.model.model[-1].end2end = True     

    def forward(self, x):
        if not self.v10detect:
            # For YOLOv8, use the end2end process
            x = self.model(x)
            x = self.end2end(x)
            return x
        else:
            # For YOLOv10 / YOLO11 / YOLO26 end-to-end: tensor is already [x1,y1,x2,y2, score, cls] per anchor after top-k.
            det_boxes = x[:, :, :4]
            det_scores = x[:, :, 4]
            det_classes = x[:, :, 5].int()
            num_dets = (x[:, :, 4] > 0.0).sum(dim=1, keepdim=True).int()
            return num_dets, det_boxes, det_scores, det_classes

class ONNX_End2End_MASK_TRT(torch.nn.Module):
    """onnx module with ONNX-TensorRT NMS/ROIAlign operation."""
    def __init__(
        self,
        class_agnostic=False,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        mask_resolution=160,
        pooler_scale=0.25,
        sampling_ratio=0,
        max_wh=None,
        device=None,
        n_classes=80
    ):
        super().__init__()
        assert isinstance(max_wh,(int)) or max_wh is None
        self.device = device if device else torch.device('cpu')
        self.class_agnostic = 1 if class_agnostic else 0
        self.max_obj = max_obj
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio
       
    def forward(self, x):
        det=x[0]
        proto=x[1]
        det = det.permute(0, 2, 1)

        bboxes = det[..., :4].unsqueeze(2)  # [B, N, 1, 4]
        scores = det[..., 4: 4 + self.n_classes]
       
        batch_size, nm, proto_h, proto_w = proto.shape
        total_object = batch_size * self.max_obj
        masks = det[..., 4 + self.n_classes : 4 + self.n_classes + nm]
        if self.class_agnostic == 1:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold,self.class_agnostic)
        else:
            num_det, det_boxes, det_scores, det_classes, det_indices = TRT_EfficientNMSX_85.apply(bboxes, scores, self.background_class, self.box_coding,
                                                                        self.iou_threshold, self.max_obj,
                                                                        self.plugin_version, self.score_activation,
                                                                        self.score_threshold)
        
        batch_indices = torch.ones_like(det_indices) * torch.arange(batch_size, device=self.device, dtype=torch.int32).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)
        det_masks = masks[batch_indices, det_indices]


        pooled_proto = TRT_ROIAlign.apply(  proto,
                                            det_boxes.view(total_object, 4),
                                            batch_indices,
                                            1,
                                            1,
                                            self.mask_resolution,
                                            self.mask_resolution,
                                            self.sampling_ratio,
                                            self.pooler_scale
                                        )
        pooled_proto = pooled_proto.view(
            total_object, nm, self.mask_resolution * self.mask_resolution,
        )

        det_masks = (
            torch.matmul(det_masks.unsqueeze(dim=1), pooled_proto)
            .sigmoid()
            .view(batch_size, self.max_obj, self.mask_resolution * self.mask_resolution)
        )

        return num_det, det_boxes, det_scores, det_classes, det_masks
