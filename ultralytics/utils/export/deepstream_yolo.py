# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DeepStream-Yolo ONNX layout (see ``models/DeepStream-Yolo/utils/export_*.py``).

Single ``output`` tensor: no NMS in the graph; ``DeepStreamOutput`` packs boxes, score, and class id per anchor.
Input name is ``input`` (not ``images``).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, Detect, v10Detect


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    """Match DeepStream-Yolo utils: xyxy corners from DFL distances."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


@contextmanager
def deepstream_tal_dist2bbox_patch():
    """Temporarily patch ``ultralytics.utils.tal.dist2bbox`` for tracing (utils scripts rely on this)."""
    import ultralytics.utils.tal as tal

    old = tal.dist2bbox
    tal.dist2bbox = _dist2bbox
    try:
        yield
    finally:
        tal.dist2bbox = old


def forward_deepstream(self, x):
    """Replacement forward for some heads; matches ``export_yolo26.py`` / ``export_yoloV10.py``."""
    x_detach = [xi.detach() for xi in x]
    if hasattr(self, "inference"):
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        y = self.inference(one2one)
    else:
        one2one = self.forward_head(x_detach, **self.one2one)
        y = self._inference(one2one)
    return y


class DeepStreamOutput(nn.Module):
    """Post-head layer: transpose, take box + argmax class score/label (no NMS)."""

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def _is_yolo26_family_stem(model: nn.Module) -> bool:
    """YOLO26 DeepStream scripts patch ``Detect`` forward; YOLOv8/11/12 scripts do not."""
    p = getattr(model, "pt_path", None)
    yf = ""
    if p is not None:
        yf = str(p)
    elif hasattr(model, "yaml") and isinstance(model.yaml, dict):
        yf = str(model.yaml.get("yaml_file", ""))
    stem = Path(yf).stem.lower()
    return "yolo26" in stem or "yolov26" in stem


def prepare_detection_model_deepstream(model: nn.Module) -> None:
    """In-place head/C2f patches aligned with ``models/DeepStream-Yolo/utils`` export scripts."""
    for _, m in model.named_modules():
        if isinstance(m, (Detect, v10Detect)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
            if isinstance(m, v10Detect):
                m.forward = types.MethodType(forward_deepstream, m)
            elif isinstance(m, Detect) and _is_yolo26_family_stem(model):
                m.forward = types.MethodType(forward_deepstream, m)
        elif isinstance(m, C2f):
            m.forward = m.forward_split
