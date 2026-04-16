# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Standard ONNX ``NonMaxSuppression`` op for detection export.

TensorRT maps this operator to ``nvinfer1::INMSLayer`` when building an engine from ONNX; see NVIDIA’s INMSLayer I/O
(https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_n_m_s_layer.html).
The exporter then packs **SelectedIndices**-style results into ``num_dets`` / ``det_*`` for parity with ``enms``.

Operator spec: https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html — use ``onnx_output=onnx_nms`` in ``Exporter``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.utils.ops import xywh2xyxy

try:
    from torchvision.ops import nms as tv_nms
except ImportError:  # pragma: no cover
    tv_nms = None


class OnnxNonMaxSuppressionFunction(torch.autograd.Function):
    """Maps to ONNX ``NonMaxSuppression`` (``center_point_box=1``: x, y, w, h boxes).

    Forward uses class-wise torchvision NMS (xywh → xyxy) as a reference; the exported graph uses the ONNX op.
    """

    @staticmethod
    def forward(
        ctx,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        max_output_boxes_per_class: torch.Tensor,
        iou_threshold: torch.Tensor,
        score_threshold: torch.Tensor,
    ) -> torch.Tensor:
        out = _reference_nms_indices(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )
        # Tie to backbone activations so ONNX tracing does not constant-fold the detector away when
        # dummy inputs yield no detections (scores all below threshold).
        if torch.onnx.is_in_onnx_export():
            tie = (boxes.sum() * 0).to(torch.int64)
            if out.numel() == 0:
                out = torch.zeros((1, 3), device=boxes.device, dtype=torch.int64)
            out = out + tie  # scalar 0; keeps activations live for ONNX tracing
        return out

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            center_point_box_i=1,
            outputs=1,
        )


def _reference_nms_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_boxes_per_class: torch.Tensor,
    iou_threshold: torch.Tensor,
    score_threshold: torch.Tensor,
) -> torch.Tensor:
    """Return ``selected_indices`` [S, 3] int64 — (batch_index, class_index, box_index)."""
    if tv_nms is None:
        raise ImportError("onnx_nms export requires torchvision (pip install torchvision).")

    B, N, _ = boxes.shape
    _, C, _ = scores.shape
    max_out = int(max_output_boxes_per_class.detach().cpu().item())
    iou_t = float(iou_threshold.detach().cpu().item())
    score_t = float(score_threshold.detach().cpu().item())

    rows: list[list[int]] = []
    for b in range(B):
        for c in range(C):
            sc = scores[b, c]
            mask = sc > score_t
            if not mask.any():
                continue
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            box_xywh = boxes[b, valid_idx]
            box_xyxy = xywh2xyxy(box_xywh)
            sco = sc[valid_idx]
            keep = tv_nms(box_xyxy, sco, iou_t)
            keep = keep[:max_out]
            for j in keep:
                vi = int(valid_idx[j])
                rows.append([b, c, vi])

    if not rows:
        return torch.zeros((0, 3), device=boxes.device, dtype=torch.int64)
    return torch.tensor(rows, device=boxes.device, dtype=torch.int64)


def _pack_efficient_layout(
    boxes: torch.Tensor,
    scores_bcn: torch.Tensor,
    selected_indices: torch.Tensor,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad/truncate to ``num_dets``, ``det_boxes``, ``det_scores``, ``det_classes`` (same names as plugin export)."""
    B = boxes.shape[0]
    device = boxes.device
    dtype = boxes.dtype
    det_boxes = torch.zeros(B, max_det, 4, device=device, dtype=dtype)
    det_scores = torch.zeros(B, max_det, device=device, dtype=dtype)
    det_classes = torch.zeros(B, max_det, device=device, dtype=dtype)
    num_dets = torch.zeros(B, 1, device=device, dtype=torch.int32)

    if selected_indices.numel() == 0:
        return num_dets, det_boxes, det_scores, det_classes

    for b in range(B):
        sb = selected_indices[selected_indices[:, 0] == b]
        nk = int(sb.shape[0])
        if nk == 0:
            continue
        k = min(nk, max_det)
        num_dets[b, 0] = torch.tensor(k, device=device, dtype=torch.int32)
        cidx = sb[:k, 1].long()
        bidx = sb[:k, 2].long()
        det_boxes[b, :k] = boxes[b, bidx]
        det_scores[b, :k] = scores_bcn[b, cidx, bidx]
        det_classes[b, :k] = cidx.float()

    return num_dets, det_boxes, det_scores, det_classes


class ONNX_NMS(nn.Module):
    """Append ``onnx::NonMaxSuppression`` + packed detection tensors (``onnx_output=onnx_nms``).

    Inputs match the raw two-stage head (same tensor layout as ``ONNX_EfficientNMS_TRT``). Outputs:
    ``num_dets``, ``det_boxes``, ``det_scores``, ``det_classes``.
    """

    def __init__(
        self,
        max_obj: int = 100,
        iou_thres: float = 0.45,
        score_thres: float = 0.25,
        device=None,
        n_classes: int = 80,
    ):
        super().__init__()
        self.max_obj = max_obj
        self.iou_thres = iou_thres
        self.score_thres = score_thres
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor | list | tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)):
            x = x[1]
        x = x.permute(0, 2, 1)
        bboxes = torch.cat([x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]], dim=-1)
        scores = x[..., 4:].permute(0, 2, 1)

        device = bboxes.device
        dtype = bboxes.dtype
        max_out = torch.tensor(self.max_obj, dtype=torch.int64, device=device)
        iou_t = torch.tensor(self.iou_thres, dtype=dtype, device=device)
        score_t = torch.tensor(self.score_thres, dtype=dtype, device=device)

        selected = OnnxNonMaxSuppressionFunction.apply(bboxes, scores, max_out, iou_t, score_t)
        return _pack_efficient_layout(bboxes, scores, selected, self.max_obj)
