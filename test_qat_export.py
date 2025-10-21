#!/usr/bin/env python3
"""
Test script to export QAT model without re-running full QAT workflow.
This loads a saved QAT checkpoint and exports it to ONNX.

Usage:
    python test_qat_export.py --model runs/detect/qat16/best_qat.pt --imgsz 320
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER


def export_qat_model(model_path, base_model="yolo11n.pt", imgsz=640, format="onnx"):
    """
    Export a saved QAT model to ONNX format.
    
    Args:
        model_path (str): Path to saved QAT model (.pt file from mto.save())
        base_model (str): Path to base YOLO model for structure
        imgsz (int): Image size for export
        format (str): Export format (default: onnx)
    """
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
        
        LOGGER.info(f"Loading base model: {base_model}")
        
        # Load base YOLO model to get structure
        yolo_model = YOLO(base_model)
        base_torch_model = yolo_model.model
        
        LOGGER.info(f"Restoring QAT weights from: {model_path}")
        
        # Restore QAT model using nvidia-modelopt
        # mto.restore() needs: (model, checkpoint_path)
        model = mto.restore(base_torch_model, model_path)
        
        # Ensure model has required attributes for export
        if not hasattr(model, 'task'):
            model.task = 'detect'  # Default to detection task
        
        if not hasattr(model, 'names'):
            # Load from COCO8 default
            model.names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                          4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
        
        LOGGER.info(f"Model task: {model.task}")
        LOGGER.info(f"Model has {len(model.names)} classes")
        
        # Export to ONNX
        from ultralytics.engine.exporter import Exporter
        
        export_args = {
            "format": format,
            "imgsz": imgsz,
            "batch": 1,
            "verbose": True,
        }
        
        LOGGER.info(f"Exporting to {format.upper()}...")
        exporter = Exporter(overrides=export_args)
        export_path = exporter(model=model)
        
        LOGGER.info(f"✅ Export successful: {export_path}")
        return export_path
        
    except Exception as e:
        LOGGER.error(f"❌ Export failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Export QAT model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to QAT model (.pt)")
    parser.add_argument("--base", type=str, default="yolo11n.pt", help="Base YOLO model")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")
    parser.add_argument("--format", type=str, default="onnx", help="Export format")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    export_path = export_qat_model(str(model_path), args.base, args.imgsz, args.format)
    print(f"\n🎉 Success! Exported model to: {export_path}")


if __name__ == "__main__":
    main()

