# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics.utils import LOGGER, checks
from ultralytics.engine.trainer import BaseTrainer


class QATTrainer(BaseTrainer):
    """
    Quantization Aware Training (QAT) trainer for YOLO models.
    
    This trainer extends BaseTrainer to implement quantization aware training using nvidia-modelopt.
    It handles model quantization, calibration, and export to quantized formats.
    
    Attributes:
        quantized_model: The quantized version of the model
        calibration_data: Data used for model calibration
        quantization_config: Configuration for quantization process
        
    Methods:
        setup_quantization: Initialize quantization modules
        calibrate_model: Calibrate the quantized model
        export_quantized: Export quantized model to ONNX
    """
    
    def __init__(self, cfg=None, overrides: Optional[Dict[str, Any]] = None, _callbacks=None):
        """
        Initialize QAT trainer.
        
        Args:
            cfg: Configuration object
            overrides: Configuration overrides
            _callbacks: Callback functions
        """
        super().__init__(cfg, overrides, _callbacks)
        self.quantized_model = None
        self.calibration_data = None
        self.quantization_config = {}
        
        # Initialize quantization-specific attributes
        self._setup_quantization_config()
    
    def _setup_quantization_config(self):
        """Setup default quantization configuration."""
        self.quantization_config = {
            "quantization_scheme": "int8",  # Default to int8 quantization
            "calibration_method": "minmax",  # Default calibration method
            "export_format": "onnx",  # Default export format
            "calibration_samples": 100,  # Number of samples for calibration
        }
    
    def setup_quantization(self):
        """
        Setup quantization modules using nvidia-modelopt.
        
        This method will integrate with nvidia-modelopt to prepare the model for quantization.
        """
        try:
            # Import nvidia-modelopt modules
            from nvidia_modelopt import QuantizationConfig, QuantizationScheme
            from nvidia_modelopt.torch import quantize_model
            
            LOGGER.info("Setting up quantization with nvidia-modelopt...")
            
            # Create quantization configuration
            quant_config = QuantizationConfig(
                scheme=QuantizationScheme.INT8,
                calibration_method=self.quantization_config["calibration_method"]
            )
            
            # Prepare model for quantization
            self.quantized_model = quantize_model(
                self.model,
                quant_config
            )
            
            LOGGER.info("Quantization setup completed successfully")
            
        except ImportError as e:
            LOGGER.error(f"nvidia-modelopt not available: {e}")
            raise ImportError("nvidia-modelopt is required for QAT functionality")
        except Exception as e:
            LOGGER.error(f"Failed to setup quantization: {e}")
            raise
    
    def calibrate_model(self):
        """
        Calibrate the quantized model using calibration data.
        
        This method runs calibration to determine quantization parameters.
        """
        if self.quantized_model is None:
            raise RuntimeError("Quantization not setup. Call setup_quantization() first.")
        
        LOGGER.info("Starting model calibration...")
        
        try:
            # Get calibration dataloader
            calibration_loader = self.get_calibration_dataloader()
            
            # Run calibration
            with torch.no_grad():
                for i, batch in enumerate(calibration_loader):
                    if i >= self.quantization_config["calibration_samples"]:
                        break
                    
                    # Forward pass for calibration
                    batch = self.preprocess_batch(batch)
                    _ = self.quantized_model(batch["img"])
            
            LOGGER.info("Model calibration completed successfully")
            
        except Exception as e:
            LOGGER.error(f"Calibration failed: {e}")
            raise
    
    def get_calibration_dataloader(self):
        """
        Get dataloader for calibration.
        
        Reuses the existing dataloader infrastructure from BaseTrainer.
        """
        # Reuse existing dataloader setup
        return self.get_dataloader(self.train_dataset, batch=self.batch_size, rank=-1, mode="train")
    
    def export_quantized(self, **kwargs):
        """
        Export quantized model to ONNX format.
        
        Args:
            **kwargs: Additional export arguments
            
        Returns:
            str: Path to exported quantized model
        """
        if self.quantized_model is None:
            raise RuntimeError("Quantized model not available. Run setup_quantization() and calibrate_model() first.")
        
        LOGGER.info("Exporting quantized model...")
        
        try:
            # Use existing export infrastructure but with quantized model
            from ultralytics.engine.exporter import Exporter
            
            # Prepare export arguments
            export_args = {
                "format": self.quantization_config["export_format"],
                "imgsz": self.args.imgsz,
                "batch": 1,
                "device": self.device,
                "verbose": True,
                **kwargs
            }
            
            # Export quantized model
            exporter = Exporter(overrides=export_args, _callbacks=self.callbacks)
            export_path = exporter(model=self.quantized_model)
            
            LOGGER.info(f"Quantized model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            LOGGER.error(f"Export failed: {e}")
            raise
    
    def train(self):
        """
        Execute QAT training process.
        
        This method orchestrates the complete QAT workflow:
        1. Setup quantization
        2. Calibrate model
        3. Export quantized model
        """
        LOGGER.info("Starting Quantization Aware Training (QAT)...")
        
        try:
            # Step 1: Setup quantization
            self.setup_quantization()
            
            # Step 2: Calibrate model
            self.calibrate_model()
            
            # Step 3: Export quantized model
            export_path = self.export_quantized()
            
            LOGGER.info("QAT process completed successfully")
            return {"export_path": export_path}
            
        except Exception as e:
            LOGGER.error(f"QAT process failed: {e}")
            raise
