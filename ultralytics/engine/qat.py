# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics.utils import LOGGER, checks


class QATMixin:
    """
    Quantization Aware Training (QAT) trainer for YOLO models.
    
    This trainer extends BaseTrainer to implement quantization aware training using nvidia-modelopt.
    It follows the official nvidia-modelopt QAT workflow:
    1. Load FP32 model and evaluate baseline
    2. Apply PTQ quantization with calibration
    3. Fine-tune quantized model (QAT)
    4. Export quantized model
    
    Attributes:
        quantized_model: The quantized version of the model
        calibration_data: Data used for model calibration
        quantization_config: Configuration for quantization process
        fp32_metrics: Baseline FP32 model metrics
        ptq_metrics: PTQ model metrics
        
    Methods:
        setup_quantization: Initialize quantization modules
        calibrate_model: Calibrate the quantized model
        export_quantized: Export quantized model to ONNX
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize QAT trainer with quantization-specific attributes."""
        super().__init__(*args, **kwargs)
        self.quantized_model = None
        self.calibration_data = None
        self.quantization_config = {}
        self.fp32_metrics = None
        self.ptq_metrics = None
        
        # Initialize quantization-specific attributes
        self._setup_quantization_config()
    
    def _setup_quantization_config(self):
        """Setup default quantization configuration following nvidia-modelopt patterns."""
        self.quantization_config = {
            "quantization_scheme": "int8",  # Default to int8 quantization
            "calibration_method": "minmax",  # Default calibration method
            "export_format": "onnx",  # Default export format
            "calibration_samples": 512,  # Number of samples for calibration (following nvidia example)
            "qat_epochs": 5,  # Number of QAT fine-tuning epochs
            "qat_lr": 1e-4,  # Learning rate for QAT fine-tuning
        }
    
    def setup_quantization(self):
        """
        Setup quantization modules using nvidia-modelopt following official patterns.
        
        This method integrates with nvidia-modelopt to prepare the model for quantization
        using the same approach as the official torchvision_qat.py example.
        """
        try:
            # Import nvidia-modelopt modules following official example
            import modelopt.torch.quantization as mtq
            import modelopt.torch.opt as mto
            
            LOGGER.info("Setting up quantization with nvidia-modelopt...")
            
            # Use default INT8 configuration from nvidia-modelopt
            quant_cfg = mtq.INT8_DEFAULT_CFG
            
            # Create calibration function following nvidia example pattern
            def calibrate_fn(model):
                """Calibration function following nvidia-modelopt pattern."""
                model.eval()
                # Ensure model is on correct device
                model.to(self.device)
                seen = 0
                calibration_loader = self.get_calibration_dataloader()
                
                with torch.no_grad():
                    for batch in calibration_loader:
                        batch = self.preprocess_batch(batch)
                        # Ensure batch is on same device as model
                        imgs = batch["img"].to(self.device)
                        model(imgs)
                        seen += imgs.size(0)
                        if seen >= self.quantization_config["calibration_samples"]:
                            break
            
            # Apply PTQ quantization following official pattern
            self.quantized_model = mtq.quantize(
                self.model, 
                quant_cfg, 
                calibrate_fn
            )
            
            # Restore model attributes that may be lost during quantization
            # Args come from trainer, not model
            if not hasattr(self.quantized_model, 'args'):
                self.quantized_model.args = self.args
            
            # Copy other important attributes if they exist in original model
            for attr in ['names', 'stride', 'yaml', 'save', 'inplace']:
                if hasattr(self.model, attr) and not hasattr(self.quantized_model, attr):
                    setattr(self.quantized_model, attr, getattr(self.model, attr))
            
            LOGGER.info("Quantization setup completed successfully")
            
        except ImportError as e:
            LOGGER.error(f"nvidia-modelopt not available: {e}")
            raise ImportError("nvidia-modelopt is required for QAT functionality")
        except Exception as e:
            LOGGER.error(f"Failed to setup quantization: {e}")
            raise
    
    def evaluate_fp32_baseline(self):
        """
        Evaluate FP32 baseline model performance.
        
        Returns:
            dict: FP32 model metrics
        """
        LOGGER.info("Evaluating FP32 baseline...")
        
        try:
            # Ensure datasets are loaded
            if not hasattr(self, 'test_loader') or self.test_loader is None:
                self.test_loader = self.get_dataloader(
                    self.data["val"], 
                    batch_size=self.args.batch * 2, 
                    rank=-1, 
                    mode="val"
                )
            
            # Use existing validation infrastructure
            validator = self.get_validator()
            self.fp32_metrics = validator()
            
            LOGGER.info(f"FP32 baseline metrics: {self.fp32_metrics}")
            return self.fp32_metrics
            
        except Exception as e:
            LOGGER.error(f"FP32 evaluation failed: {e}")
            raise
    
    def evaluate_ptq_model(self):
        """
        Evaluate PTQ quantized model performance.
        
        Returns:
            dict: PTQ model metrics
        """
        if self.quantized_model is None:
            raise RuntimeError("Quantized model not available. Call setup_quantization() first.")
        
        LOGGER.info("Evaluating PTQ quantized model...")
        
        try:
            # Temporarily replace model for evaluation
            original_model = self.model
            self.model = self.quantized_model
            
            # Use existing validation infrastructure
            validator = self.get_validator()
            self.ptq_metrics = validator()
            
            # Restore original model
            self.model = original_model
            
            LOGGER.info(f"PTQ model metrics: {self.ptq_metrics}")
            return self.ptq_metrics
            
        except Exception as e:
            LOGGER.error(f"PTQ evaluation failed: {e}")
            raise
    
    def get_calibration_dataloader(self):
        """
        Get dataloader for calibration following nvidia-modelopt pattern.
        
        Uses validation dataset for calibration (following nvidia example).
        Reuses the existing dataloader infrastructure from BaseTrainer.
        """
        # Get validation dataloader for calibration
        # Using validation set with batch_size for calibration (following nvidia example)
        return self.get_dataloader(
            self.data["val"], 
            batch_size=self.batch_size, 
            rank=-1, 
            mode="val"
        )
    
    def qat_fine_tuning(self):
        """
        Perform QAT fine-tuning following nvidia-modelopt pattern.
        
        This method implements the QAT fine-tuning loop similar to the official example.
        """
        if self.quantized_model is None:
            raise RuntimeError("Quantized model not available. Call setup_quantization() first.")
        
        LOGGER.info("Starting QAT fine-tuning...")
        
        try:
            import modelopt.torch.opt as mto
            
            # Replace model with quantized version for QAT
            original_model = self.model
            self.model = self.quantized_model
            
            # Setup optimizer and scheduler for QAT (following nvidia example)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.quantization_config["qat_lr"],
                momentum=0.9,
                weight_decay=1e-4
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.quantization_config["qat_epochs"]
            )
            
            # QAT training loop
            best_metrics = None
            for epoch in range(1, self.quantization_config["qat_epochs"] + 1):
                LOGGER.info(f"QAT Epoch {epoch}/{self.quantization_config['qat_epochs']}")
                
                # Training step
                self._qat_train_epoch(optimizer)
                
                # Validation step
                validator = self.get_validator()
                metrics = validator()
                
                # Update scheduler
                scheduler.step()
                
                # Save best model
                if best_metrics is None or metrics.get("fitness", 0) > best_metrics.get("fitness", 0):
                    best_metrics = metrics
                    # Save QAT model using nvidia-modelopt save function
                    qat_path = self.save_dir / "best_qat.pt"
                    mto.save(self.model, str(qat_path))
                    LOGGER.info(f"Saved best QAT model: {qat_path}")
                
                LOGGER.info(f"QAT Epoch {epoch} metrics: {metrics}")
            
            # Restore original model reference
            self.model = original_model
            
            LOGGER.info("QAT fine-tuning completed successfully")
            return best_metrics
            
        except Exception as e:
            LOGGER.error(f"QAT fine-tuning failed: {e}")
            raise
    
    def _qat_train_epoch(self, optimizer):
        """
        Single QAT training epoch following nvidia-modelopt pattern.
        
        Args:
            optimizer: Optimizer for QAT training
        """
        self.model.train()
        
        # Get training dataloader
        train_loader = self.get_dataloader(
            self.data["train"], 
            batch_size=self.batch_size, 
            rank=-1, 
            mode="train"
        )
        
        for i, batch in enumerate(train_loader):
            # Preprocess batch
            batch = self.preprocess_batch(batch)
            
            # Forward pass
            loss, loss_items = self.model(batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress
            if i % 50 == 0:  # Log every 50 batches
                LOGGER.info(f"QAT Batch {i}: Loss = {loss.item():.4f}")
    
    def export_quantized(self, **kwargs):
        """
        Export quantized model to ONNX format following nvidia-modelopt pattern.
        
        Args:
            **kwargs: Additional export arguments
            
        Returns:
            str: Path to exported quantized model
        """
        if self.quantized_model is None:
            raise RuntimeError("Quantized model not available. Run setup_quantization() first.")
        
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
            
            # Temporarily replace model for export
            original_model = self.model
            self.model = self.quantized_model
            
            # Export quantized model
            exporter = Exporter(overrides=export_args, _callbacks=self.callbacks)
            export_path = exporter(model=self.quantized_model)
            
            # Restore original model
            self.model = original_model
            
            LOGGER.info(f"Quantized model exported to: {export_path}")
            return export_path
            
        except Exception as e:
            LOGGER.error(f"Export failed: {e}")
            raise
    
    def train(self):
        """
        Execute complete QAT workflow following nvidia-modelopt pattern.
        
        This method orchestrates the complete QAT workflow:
        1. Evaluate FP32 baseline
        2. Setup quantization (PTQ)
        3. Evaluate PTQ model
        4. QAT fine-tuning
        5. Export quantized model
        """
        LOGGER.info("Starting Quantization Aware Training (QAT) workflow...")
        
        try:
            # Step 1: Evaluate FP32 baseline
            fp32_metrics = self.evaluate_fp32_baseline()
            
            # Step 2: Setup quantization (PTQ)
            self.setup_quantization()
            
            # Step 3: Evaluate PTQ model
            ptq_metrics = self.evaluate_ptq_model()
            
            # Step 4: QAT fine-tuning
            qat_metrics = self.qat_fine_tuning()
            
            # Step 5: Export quantized model
            export_path = self.export_quantized()
            
            # Compile results
            results = {
                "fp32_metrics": fp32_metrics,
                "ptq_metrics": ptq_metrics,
                "qat_metrics": qat_metrics,
                "export_path": export_path,
                "quantization_config": self.quantization_config
            }
            
            LOGGER.info("QAT workflow completed successfully")
            LOGGER.info(f"Results: {results}")
            return results
            
        except Exception as e:
            LOGGER.error(f"QAT workflow failed: {e}")
            raise


# Task-specific QAT Trainers that inherit from both QATMixin and the task trainer
class DetectionQATTrainer(QATMixin):
    """QAT Trainer for detection tasks."""
    pass


class SegmentationQATTrainer(QATMixin):
    """QAT Trainer for segmentation tasks."""
    pass


class ClassificationQATTrainer(QATMixin):
    """QAT Trainer for classification tasks."""
    pass


class PoseQATTrainer(QATMixin):
    """QAT Trainer for pose estimation tasks."""
    pass


class OBBQATTrainer(QATMixin):
    """QAT Trainer for oriented bounding box tasks."""
    pass


def get_qat_trainer(task):
    """
    Get the appropriate QAT trainer for the specified task.
    
    Args:
        task (str): Task name (detect, segment, classify, pose, obb)
        
    Returns:
        QAT trainer class for the specified task
    """
    from ultralytics.models import yolo
    
    # Map tasks to their trainers
    task_trainer_map = {
        "detect": (yolo.detect.DetectionTrainer, DetectionQATTrainer),
        "segment": (yolo.segment.SegmentationTrainer, SegmentationQATTrainer),
        "classify": (yolo.classify.ClassificationTrainer, ClassificationQATTrainer),
        "pose": (yolo.pose.PoseTrainer, PoseQATTrainer),
        "obb": (yolo.obb.OBBTrainer, OBBQATTrainer),
    }
    
    if task not in task_trainer_map:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {list(task_trainer_map.keys())}")
    
    base_trainer, qat_trainer = task_trainer_map[task]
    
    # Dynamically create QAT trainer class that inherits from both QATMixin and base trainer
    # This ensures proper MRO (Method Resolution Order) with QATMixin first
    qat_trainer_class = type(
        f"{task.capitalize()}QATTrainer",
        (QATMixin, base_trainer),
        {"__module__": __name__}
    )
    
    return qat_trainer_class
