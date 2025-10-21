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
        # Get user-specified config from args if available
        calibration_samples = getattr(self.args, 'calibration_samples', 512)
        qat_epochs = getattr(self.args, 'epochs', 100)
        
        # Following NVIDIA best practices: QAT for 10% of original training epochs
        # For pre-trained models, use a reasonable default
        if qat_epochs > 50:
            qat_epochs = max(int(qat_epochs * 0.1), 5)  # 10% with minimum of 5 epochs
        
        # Following NVIDIA best practices: Use 10% of original learning rate for QAT
        # Get the original learning rate from args
        original_lr = getattr(self.args, 'lr0', 0.01)
        qat_lr = original_lr * 0.1  # 10% of original learning rate
        
        self.quantization_config = {
            "quantization_scheme": getattr(self.args, 'quantization_scheme', 'int8'),
            "calibration_method": getattr(self.args, 'calibration_method', 'max'),  # 'max' is default in nvidia-modelopt
            "export_format": getattr(self.args, 'export_format', 'onnx'),
            "calibration_samples": calibration_samples,
            "qat_epochs": qat_epochs,
            "qat_lr": qat_lr,
        }
        
        LOGGER.info(f"QAT Configuration: epochs={qat_epochs}, lr={qat_lr:.6f}, calibration_samples={calibration_samples}")
    
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
            
            # Select quantization config based on scheme (following NVIDIA best practices)
            scheme = self.quantization_config["quantization_scheme"].lower()
            if scheme == "int8":
                quant_cfg = mtq.INT8_DEFAULT_CFG
                LOGGER.info("Using INT8_DEFAULT_CFG for CNN quantization")
            elif scheme == "int8_smoothquant":
                quant_cfg = mtq.INT8_SMOOTHQUANT_CFG
                LOGGER.info("Using INT8_SMOOTHQUANT_CFG (better for large models)")
            elif scheme == "fp8":
                quant_cfg = mtq.FP8_DEFAULT_CFG
                LOGGER.info("Using FP8_DEFAULT_CFG (requires modern GPUs)")
            else:
                LOGGER.warning(f"Unknown quantization scheme '{scheme}', falling back to INT8_DEFAULT_CFG")
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
            
            # Following NVIDIA best practices: Print quantization summary to verify quantizer placement
            LOGGER.info("Quantization summary (verify quantizer placement):")
            mtq.print_quant_summary(self.quantized_model)
            
            # Restore model attributes that may be lost during quantization
            # Args come from trainer, not model
            if not hasattr(self.quantized_model, 'args'):
                self.quantized_model.args = self.args
            
            # Copy other important attributes if they exist in original model
            # Including 'task' which is required for export
            for attr in ['names', 'stride', 'yaml', 'save', 'inplace', 'task']:
                if hasattr(self.model, attr) and not hasattr(self.quantized_model, attr):
                    setattr(self.quantized_model, attr, getattr(self.model, attr))
            
            # Mark model as quantized for automatic detection during export
            self.quantized_model._is_quantized = True
            LOGGER.info("Model marked as quantized (_is_quantized=True) for export detection")
            
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
            # Following NVIDIA best practices: Set model to eval mode for PTQ evaluation
            self.quantized_model.eval()
            
            # Temporarily replace model for evaluation
            original_model = self.model
            self.model = self.quantized_model
            
            # Use existing validation infrastructure
            with torch.no_grad():  # Following best practices: no gradients during PTQ evaluation
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
        Following NVIDIA best practices: freeze quantizer states during QAT fine-tuning.
        """
        if self.quantized_model is None:
            raise RuntimeError("Quantized model not available. Call setup_quantization() first.")
        
        LOGGER.info("Starting QAT fine-tuning...")
        
        try:
            import modelopt.torch.quantization as mtq
            import modelopt.torch.opt as mto
            
            # Replace model with quantized version for QAT
            original_model = self.model
            self.model = self.quantized_model
            
            # Following NVIDIA best practices: Disable quantizers before QAT
            # This freezes the quantizer parameters (scales/zero-points) from PTQ calibration
            # Only the model weights will be fine-tuned during QAT
            LOGGER.info("Freezing quantizer states (following NVIDIA best practices)...")
            mtq.disable_quantizer(self.model, "*")
            mtq.enable_quantizer(self.model, "*weight_quantizer")  # Keep weight quantizers enabled
            mtq.enable_quantizer(self.model, "*input_quantizer")   # Keep input quantizers enabled
            
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
            
            # Initialize loss tracking (for formatted output like training)
            self.tloss = None
            
            # QAT training loop
            best_metrics = None
            for epoch in range(1, self.quantization_config["qat_epochs"] + 1):
                self.epoch = epoch  # Store current epoch for display
                
                # Training step
                self._qat_train_epoch(optimizer, epoch)
                
                # Validation step
                validator = self.get_validator()
                metrics = validator()
                
                # Update scheduler
                scheduler.step()
                
                # Save best model
                if best_metrics is None or metrics.get("fitness", 0) > best_metrics.get("fitness", 0):
                    best_metrics = metrics
                    
                    # Following NVIDIA ModelOpt best practices for saving:
                    # https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_save_load.html
                    
                    # Method 1: mto.save() - Saves modelopt_state + weights (but NOT calibration _amax)
                    qat_path = self.save_dir / "best_qat.pt"
                    mto.save(self.model, str(qat_path))
                    LOGGER.info(f"Saved QAT model (ModelOpt state + weights): {qat_path}")
                    
                    # Method 2: Save modelopt_state separately + full model (NVIDIA recommended)
                    # This preserves calibration values (_amax) in the full model
                    modelopt_state_path = self.save_dir / "best_qat_modelopt_state.pth"
                    torch.save(mto.modelopt_state(self.model), str(modelopt_state_path))
                    LOGGER.info(f"Saved ModelOpt state separately: {modelopt_state_path}")
                    
                    # Save complete model with ALL attributes (including _amax calibration)
                    qat_full_path = self.save_dir / "best_qat_full.pt"
                    torch.save({
                        'model': self.model,  # Full model with calibration
                        'model_state_dict': self.model.state_dict(),
                        'modelopt_state': mto.modelopt_state(self.model),  # ModelOpt state
                        'quantization_config': self.quantization_config,
                        'metrics': best_metrics,
                        'is_quantized': True,
                        'task': self.model.task,
                        'names': self.model.names,
                        'stride': self.model.stride,
                    }, str(qat_full_path))
                    LOGGER.info(f"Saved complete QAT model (with calibration): {qat_full_path}")
                
                LOGGER.info(f"QAT Epoch {epoch} metrics: {metrics}")
            
            # Restore original model reference
            self.model = original_model
            
            LOGGER.info("QAT fine-tuning completed successfully")
            return best_metrics
            
        except Exception as e:
            LOGGER.error(f"QAT fine-tuning failed: {e}")
            raise
    
    def _qat_train_epoch(self, optimizer, epoch):
        """
        Single QAT training epoch following nvidia-modelopt pattern.
        
        Args:
            optimizer: Optimizer for QAT training
            epoch: Current epoch number
        """
        from ultralytics.utils import TQDM
        
        self.model.train()
        
        # Get training dataloader
        train_loader = self.get_dataloader(
            self.data["train"], 
            batch_size=self.batch_size, 
            rank=-1, 
            mode="train"
        )
        
        # Initialize progress bar similar to standard training
        nb = len(train_loader)  # number of batches
        pbar = TQDM(enumerate(train_loader), total=nb, bar_format="{l_bar}{bar:10}{r_bar}")
        
        for i, batch in pbar:
            # Preprocess batch
            batch = self.preprocess_batch(batch)
            
            # Forward pass
            loss, loss_items = self.model(batch)
            
            # Ensure loss is scalar (sum if it's a tensor with multiple values)
            if loss.numel() > 1:
                loss = loss.sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update running loss (exponential moving average)
            if self.tloss is None:
                self.tloss = loss_items
            else:
                self.tloss = (self.tloss * i + loss_items) / (i + 1)  # update mean losses
            
            # Format progress bar output similar to training
            # Format: Epoch/Total  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
            loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
            mem = f"{self._get_memory():.3g}G" if torch.cuda.is_available() else "N/A"
            
            pbar.set_description(
                ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                % (
                    f"{epoch}/{self.quantization_config['qat_epochs']}",  # Epoch
                    mem,  # GPU memory
                    *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                    batch["cls"].shape[0],  # batch size (instances)
                    batch["img"].shape[-1],  # image size
                )
            )
    
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
