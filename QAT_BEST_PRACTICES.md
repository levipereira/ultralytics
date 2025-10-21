# QAT Implementation - NVIDIA Best Practices Applied

This document summarizes the best practices from NVIDIA TensorRT Model Optimizer that have been implemented in the Ultralytics QAT module.

**Reference:** [NVIDIA TensorRT Model Optimizer - PyTorch Quantization](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html)

---

## ✅ Best Practices Implemented

### 1. **Quantization Summary Verification**
```python
mtq.print_quant_summary(self.quantized_model)
```
- **Why:** Verifies that quantizers are correctly placed in the model
- **When:** After `mtq.quantize()` call
- **Benefit:** Debug and validate quantizer placement before QAT

### 2. **Freeze Quantizer States During QAT**
```python
mtq.disable_quantizer(self.model, "*")
mtq.enable_quantizer(self.model, "*weight_quantizer")
mtq.enable_quantizer(self.model, "*input_quantizer")
```
- **Why:** NVIDIA recommendation - freeze quantizer parameters (scales/zero-points) from PTQ calibration
- **When:** Before QAT fine-tuning starts
- **Benefit:** Only model weights are fine-tuned, not quantizer parameters
- **Impact:** Better convergence and stability during QAT

### 3. **Dynamic Learning Rate (10% of Original)**
```python
original_lr = getattr(self.args, 'lr0', 0.01)
qat_lr = original_lr * 0.1  # 10% of original learning rate
```
- **Why:** NVIDIA recommendation for QAT fine-tuning
- **When:** Automatically calculated from original model's `lr0`
- **Benefit:** Optimal learning rate for quantized model fine-tuning
- **Previous:** Fixed at 1e-4 (not adaptive)

### 4. **QAT Epochs: 10% of Original Training**
```python
if qat_epochs > 50:
    qat_epochs = max(int(qat_epochs * 0.1), 5)  # 10% with minimum of 5 epochs
```
- **Why:** NVIDIA recommendation - QAT requires much less training than original
- **When:** Automatically calculated from `epochs` parameter
- **Benefit:** Efficient QAT without over-training
- **Note:** For LLMs, NVIDIA recommends even less than 1% of pre-training duration

### 5. **Multiple Quantization Schemes**
```python
if scheme == "int8":
    quant_cfg = mtq.INT8_DEFAULT_CFG  # CNN default
elif scheme == "int8_smoothquant":
    quant_cfg = mtq.INT8_SMOOTHQUANT_CFG  # Better for large models/LLMs
elif scheme == "fp8":
    quant_cfg = mtq.FP8_DEFAULT_CFG  # Modern GPUs (H100, A100)
```
- **Why:** Different models and hardware benefit from different quantization formats
- **Options:**
  - `int8`: Default for CNNs (YOLO models)
  - `int8_smoothquant`: Better for large models (reduces quantization error)
  - `fp8`: For modern GPUs with FP8 support
- **Usage:** `yolo detect qat model=yolo11n.pt quantization_scheme=int8_smoothquant`

### 6. **Proper Evaluation Mode**
```python
self.quantized_model.eval()
with torch.no_grad():
    validator = self.get_validator()
    self.ptq_metrics = validator()
```
- **Why:** PTQ evaluation should not compute gradients
- **When:** During PTQ model evaluation
- **Benefit:** Faster evaluation and correct behavior

### 7. **Calibration Method: 'max' (Default)**
```python
calibration_method: max  # nvidia-modelopt default
```
- **Why:** NVIDIA uses 'max' as default calibration algorithm
- **Previous:** Used 'minmax'
- **Alternatives:** `max`, `awq`, `smoothquant`, `histogram`

---

## 📊 Configuration Parameters

### Command Line Usage
```bash
# Basic QAT with defaults
yolo detect qat data=coco.yaml model=yolo11n.pt

# Custom QAT configuration
yolo detect qat data=coco.yaml model=yolo11n.pt \
    quantization_scheme=int8_smoothquant \
    calibration_samples=256 \
    epochs=10 \
    lr0=0.01

# FP8 quantization for modern GPUs
yolo detect qat data=coco.yaml model=yolo11n.pt \
    quantization_scheme=fp8 \
    batch=32
```

### Configuration Options
| Parameter | Default | Description | Best Practice |
|-----------|---------|-------------|---------------|
| `quantization_scheme` | `int8` | Quantization format | `int8` for CNNs, `int8_smoothquant` for large models, `fp8` for H100/A100 |
| `calibration_method` | `max` | Calibration algorithm | `max` (default), `awq` for weights, `smoothquant` for activations |
| `calibration_samples` | `512` | Number of calibration samples | 128-512 recommended by NVIDIA |
| `qat_lr` | `lr0 * 0.1` | QAT learning rate | Auto-calculated as 10% of original LR |
| `epochs` | `10% of original` | QAT fine-tuning epochs | Auto-calculated, minimum 5 epochs |
| `export_format` | `onnx` | Export format | `onnx` for TensorRT deployment |

---

## 🎯 QAT Workflow

The implementation follows the official NVIDIA workflow:

1. **FP32 Baseline Evaluation**
   - Load pre-trained FP32 model
   - Evaluate on validation set
   - Establish baseline metrics

2. **PTQ Quantization**
   - Apply `mtq.quantize()` with calibration
   - Insert quantizers (TensorQuantizer modules)
   - Calibrate using validation data subset
   - Print quantization summary

3. **PTQ Evaluation**
   - Evaluate quantized model (no fine-tuning yet)
   - Compare with FP32 baseline
   - Identify accuracy drop

4. **QAT Fine-Tuning**
   - Freeze quantizer states (scales/zero-points)
   - Fine-tune model weights only
   - Use 10% of original LR
   - Train for 10% of original epochs
   - Save best checkpoint

5. **Export Quantized Model**
   - Export to ONNX with quantization info
   - Ready for TensorRT deployment
   - Preserve quantization parameters

---

## 📈 Expected Results

### Typical Accuracy Recovery
- **PTQ (Post-Training Quantization):** 1-3% accuracy drop
- **QAT (After Fine-Tuning):** <1% accuracy drop (often recovers to FP32 level)

### Performance Gains (TensorRT Deployment)
- **INT8:** ~3-4x speedup, 75% memory reduction
- **FP8:** ~2x speedup (on H100/A100)

---

## 🔍 Debugging

### Verify Quantizer Placement
The quantization summary shows all inserted quantizers:
```
Quantization summary (verify quantizer placement):
module.0.conv.weight_quantizer: num_bits=8, type=static
module.0.conv.input_quantizer: num_bits=8, type=static
...
```

### Check Configuration
```bash
yolo detect qat data=coco8.yaml model=yolo11n.pt --verbose
```
Look for:
```
QAT Configuration: epochs=5, lr=0.001000, calibration_samples=512
Using INT8_DEFAULT_CFG for CNN quantization
```

---

## 📚 Additional Resources

- [NVIDIA TensorRT Model Optimizer Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/)
- [PyTorch Quantization Guide](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html)
- [Quantization Best Practices](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat)

---

## 🎓 Key Takeaways

1. ✅ Always print quantization summary to verify quantizer placement
2. ✅ Freeze quantizer states before QAT (NVIDIA best practice)
3. ✅ Use 10% of original LR for QAT fine-tuning
4. ✅ Use 10% of original epochs for QAT
5. ✅ Choose appropriate quantization scheme for your model/hardware
6. ✅ Use 128-512 calibration samples (more is not always better)
7. ✅ Evaluate PTQ before QAT to understand accuracy gap

---

**Implementation Status:** ✅ All NVIDIA best practices applied
**Last Updated:** 2025-10-21
**Reference Implementation:** ultralytics/engine/qat.py

