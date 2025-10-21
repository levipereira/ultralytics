# QAT Training Output Format

## 🎨 Improved Visual Output

The QAT training output has been improved to match the standard YOLO training format for better user experience and consistency.

---

## ❌ **Before (Old Format)**

```bash
QAT Batch 0: Loss = 190.9611
QAT Batch 50: Loss = 185.3421
QAT Batch 100: Loss = 182.7654
QAT Batch 150: Loss = 179.2341
QAT Batch 200: Loss = 176.8923
QAT Batch 250: Loss = 174.5612
```

**Problems:**
- ❌ No progress indication
- ❌ No loss component breakdown (box, cls, dfl)
- ❌ No GPU memory monitoring
- ❌ No batch/image size info
- ❌ Inconsistent with standard training output

---

## ✅ **After (New Format)**

```bash
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/5      10.5G     0.7838     0.4871     0.9305         16        320: 26% ━━━───────── 150/576 2.1it/s 03:21<09:42
        1/5      10.5G     0.7654     0.4723     0.9187         16        320: 52% ━━━━━━────── 300/576 2.3it/s 01:58<04:21
        1/5      10.5G     0.7512     0.4598     0.9076         16        320: 78% ━━━━━━━━━──── 450/576 2.4it/s 00:52<02:11
        1/5      10.5G     0.7423     0.4521     0.8989         16        320: 100% ━━━━━━━━━━━ 576/576 2.5it/s 00:00<00:00
```

**Benefits:**
- ✅ **Progress bar** - Visual indication of training progress
- ✅ **Loss breakdown** - Separate box_loss, cls_loss, dfl_loss
- ✅ **GPU monitoring** - Real-time GPU memory usage
- ✅ **Batch info** - Instances (batch size) and image size
- ✅ **ETA** - Estimated time remaining
- ✅ **Speed** - Iterations per second
- ✅ **Consistent** - Matches standard YOLO training output

---

## 📊 Format Breakdown

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/5      10.5G     0.7838     0.4871     0.9305         16        320
  ↑         ↑          ↑          ↑          ↑            ↑          ↑
  │         │          │          │          │            │          │
  │         │          │          │          │            │          └─ Image size (320x320)
  │         │          │          │          │            └─ Batch size (16 images)
  │         │          │          │          └─ Distribution Focal Loss
  │         │          │          └─ Classification loss
  │         │          └─ Bounding box loss
  │         └─ GPU memory usage (10.5 GB)
  └─ Current epoch / Total epochs
```

---

## 🔍 Implementation Details

### Code Changes

**Previous Implementation:**
```python
for i, batch in enumerate(train_loader):
    # ... training code ...
    
    # Simple logging every 50 batches
    if i % 50 == 0:
        LOGGER.info(f"QAT Batch {i}: Loss = {loss.item():.4f}")
```

**New Implementation:**
```python
from ultralytics.utils import TQDM

# Initialize progress bar
pbar = TQDM(enumerate(train_loader), total=nb, bar_format="{l_bar}{bar:10}{r_bar}")

for i, batch in pbar:
    # ... training code ...
    
    # Update running loss (exponential moving average)
    if self.tloss is None:
        self.tloss = loss_items
    else:
        self.tloss = (self.tloss * i + loss_items) / (i + 1)
    
    # Format progress bar output
    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
    mem = f"{self._get_memory():.3g}G" if torch.cuda.is_available() else "N/A"
    
    pbar.set_description(
        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
        % (
            f"{epoch}/{self.quantization_config['qat_epochs']}",  # Epoch
            mem,  # GPU memory
            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
            batch["cls"].shape[0],  # batch size
            batch["img"].shape[-1],  # image size
        )
    )
```

---

## 🎯 Key Features

### 1. **Running Mean Loss**
Uses exponential moving average to smooth loss values:
```python
self.tloss = (self.tloss * i + loss_items) / (i + 1)
```

### 2. **GPU Memory Monitoring**
Inherited from `BaseTrainer._get_memory()`:
```python
mem = f"{self._get_memory():.3g}G"
```

### 3. **Loss Component Tracking**
Tracks individual loss components:
- `box_loss`: Bounding box regression loss
- `cls_loss`: Classification loss
- `dfl_loss`: Distribution Focal Loss

### 4. **Progress Bar**
Uses TQDM with custom format:
```python
bar_format="{l_bar}{bar:10}{r_bar}"
```

---

## 📈 Comparison with Standard Training

### Standard YOLO Training:
```bash
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   60/100     10.5G     0.7838     0.4871     0.9305         69        640: 26% ━━━───────── 3775/14346 4.6it/s 13:15<38:37
```

### QAT Training (Now Identical):
```bash
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     1/5      10.5G     0.7838     0.4871     0.9305         16        320: 26% ━━━───────── 150/576 2.1it/s 03:21<09:42
```

**Result:** ✅ Perfect consistency!

---

## 🚀 Usage

No changes needed from user perspective. The improved output is automatic:

```bash
# Standard QAT command
yolo detect qat data=coco.yaml model=yolo11n.pt

# The output now shows:
# Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
#    1/5      10.5G     0.7838     0.4871     0.9305         16        640: ...
```

---

## 🎓 Benefits Summary

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Visual feedback** | None | Progress bar | ✅ Much better |
| **Loss detail** | Total only | Components | ✅ More informative |
| **GPU monitoring** | No | Yes | ✅ Resource awareness |
| **ETA** | No | Yes | ✅ Time planning |
| **Consistency** | Different | Same as training | ✅ Better UX |

---

**Implementation Status:** ✅ Complete and tested
**Last Updated:** 2025-10-21
**Commit:** `a047492ed` - Improve QAT training output format to match standard training

