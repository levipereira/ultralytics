# QAT Pickle Error - Technical Analysis & Solution

## ❌ **Error Encountered**

```python
_pickle.PicklingError: Can't pickle <class 'modelopt.torch.opt.dynamic.QuantConv2d'>: 
attribute lookup QuantConv2d on modelopt.torch.opt.dynamic failed
```

**Location:** `qat.py:318` - During `torch.save({'model': self.model})`

---

## 🔍 **Root Cause Analysis**

### **What Happened:**

```python
# Our original code (WRONG):
torch.save({
    'model': self.model,  # ❌ Trying to pickle model with dynamic classes
    'model_state_dict': self.model.state_dict(),
    ...
}, "best_qat_full.pt")
```

### **Why It Failed:**

1. **ModelOpt Creates Dynamic Classes**
   ```python
   # During mtq.quantize(), ModelOpt does:
   import modelopt.torch.opt as mto
   
   # Creates dynamic class at RUNTIME
   QuantConv2d = type(
       'QuantConv2d',
       (nn.Conv2d,),
       {'forward': quantized_forward, ...}
   )
   
   # Replace Conv2d with QuantConv2d
   model.conv = QuantConv2d(...)
   ```

2. **Pickle Cannot Serialize Dynamic Classes**
   ```python
   import pickle
   
   # Static class (defined in module) - OK
   class StaticConv2d(nn.Module):
       pass
   
   pickle.dumps(StaticConv2d)  # ✅ Works
   
   # Dynamic class (created at runtime) - FAIL
   DynamicConv2d = type('DynamicConv2d', (nn.Module,), {})
   pickle.dumps(DynamicConv2d)  # ❌ PicklingError
   ```

3. **torch.save() Uses Pickle**
   ```python
   torch.save({'model': model})
   # Internally calls pickle.dump(model)
   # → Tries to serialize QuantConv2d class
   # → Pickle can't find QuantConv2d in modelopt.torch.opt.dynamic
   # → Error!
   ```

---

## ✅ **Solution: Save state_dict, Not Model Object**

### **Key Insight:**

**PyTorch `state_dict()` includes registered buffers, which contain calibration values!**

```python
# Model structure
class TensorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        # These ARE in state_dict (registered buffers)
        self.register_buffer('_amax', torch.tensor(0.0))
        self.register_buffer('_scale', torch.tensor(1.0))
        self.register_buffer('_zero_point', torch.tensor(0))

# After calibration
quantizer._amax = torch.tensor([171.3999])  # Set during calibration

# state_dict includes _amax!
state = model.state_dict()
print(state['model.0.conv._input_quantizer._amax'])
# → tensor([171.3999])  ✅ Calibration preserved!
```

### **Corrected Implementation:**

#### **1. Save (qat.py:315-335):**

```python
# Method 3: Save state_dict + metadata (NO model object)
qat_full_path = self.save_dir / "best_qat_full.pt"

# Get complete state dict including all buffers (_amax, _scale, etc.)
full_state = self.model.state_dict()

# Save ONLY serializable data
torch.save({
    'model_state_dict': full_state,  # ✅ Includes _amax buffers
    'modelopt_state': mto.modelopt_state(self.model),  # ✅ Architecture info
    'quantization_config': self.quantization_config,
    'metrics': best_metrics,
    'is_quantized': True,
    'task': getattr(self.model, 'task', 'detect'),
    'names': getattr(self.model, 'names', None),
    'stride': getattr(self.model, 'stride', None),
    'yaml': getattr(self.model, 'yaml', None),
}, str(qat_full_path))
# ✅ No pickle error (no model object)
# ✅ Has calibration (_amax in state_dict)
```

#### **2. Load (tasks.py:1596-1602):**

```python
# Restore architecture using modelopt_state
base_model = YOLO("yolo11n.pt").model
model = mto.restore_from_modelopt_state(base_model, ckpt["modelopt_state"])

# Load weights + calibration buffers
model.load_state_dict(ckpt["model_state_dict"])
# ✅ Architecture restored
# ✅ Weights loaded
# ✅ Calibration (_amax) loaded from buffers
```

---

## 📊 **Comparison: What Gets Saved**

| Method | Model Object | state_dict | _amax (calibration) | Pickle-Safe? |
|--------|--------------|------------|---------------------|--------------|
| **torch.save(model)** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ **NO** (dynamic classes) |
| **torch.save(state_dict)** | ❌ No | ✅ Yes | ✅ Yes (in buffers) | ✅ **YES** |
| **mto.save()** | ❌ No | ✅ Yes | ❌ No (not in state_dict) | ✅ YES |

**Winner:** `torch.save(state_dict)` - Has calibration + pickle-safe!

---

## 🔬 **Technical Deep Dive**

### **Why state_dict Includes _amax:**

```python
# In TensorQuantizer.__init__():
self.register_buffer('_amax', torch.tensor(0.0))

# register_buffer() tells PyTorch:
# 1. This is a non-trainable tensor (not a parameter)
# 2. Include it in state_dict()
# 3. Move it with .to(device)
# 4. Serialize it with torch.save()

# During calibration:
quantizer._amax = torch.max(torch.abs(input))
# Updates the registered buffer

# When we call state_dict():
state = model.state_dict()
# Includes: 'module.quantizer._amax': tensor([171.3999])
```

### **Why We Can't Pickle Dynamic Classes:**

```python
# Pickle works by storing:
# 1. Module name: 'modelopt.torch.opt.dynamic'
# 2. Class name: 'QuantConv2d'
# 3. Attributes: {...}

# On unpickling:
# import modelopt.torch.opt.dynamic
# cls = getattr(modelopt.torch.opt.dynamic, 'QuantConv2d')
# ^ AttributeError: QuantConv2d doesn't exist!

# Why? Because QuantConv2d was created dynamically
# It's not defined in the module, it was created at runtime
```

### **NVIDIA's Solution:**

From [NVIDIA docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_save_load.html):

> "ModelOpt supports saving the architecture modifications together with model weights or separately."

**Two methods:**
1. **Together:** `mto.save(model, path)` - Saves modelopt_state + state_dict
2. **Separately:** `torch.save(mto.modelopt_state(model))` + `torch.save(model.state_dict())`

**Both avoid pickling the model object!**

---

## 📝 **Three-Tier Save Strategy**

### **File 1: best_qat.pt**
```python
mto.save(model, "best_qat.pt")
```
- **Format:** ModelOpt proprietary
- **Contains:** modelopt_state + state_dict
- **Calibration:** ❌ No (_amax not in state_dict by default)
- **Use:** Continue training, mto.restore()

### **File 2: best_qat_modelopt_state.pth**
```python
torch.save(mto.modelopt_state(model), "best_qat_modelopt_state.pth")
```
- **Format:** Pure Python dict
- **Contains:** Architecture modifications only
- **Calibration:** ❌ No
- **Use:** Custom restore workflows

### **File 3: best_qat_full.pt** ✨ **NEW**
```python
torch.save({
    'model_state_dict': model.state_dict(),  # ← Has _amax!
    'modelopt_state': mto.modelopt_state(model),
    ...metadata...
}, "best_qat_full.pt")
```
- **Format:** PyTorch checkpoint
- **Contains:** state_dict + modelopt_state + metadata
- **Calibration:** ✅ **YES** (_amax in buffers)
- **Use:** INT8 ONNX export
- **Pickle-Safe:** ✅ **YES** (no model object)

---

## 🎯 **Workflow Examples**

### **Example 1: Train → Export INT8**

```bash
# 1. Train QAT
yolo detect qat data=coco.yaml model=yolo11n.pt epochs=10

# Files created:
# ✅ best_qat.pt                   (mto.save)
# ✅ best_qat_modelopt_state.pth   (architecture)
# ✅ best_qat_full.pt              (state_dict with _amax) ← Use this!

# 2. Export INT8 ONNX
yolo export model=runs/detect/qat/best_qat_full.pt format=onnx

# Load process:
# 1. Load modelopt_state from best_qat_full.pt
# 2. mto.restore_from_modelopt_state(base_model, modelopt_state)
# 3. model.load_state_dict(checkpoint['model_state_dict'])
# 4. Export with calibration (_amax from buffers)
# ✅ INT8 ONNX with quantization ops!
```

### **Example 2: Continue Training**

```bash
# Use best_qat.pt (ModelOpt format)
yolo detect train model=runs/detect/qat/best_qat.pt epochs=5
```

### **Example 3: Custom Restore**

```python
import torch
import modelopt.torch.opt as mto
from ultralytics import YOLO

# Load checkpoint
ckpt = torch.load("best_qat_full.pt")

# Get base model
base_model = YOLO("yolo11n.pt").model

# Restore architecture
model = mto.restore_from_modelopt_state(base_model, ckpt['modelopt_state'])

# Load weights + calibration
model.load_state_dict(ckpt['model_state_dict'])

# Verify calibration
for name, module in model.named_modules():
    if hasattr(module, '_input_quantizer'):
        if hasattr(module._input_quantizer, '_amax'):
            print(f"{name}: _amax = {module._input_quantizer._amax}")
# ✅ Prints calibration values!
```

---

## 🐛 **Troubleshooting**

### **Problem: "Can't pickle QuantConv2d"**
```
Solution: ✅ Fixed - we now save state_dict, not model object
```

### **Problem: "Missing calibration after loading"**
```python
# Check if _amax is in state_dict
ckpt = torch.load("best_qat_full.pt")
amax_keys = [k for k in ckpt['model_state_dict'].keys() if '_amax' in k]
print(f"Found {len(amax_keys)} _amax buffers")
# Should be > 0 ✅
```

### **Problem: "modelopt_state not found"**
```
Cause: Old checkpoint format
Solution: Re-train with updated code
```

---

## 📚 **References**

1. **NVIDIA ModelOpt Save/Restore:** https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_save_load.html
2. **PyTorch Pickling:** https://pytorch.org/docs/stable/notes/serialization.html
3. **Dynamic Class Creation:** https://docs.python.org/3/library/functions.html#type

---

## ✅ **Summary**

| Aspect | Before (Wrong) | After (Correct) |
|--------|---------------|-----------------|
| **Save method** | torch.save({'model': model}) | torch.save({'model_state_dict': state_dict}) |
| **Pickle error** | ❌ Yes | ✅ No |
| **Has calibration** | ✅ Yes (_amax in model) | ✅ Yes (_amax in buffers) |
| **Can export INT8** | ❌ No (pickle error) | ✅ Yes |
| **NVIDIA compliant** | ❌ No | ✅ Yes |

**Lesson Learned:** 
- Don't pickle models with dynamic classes
- Use `state_dict()` + `modelopt_state()` instead
- Calibration buffers (_amax) ARE in state_dict
- Follow NVIDIA's separate save/restore pattern

---

**Status:** ✅ Pickle error resolved, calibration preserved, INT8 export working!

**Commit:** `388da79df` - Fix pickle error: cannot serialize dynamic ModelOpt classes

