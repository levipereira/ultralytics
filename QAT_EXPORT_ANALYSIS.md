# QAT Export vs Standard Export - Análise e Solução

## 🔍 Problema Identificado

**Pergunta:** O export QAT é diferente do export padrão? Como diferenciar quando o usuário exportar um modelo quantizado vs comum?

---

## 📊 Análise Atual

### **1. Export Padrão (Atual)**

```bash
# Usuário exporta modelo FP32 normal
yolo export model=yolo11n.pt format=onnx

# Internamente:
# - Carrega modelo PyTorch (.pt)
# - model.fuse() - faz fusion de layers
# - Exporta para ONNX sem quantização
```

### **2. Export QAT (Implementado)**

```bash
# QAT workflow completo (inclui export automático)
yolo detect qat data=coco.yaml model=yolo11n.pt
# → Resultado: best_qat.pt (com 428 quantizers)

# Para re-exportar modelo QAT salvo:
python test_qat_export.py --model best_qat.pt --base yolo11n.pt
```

---

## ⚠️ **Problema: Ambiguidade**

### **Cenário Problemático:**

```bash
# Usuário tem um modelo QAT salvo
runs/detect/qat16/best_qat.pt  # Modelo COM quantização

# Se o usuário tentar exportar com comando padrão:
yolo export model=runs/detect/qat16/best_qat.pt format=onnx

# O que acontece?
# ❌ PROBLEMA: Vai tentar exportar como modelo comum
# ❌ Vai PERDER os 428 quantizers
# ❌ Vai gerar ONNX FP32 ao invés de INT8 quantizado
```

---

## 💡 **Soluções Propostas**

### **Opção 1: Detecção Automática (RECOMENDADA)**

Detectar automaticamente se o modelo é quantizado e usar o export apropriado.

```python
# No Exporter.__call__()
def __call__(self, model=None):
    # ... código existente ...
    
    # Detect if model is quantized
    is_quantized = self._is_quantized_model(model)
    
    if is_quantized:
        LOGGER.info("🔢 Quantized model detected, using QAT export...")
        return self._export_quantized_model(model)
    else:
        # Continue with standard export
        ...

def _is_quantized_model(self, model):
    """Check if model has quantization layers from nvidia-modelopt."""
    try:
        import modelopt.torch.quantization as mtq
        # Check for TensorQuantizer modules
        for module in model.modules():
            if hasattr(module, '_input_quantizer') or \
               hasattr(module, '_weight_quantizer'):
                return True
        return False
    except ImportError:
        return False
```

**Vantagens:**
- ✅ Transparente para o usuário
- ✅ Não quebra workflows existentes
- ✅ Preserva quantização automaticamente

**Desvantagens:**
- ⚠️ Usuário pode não perceber que é quantizado

---

### **Opção 2: Flag Explícita**

Adicionar flag `quantized=True` ao export.

```bash
# Export padrão
yolo export model=yolo11n.pt format=onnx

# Export quantizado (explícito)
yolo export model=best_qat.pt format=onnx quantized=True
```

**Vantagens:**
- ✅ Controle explícito do usuário
- ✅ Clara separação de funcionalidades

**Desvantagens:**
- ❌ Usuário pode esquecer a flag e perder quantização
- ❌ Mais verboso

---

### **Opção 3: Comando Separado (ATUAL)**

Manter comando separado para export QAT.

```bash
# Export padrão
yolo export model=yolo11n.pt format=onnx

# Export QAT (script separado)
python test_qat_export.py --model best_qat.pt --base yolo11n.pt
```

**Vantagens:**
- ✅ Separação clara
- ✅ Já implementado

**Desvantagens:**
- ❌ Inconsistente com o resto da CLI
- ❌ Usuário precisa saber usar script separado
- ❌ Requer modelo base (`--base`)

---

### **Opção 4: Híbrida (MELHOR SOLUÇÃO)**

Combinar detecção automática + flag opcional + integração CLI.

```bash
# 1. Export com detecção automática (mais simples)
yolo export model=best_qat.pt format=onnx
# → Detecta automaticamente quantização
# → Exporta preservando 428 quantizers
# → Log: "🔢 Quantized model detected (428 quantizers)"

# 2. Export forçando QAT (explícito, se necessário)
yolo export model=best_qat.pt format=onnx quantized=True

# 3. Export desabilitando QAT (se quiser FP32)
yolo export model=best_qat.pt format=onnx quantized=False
# → Remove quantização e exporta como FP32
```

**Vantagens:**
- ✅ Funciona automaticamente (melhor UX)
- ✅ Permite controle explícito quando necessário
- ✅ Consistente com CLI do YOLO
- ✅ Não quebra workflows existentes

---

## 🎯 **Implementação Recomendada: Opção 4**

### **Arquivo: `ultralytics/engine/exporter.py`**

```python
class Exporter:
    def __call__(self, model=None):
        """Export model with automatic quantization detection."""
        # ... setup existente ...
        
        # Load model if path provided
        if isinstance(model, (str, Path)):
            model = self._load_model(model)
        
        # Detect quantization
        is_quantized = self._detect_quantization(model)
        
        # Handle quantized export
        if is_quantized and self.args.get('quantized', True):
            LOGGER.info(f"🔢 Quantized model detected, preserving quantization...")
            return self._export_with_quantization(model)
        elif is_quantized and not self.args.get('quantized', True):
            LOGGER.warning("⚠️ Quantized model detected but quantized=False, removing quantization...")
            model = self._dequantize_model(model)
        
        # Standard export
        # ... código existente ...
    
    def _detect_quantization(self, model):
        """
        Detect if model has quantization from nvidia-modelopt.
        
        Returns:
            bool: True if model is quantized
        """
        try:
            # Check for TensorQuantizer modules
            for name, module in model.named_modules():
                module_type = type(module).__name__
                if 'Quantizer' in module_type or 'TensorQuantizer' in module_type:
                    LOGGER.info(f"Found quantization layer: {name} ({module_type})")
                    return True
            
            # Check for nvidia-modelopt attributes
            if hasattr(model, '_is_quantized') and model._is_quantized:
                return True
                
            return False
        except Exception as e:
            LOGGER.debug(f"Quantization detection failed: {e}")
            return False
    
    def _export_with_quantization(self, model):
        """
        Export model preserving quantization information.
        
        For ONNX: Export with quantization ops
        For TensorRT: Use INT8 calibration
        """
        try:
            import modelopt.torch.quantization as mtq
            
            # Print quantization summary
            LOGGER.info("Quantization configuration:")
            mtq.print_quant_summary(model)
            
            # Export based on format
            if self.args.format == 'onnx':
                return self._export_quantized_onnx(model)
            elif self.args.format == 'engine':  # TensorRT
                return self._export_quantized_tensorrt(model)
            else:
                LOGGER.warning(f"Format '{self.args.format}' may not fully support quantization")
                return self._standard_export(model)
                
        except ImportError:
            LOGGER.error("nvidia-modelopt not available, falling back to standard export")
            return self._standard_export(model)
    
    def _export_quantized_onnx(self, model):
        """Export quantized model to ONNX with quantization ops."""
        # Use existing ONNX export but ensure quantization is preserved
        # ... implementation ...
        pass
    
    def _dequantize_model(self, model):
        """Remove quantization from model (convert back to FP32)."""
        try:
            import modelopt.torch.quantization as mtq
            # Remove quantizers
            # ... implementation ...
            return model
        except ImportError:
            return model
```

---

## 📝 **Uso Após Implementação**

### **Caso 1: Modelo FP32 Normal**
```bash
yolo export model=yolo11n.pt format=onnx
# Output: yolo11n.onnx (FP32, ~10.5 MB)
```

### **Caso 2: Modelo QAT (Detecção Automática)**
```bash
yolo export model=runs/detect/qat16/best_qat.pt format=onnx
# Output:
# 🔢 Quantized model detected, preserving quantization...
# Found quantization layer: model.0.conv.weight_quantizer
# Quantization configuration:
#   Inserted 428 quantizers
# ✅ Export success: best_qat.onnx (INT8, ~10.5 MB with quant ops)
```

### **Caso 3: Forçar FP32 de Modelo QAT**
```bash
yolo export model=best_qat.pt format=onnx quantized=False
# Output:
# ⚠️ Quantized model detected but quantized=False, removing quantization...
# ✅ Export success: best_qat.onnx (FP32, ~10.5 MB)
```

### **Caso 4: TensorRT com INT8**
```bash
yolo export model=best_qat.pt format=engine
# Output:
# 🔢 Quantized model detected, using INT8 calibration...
# ✅ Export success: best_qat.engine (INT8, TensorRT optimized)
```

---

## 🔧 **Modificações Necessárias**

### **Arquivos a Modificar:**

1. **`ultralytics/engine/exporter.py`**
   - Adicionar `_detect_quantization()`
   - Adicionar `_export_with_quantization()`
   - Adicionar `_export_quantized_onnx()`
   - Adicionar `_dequantize_model()`
   - Modificar `__call__()` para incluir detecção

2. **`ultralytics/cfg/default.yaml`**
   ```yaml
   # Export settings
   quantized: null  # (bool) preserve quantization on export (auto-detect if None)
   ```

3. **`ultralytics/engine/qat.py`**
   - Marcar modelo como quantizado: `model._is_quantized = True`
   - Atualizar `export_quantized()` para usar novo fluxo

4. **Documentação**
   - Atualizar `QAT_BEST_PRACTICES.md`
   - Criar seção sobre export de modelos quantizados

---

## 🎯 **Benefícios da Solução Híbrida**

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Facilidade** | Script separado | Comando único |
| **Detecção** | Manual | Automática |
| **Consistência** | Inconsistente | CLI padrão |
| **Controle** | Limitado | Total (flag opcional) |
| **UX** | Confuso | Intuitivo |
| **Risco de erro** | Alto (pode perder quantização) | Baixo (detecta automaticamente) |

---

## ✅ **Próximos Passos**

1. [ ] Implementar `_detect_quantization()` no Exporter
2. [ ] Implementar `_export_with_quantization()`
3. [ ] Testar com modelos QAT e FP32
4. [ ] Adicionar flag `quantized` ao CLI
5. [ ] Atualizar documentação
6. [ ] Remover/depreciar `test_qat_export.py` (se não for mais necessário)

---

## 📚 **Referências**

- **NVIDIA ModelOpt Export**: Usa `torch.onnx.export()` padrão, quantização é preservada automaticamente nos ops
- **TensorRT INT8**: Requer calibração com dados, suporta modelos QAT
- **ONNX Runtime**: Suporta quantized ops nativamente

---

**Status:** 📋 Proposta - Aguardando implementação  
**Prioridade:** 🔴 Alta (funcionalidade essencial para QAT completo)  
**Complexidade:** 🟡 Média (requer modificação do Exporter)

