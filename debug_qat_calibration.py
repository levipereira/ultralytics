#!/usr/bin/env python3
"""
Debug script to check if QAT model has calibration data (_amax).
"""
import torch
import sys
from pathlib import Path

def check_calibration(model_path):
    """Check if a QAT model has calibration data."""
    print(f"\n{'='*80}")
    print(f"Checking calibration in: {model_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint keys: {list(ckpt.keys())}\n")
    
    # Check if it's a full QAT checkpoint
    if 'is_quantized' in ckpt and ckpt['is_quantized']:
        print("✅ Checkpoint is marked as quantized")
        
        if 'modelopt_state' in ckpt:
            print("✅ Checkpoint has modelopt_state")
        else:
            print("❌ Checkpoint missing modelopt_state")
            
        if 'model_state_dict' in ckpt:
            print("✅ Checkpoint has model_state_dict")
            
            # Search for _amax in state_dict
            state_dict = ckpt['model_state_dict']
            amax_keys = [k for k in state_dict.keys() if '_amax' in k]
            
            print(f"\n{'='*80}")
            print(f"Found {len(amax_keys)} _amax buffers in state_dict")
            print(f"{'='*80}")
            
            if len(amax_keys) > 0:
                print("\n✅ Model HAS calibration data!")
                print("\nFirst 10 _amax entries:")
                for i, key in enumerate(amax_keys[:10]):
                    value = state_dict[key]
                    print(f"  {i+1}. {key}")
                    print(f"     Value: {value.item() if value.numel() == 1 else value}")
                
                if len(amax_keys) > 10:
                    print(f"  ... and {len(amax_keys) - 10} more")
            else:
                print("\n❌ No _amax buffers found in state_dict")
                print("This means the model was NOT calibrated during QAT training")
                
        else:
            print("❌ Checkpoint missing model_state_dict")
    else:
        print("❌ Checkpoint not marked as quantized")
        
        # Try to load with mto.restore
        print("\nTrying to load with mto.restore()...")
        try:
            import modelopt.torch.opt as mto
            from ultralytics import YOLO
            
            base_model = YOLO("yolo11n.pt").model
            model = mto.restore(base_model, str(model_path))
            
            print("✅ Successfully loaded with mto.restore()")
            
            # Check for quantizers in loaded model
            quantizer_count = 0
            amax_count = 0
            
            for name, module in model.named_modules():
                # Check for TensorQuantizer
                module_type = type(module).__name__
                if 'Quantizer' in module_type or 'TensorQuantizer' in module_type:
                    quantizer_count += 1
                    
                    # Check if has _amax
                    if hasattr(module, '_amax') and module._amax is not None:
                        amax_count += 1
                        if amax_count <= 3:
                            print(f"  {name}: _amax = {module._amax}")
            
            print(f"\nFound {quantizer_count} quantizers")
            print(f"Found {amax_count} with _amax values")
            
            if amax_count > 0:
                print("\n✅ Loaded model HAS calibration!")
            else:
                print("\n❌ Loaded model has NO calibration")
                
        except Exception as e:
            print(f"❌ Failed to load with mto.restore(): {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_qat_calibration.py <path_to_qat_model.pt>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Error: File not found: {model_path}")
        sys.exit(1)
    
    check_calibration(model_path)

