#!/usr/bin/env python3
"""
Automatic ONNX quantization using NVIDIA ModelOpt
Based on: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/onnx_ptq/torch_quant_to_onnx.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

def quantize_onnx(onnx_path, output_path=None, quantize_mode="int8"):
    """
    Quantize ONNX model using NVIDIA ModelOpt.
    
    Args:
        onnx_path (str): Path to input ONNX model
        output_path (str): Path for output quantized model (optional)
        quantize_mode (str): Quantization mode (int8, fp16, etc.)
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_quantized{onnx_path.suffix}"
    else:
        output_path = Path(output_path)
    
    print(f"🔄 Quantizing ONNX model: {onnx_path}")
    print(f"📤 Output path: {output_path}")
    print(f"🎯 Quantization mode: {quantize_mode}")
    
    # Build command
    cmd = [
        "python3", "-m", "modelopt.onnx.quantization",
        "--onnx_path", str(onnx_path),
        "--quantize_mode", quantize_mode,
        "--output_path", str(output_path)
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ ONNX quantization completed successfully!")
        print(f"📁 Quantized model saved to: {output_path}")
        
        # Show file sizes
        original_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
        quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        print(f"\n📊 File size comparison:")
        print(f"   Original:  {original_size:.2f} MB")
        print(f"   Quantized: {quantized_size:.2f} MB")
        print(f"   Compression: {compression_ratio:.1f}x smaller")
        
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ONNX quantization failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("❌ modelopt.onnx.quantization not found!")
        print("   Make sure nvidia-modelopt is installed:")
        print("   pip install nvidia-modelopt[hf]")
        raise

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model using NVIDIA ModelOpt")
    parser.add_argument("onnx_path", help="Path to input ONNX model")
    parser.add_argument("-o", "--output", help="Output path for quantized model")
    parser.add_argument("-m", "--mode", default="int8", 
                       choices=["int8", "fp16"], 
                       help="Quantization mode (default: int8)")
    
    args = parser.parse_args()
    
    try:
        output_path = quantize_onnx(args.onnx_path, args.output, args.mode)
        print(f"\n🎉 Success! Quantized model ready: {output_path}")
        
    except Exception as e:
        print(f"\n💥 Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
