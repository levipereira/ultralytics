import argparse
from ultralytics import YOLO

def export_model(weights):
    # Initialize the model with the provided weights file
    model = YOLO(weights)
    
    # Export the model to ONNX format with TensorRT optimization
    model.export(format="onnx_trt", dynamic=True, topk_all=100, iou_thres= 0.45, conf_thres= 0.25  )

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX with TensorRT optimization.")
    
    # Add the -w/--weights argument to specify the weights file
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the YOLO weights file (e.g., yolov8n.pt)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the export_model function with the provided weights file
    export_model(args.weights)

if __name__ == "__main__":
    main()


