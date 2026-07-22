import os
import json
import torch
from src.models.yolo_model import YOLOClassifier
from src.models.cnn_model import CNNClassifier

def get_predictor(model_type, weights_path, class_map_path=None):
    """
    Returns an initialized predictor instance for the specified model type.
    
    Args:
        model_type (str): 'yolo' or 'resnet18'
        weights_path (str): Path to the saved weights file (.pt or .pth)
        class_map_path (str, optional): Path to the JSON containing class-to-index mapping (required for resnet18)
    """
    model_type = model_type.lower()
    if model_type == 'yolo':
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
        return YOLOClassifier(weights_path=weights_path)
        
    elif model_type == 'resnet18':
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"ResNet-18 weights not found at {weights_path}")
            
        if not class_map_path:
            # Try to guess class map path (e.g. replacing extension with _classes.json)
            base_path = os.path.splitext(weights_path)[0]
            class_map_path = f"{base_path}_classes.json"
            
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(
                f"Class mapping file not found at {class_map_path}. "
                "ResNet-18 requires class mapping for mapping indices to human-readable names."
            )
            
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)
            
        return CNNClassifier(weights_path=weights_path, class_to_idx=class_to_idx)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'yolo' or 'resnet18'")

def predict_image(image_path, model_type, weights_path, class_map_path=None):
    """
    Utility function to run prediction on an image for external usage (like FastAPI).
    """
    predictor = get_predictor(model_type, weights_path, class_map_path)
    return predictor.predict(image_path)

if __name__ == "__main__":
    # Test script execution
    import argparse
    parser = argparse.ArgumentParser(description="Unified Inference Client")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, choices=["yolo", "resnet18"], help="Model type to use")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file")
    parser.add_argument("--class-map", type=str, default=None, help="Path to class mapping JSON (for resnet18)")
    
    args = parser.parse_args()
    
    try:
        result = predict_image(args.image, args.model, args.weights, args.class_map)
        print("\nPrediction Result:")
        print(f"Predicted Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Class Index: {result['class_index']}")
    except Exception as e:
        print(f"Error: {e}")

