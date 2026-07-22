import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class YOLOClassifier:
    def __init__(self, weights_path=None, device=None):
        self.device = device if device else ("cuda" if torch_has_cuda() else "cpu")
        
        # Load custom model if provided, else load pre-trained yolov8n-cls
        if weights_path and os.path.exists(weights_path):
            print(f"Loading custom YOLO weights from {weights_path}...")
            self.model = YOLO(weights_path)
        else:
            print("Loading pre-trained YOLOv8n-cls model...")
            self.model = YOLO("yolov8n-cls.pt")
            
        self.class_to_idx = self.model.names  # names dict: {idx: class_name}
        # Invert it to match CNN structure
        self.idx_to_class = self.model.names

    def train_model(self, data_dir, epochs=10, batch_size=32, imgsz=224, save_dir="weights"):
        print(f"Starting YOLOv8-classify training for {epochs} epochs...")
        
        # Run training
        # Note: Ultralytics saves the results to 'runs/classify/train' by default
        results = self.model.train(
            data=str(data_dir),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=self.device,
            project="runs/classify",
            name="train_run",
            exist_ok=True
        )
        
        # Locate the best weights
        best_run_weights = Path(results.save_dir) / "weights/best.pt" if hasattr(results, 'save_dir') else Path("runs/classify/train_run/weights/best.pt")
        if not best_run_weights.exists():
            # Try alternate fallback paths
            alternatives = [
                Path("runs/classify/runs/classify/train_run/weights/best.pt"),
                Path("runs/classify/train/weights/best.pt"),
                Path("runs/classify/train_run/weights/best.pt")
            ]
            for alt in alternatives:
                if alt.exists():
                    best_run_weights = alt
                    break
                    
        os.makedirs(save_dir, exist_ok=True)
        dest_weights_path = os.path.join(save_dir, "yolov8_best.pt")
        
        if best_run_weights.exists():
            shutil.copy(best_run_weights, dest_weights_path)
            print(f"--> Saved best YOLO weights to {dest_weights_path}")
            # Re-load model with best weights
            self.model = YOLO(dest_weights_path)
            self.class_to_idx = {v: k for k, v in self.model.names.items()}
            self.idx_to_class = self.model.names
        else:
            print("Warning: Could not locate the best weights file after YOLO training.")
            
        return results

    def evaluate(self, val_dir):
        print("Evaluating YOLO model on validation set...")
        val_path = Path(val_dir)
        
        # We walk through the validation folders, get predictions, and calculate metrics
        all_preds = []
        all_labels = []
        
        classes = sorted([d.name for d in val_path.iterdir() if d.is_dir()])
        class_to_idx_local = {name: i for i, name in enumerate(classes)}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # Gather all validation images
        image_paths = []
        true_labels = []
        
        for cls_name in classes:
            cls_dir = val_path / cls_name
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    image_paths.append(str(img_file))
                    true_labels.append(cls_name)
                    
        if not image_paths:
            print("No validation images found for evaluation.")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            
        # Run batch predictions to be faster
        batch_size = 64
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_true = true_labels[i:i+batch_size]
            
            # Predict using YOLO model
            # verbose=False reduces console clutter
            results = self.model(batch_paths, verbose=False)
            
            for res, true_cls in zip(results, batch_true):
                # Get the class with the highest probability
                pred_idx = res.probs.top1
                pred_name = res.names[pred_idx]
                
                all_preds.append(pred_name)
                all_labels.append(true_cls)
                
        # Calculate standard classification metrics
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def predict(self, image_path):
        results = self.model(image_path, verbose=False)[0]
        pred_idx = results.probs.top1
        pred_name = results.names[pred_idx]
        confidence = float(results.probs.top1conf)
        
        probabilities = results.probs.data.cpu().numpy()
        probs_dict = {results.names[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        return {
            'class_index': int(pred_idx),
            'class_name': pred_name,
            'confidence': confidence,
            'probabilities': probs_dict
        }

def torch_has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
