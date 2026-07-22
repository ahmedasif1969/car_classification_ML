import os
import argparse
import pandas as pd
from src.models.cnn_model import CNNClassifier
from src.models.yolo_model import YOLOClassifier

def print_metrics_table(results):
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    print("\n" + "=" * 65)
    print("                       MODEL EVALUATION RESULTS")
    print("=" * 65)
    print(f"| {headers[0]:<12} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10} |")
    print("-" * 65)
    for model_name, metrics in results.items():
        print(f"| {model_name.upper():<12} | {metrics['accuracy']:<10.4f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1']:<10.4f} |")
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Classify and ResNet-18 Classifiers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for ResNet-18")
    parser.add_argument("--model", type=str, choices=["all", "yolo", "resnet18"], default="all",
                        help="Which model(s) to train")
    parser.add_argument("--train-dir", type=str, default="data/split/train", help="Path to training data directory")
    parser.add_argument("--val-dir", type=str, default="data/split/val", help="Path to validation data directory")
    parser.add_argument("--save-dir", type=str, default="weights", help="Directory to save weights")
    
    args = parser.parse_args()
    
    print("\n--- Training Configuration ---")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate (ResNet): {args.lr}")
    print(f"Model selection: {args.model}")
    print(f"Train directory: {args.train_dir}")
    print(f"Val directory: {args.val_dir}")
    print(f"Save directory: {args.save_dir}")
    print("-" * 30 + "\n")
    
    results_metrics = {}
    
    # Train ResNet-18 (CNN Classifier)
    if args.model in ["all", "resnet18"]:
        print("=== Training ResNet-18 CNN Classifier ===")
        resnet_clf = CNNClassifier()
        history = resnet_clf.train_model(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=args.save_dir
        )
        
        # Evaluate ResNet-18 on validation set
        _, resnet_metrics = resnet_clf.evaluate(args.val_dir)
        results_metrics['resnet18'] = resnet_metrics
        print("ResNet-18 training and evaluation complete!\n")
        
    # Train YOLOv8-Classify
    if args.model in ["all", "yolo"]:
        print("=== Training YOLOv8-Classify ===")
        # For YOLO, data is the split directory containing train and val folders
        yolo_data_dir = os.path.dirname(args.train_dir) # data/split
        
        yolo_clf = YOLOClassifier()
        yolo_clf.train_model(
            data_dir=yolo_data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=224,
            save_dir=args.save_dir
        )
        
        # Evaluate YOLO on validation set
        yolo_metrics = yolo_clf.evaluate(args.val_dir)
        results_metrics['yolo'] = yolo_metrics
        print("YOLOv8-Classify training and evaluation complete!\n")
        
    # Print comparison table
    if results_metrics:
        print_metrics_table(results_metrics)
        
        # Save metrics comparison to a JSON file
        metrics_file = os.path.join(args.save_dir, "evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            import json
            json.dump(results_metrics, f, indent=4)
        print(f"Saved evaluation metrics comparison table to {metrics_file}")
        
    print("\nTraining workflow completed. Weights are saved in the 'weights/' directory:")
    if args.model in ["all", "resnet18"]:
        print(f"- ResNet-18: {args.save_dir}/resnet18_best.pth")
    if args.model in ["all", "yolo"]:
        print(f"- YOLOv8-Classify: {args.save_dir}/yolov8_best.pt")

if __name__ == "__main__":
    main()
