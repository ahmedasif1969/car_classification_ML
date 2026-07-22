import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from PIL import Image

class CNNClassifier:
    def __init__(self, num_classes=None, class_to_idx=None, weights_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None
        
        # Load or initialize model
        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)
        elif num_classes:
            self._init_model(num_classes)
        else:
            self.model = None

    def _init_model(self, num_classes):
        # Initialize ResNet-18 with pre-trained weights
        print(f"Initializing ResNet-18 backbone with {num_classes} classes...")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.model.fc.in_features
        # Replace the final fully connected layer for our number of classes
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)

    def _get_transforms(self):
        # Standard ImageNet transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform

    def train_model(self, train_dir, val_dir, epochs=10, batch_size=32, lr=0.001, save_dir="weights"):
        train_transform, val_transform = self._get_transforms()
        
        # Datasets & Loaders
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        num_classes = len(train_dataset.classes)
        
        self._init_model(num_classes)
        
        # Use simple but robust dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0.0
        best_weights_path = os.path.join(save_dir, "resnet18_best.pth")
        
        print(f"Starting ResNet-18 training on {self.device} for {epochs} epochs...")
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        
        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)
                
            epoch_train_loss = running_loss / len(train_dataset)
            epoch_train_acc = correct / total
            
            # Validation Phase
            val_loss, val_metrics = self.evaluate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f}")
            
            # Save best weights
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save(best_weights_path)
                print(f"--> Saved new best ResNet weights with Acc: {best_val_acc:.4f}")
                
        # Load best weights back
        self.load(best_weights_path)
        return history

    def evaluate(self, val_loader_or_dir, criterion=None):
        self.model.eval()
        
        if isinstance(val_loader_or_dir, (str, os.PathLike)):
            _, val_transform = self._get_transforms()
            val_dataset = datasets.ImageFolder(val_loader_or_dir, transform=val_transform)
            loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        else:
            loader = val_loader_or_dir
            
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        val_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        return val_loss, metrics

    def predict(self, image_path):
        self.model.eval()
        _, val_transform = self._get_transforms()
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, class_idx = torch.max(probabilities, 0)
            
        class_idx = class_idx.item()
        confidence = confidence.item()
        class_name = self.idx_to_class[class_idx] if self.idx_to_class else str(class_idx)
        
        # Return structured response suitable for API
        return {
            'class_index': class_idx,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': {self.idx_to_class[i] if self.idx_to_class else str(i): float(prob) 
                              for i, prob in enumerate(probabilities.cpu().numpy())}
        }

    def save(self, filepath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx
        }
        torch.save(checkpoint, filepath)
        # Save a separate class map JSON for easy future API usage
        class_map_path = os.path.splitext(filepath)[0] + "_classes.json"
        with open(class_map_path, 'w') as f:
            json.dump(self.class_to_idx, f, indent=4)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        num_classes = len(self.class_to_idx)
        
        self._init_model(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Loaded ResNet-18 weights successfully from {filepath}")
