import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import json
import time
from collections import Counter

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Define all classes including special functions
        self.all_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'BACKSPACE', 'SPACE', 'ENTER'
        ]
        
        # Load images and labels
        for idx, class_name in enumerate(self.all_classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️ Warning: Directory {class_dir} not found. Creating empty directory.")
                os.makedirs(class_dir, exist_ok=True)
                continue
                
            image_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    # Verify image can be loaded
                    try:
                        img = Image.open(img_path)
                        img.verify()  # Verify it's a valid image
                        self.images.append(img_path)
                        self.labels.append(idx)
                        image_count += 1
                    except Exception as e:
                        print(f"⚠️ Skipping corrupted image {img_path}: {e}")
            
            print(f"✅ Loaded {image_count} images for class: {class_name}")
        
        # Data validation
        if len(self.images) == 0:
            raise ValueError("❌ No valid training images found! Please run capture.py first.")
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total images: {len(self.images)}")
        print(f"   Total classes: {len(self.all_classes)}")
        
        # Show class distribution
        label_counts = Counter(self.labels)
        for idx, count in label_counts.items():
            class_name = self.idx_to_class[idx]
            print(f"   {class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a dummy image if there's an error
            dummy_image = Image.new('RGB', (128, 128), color='white')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

class SignLanguageTrainer:
    def __init__(self, data_dir="data/train"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Enhanced Transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_model(self, num_classes):
        """Create ResNet18 model with updated PyTorch syntax"""
        try:
            # Updated way to load pretrained models (PyTorch 1.13+)
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("✅ Successfully loaded pretrained ResNet18 weights")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("🔄 Using randomly initialized model instead")
            model = models.resnet18(weights=None)
        
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ Model created with {num_classes} output classes")
        return model.to(self.device)
    
    def train(self, num_epochs=40, batch_size=32, learning_rate=0.001):
        """Training loop with error handling for permission issues"""
        print("📥 Loading dataset...")
        try:
            full_dataset = SignLanguageDataset(self.data_dir, transform=self.transform)
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            return None, None
        
        # Check if we have data
        if len(full_dataset) == 0:
            raise ValueError("❌ No training data found! Please run capture.py first to collect images.")
        
        print(f"🎯 Training for {len(full_dataset.all_classes)} classes: {full_dataset.all_classes}")
        print(f"📊 Total images: {len(full_dataset)}")
        
        # Split into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        if train_size == 0 or val_size == 0:
            raise ValueError("❌ Not enough data for training. Please collect more images.")
            
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Apply different transform to validation set
        val_dataset.dataset.transform = self.val_transform
        
        # Create data loaders (reduced workers for Windows compatibility)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create model
        num_classes = len(full_dataset.all_classes)
        try:
            model = self.create_model(num_classes)
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return None, None
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        print("🚀 Starting training...")
        best_val_acc = 0.0
        
        try:
            for epoch in range(num_epochs):
                start_time = time.time()
                
                # Training phase
                model.train()
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in train_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(train_dataset)
                epoch_acc = running_corrects.double() / len(train_dataset)
                
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_corrects = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_corrects += torch.sum(preds == labels.data)
                
                val_epoch_loss = val_loss / len(val_dataset)
                val_epoch_acc = val_corrects.double() / len(val_dataset)
                
                history['val_loss'].append(val_epoch_loss)
                history['val_acc'].append(val_epoch_acc.cpu().numpy())
                
                # Save best model
                if val_epoch_acc > best_val_acc:
                    best_val_acc = val_epoch_acc
                    try:
                        torch.save(model.state_dict(), 'models/sign_language_model_best.pth')
                        print(f"💾 New best model saved with validation accuracy: {best_val_acc:.4f}")
                    except Exception as e:
                        print(f"⚠️ Could not save best model: {e}")
                
                scheduler.step()
                
                epoch_time = time.time() - start_time
                print(f'Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.1f}s')
                print(f'   Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
                print(f'   Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')
                print(f'   Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
                print('-' * 70)
            
            # Save the final model
            try:
                torch.save(model.state_dict(), 'models/sign_language_model.pth')
                
                # Save class mappings
                with open('models/class_mapping.json', 'w') as f:
                    json.dump({
                        'class_to_idx': full_dataset.class_to_idx,
                        'idx_to_class': full_dataset.idx_to_class,
                        'total_images': len(full_dataset),
                        'best_val_accuracy': float(best_val_acc)
                    }, f, indent=4)
                
                print("✅ Training completed!")
                print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
                print(f"💾 Model saved with {num_classes} classes")
                
                # Plot training history
                self.plot_training_history(history)
                
            except Exception as e:
                print(f"⚠️ Could not save final model: {e}")
            
            return model, history
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None, None
    
    def plot_training_history(self, history):
        """Plot training history"""
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
            plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
            plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
            print("📊 Training history plot saved as 'models/training_history.png'")
        except Exception as e:
            print(f"⚠️ Could not save training plot: {e}")

def setup_torch_cache():
    """Setup PyTorch cache directory to avoid permission issues"""
    try:
        # Try to use a local cache directory
        cache_dir = os.path.join(os.getcwd(), '.torch_cache')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TORCH_HOME'] = cache_dir
        print(f"📁 Using local cache directory: {cache_dir}")
    except Exception as e:
        print(f"⚠️ Could not setup local cache: {e}")

if __name__ == "__main__":
    try:
        print("🤖 Sign Language Model Training Starting...")
        print("=" * 70)
        
        # Setup cache to avoid permission issues
        setup_torch_cache()
        
        trainer = SignLanguageTrainer()
        model, history = trainer.train(
            num_epochs=40,
            batch_size=32,
            learning_rate=0.001
        )
        
        if model is not None:
            print("🎉 Training completed successfully!")
        else:
            print("❌ Training failed. Please check the error messages above.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()