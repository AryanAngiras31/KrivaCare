import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

import config
from preprocessing import run_tabular_preprocessing, run_image_metadata_preprocessing
from datasets import TabularExpertDataset, ImageExpertDataset
from models import TabularExpert, ImageExpert

def compute_class_weights(dataloader, num_classes=2):
    """
    Dynamically computes inverse frequency weights based on the training dataset.
    This prevents the model from ignoring the minority class (Malignant).
    """
    print("Computing class weights from training distribution...")
    class_counts = [0] * num_classes
    
    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1
            
    total_samples = sum(class_counts)
    
    # Formula: Total Samples / (Number of Classes * Samples in Class)
    weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in class_counts]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(config.DEVICE)
    
    print(f"Class Counts: Malignant(0)={class_counts[0]}, Benign(1)={class_counts[1]}")
    print(f"Assigned Weights: Malignant(0)={weights[0]:.4f}, Benign(1)={weights[1]:.4f}")
    
    return weight_tensor

def evaluate_tabular(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    val_loss = running_loss / len(all_labels)
    # Calculate Macro F1-Score
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, val_f1

def train_tabular_expert():
    print(f"\n==========================================")
    print(f"TRAINING TABULAR EXPERT ({config.DEVICE})")
    print(f"==========================================")
    
    train_csv = os.path.join(config.TABULAR_DIR, 'cleaned_train.csv')
    test_csv = os.path.join(config.TABULAR_DIR, 'cleaned_test.csv')
    
    train_dataset = TabularExpertDataset(train_csv)
    test_dataset = TabularExpertDataset(test_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weights(train_loader, num_classes=config.NUM_CLASSES)
    
    input_dim = train_dataset.features.shape[1]
    model = TabularExpert(input_dim=input_dim, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Inject weights into the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    best_f1 = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        for features, labels in train_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Inject 5% random Gaussian noise only during training
            if model.training:
                noise = torch.randn_like(features) * 0.05
                features = features + noise
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
        train_loss = running_loss / len(all_labels)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        val_loss, val_f1 = evaluate_tabular(model, test_loader, criterion)
        
        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")
        
        # Save model based on best F1-Score instead of accuracy
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(config.MODELS_DIR, 'tabular_expert.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"Tabular Expert Training Complete. Best Val Macro F1: {best_f1:.4f}")

def evaluate_image(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    val_loss = running_loss / len(all_labels)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, val_f1

def train_image_expert():
    print(f"\n==========================================")
    print(f"TRAINING IMAGE EXPERT ({config.DEVICE})")
    print(f"==========================================")
    
    train_csv = os.path.join(config.IMAGE_DIR, 'mmotu_train.csv')
    val_csv = os.path.join(config.IMAGE_DIR, 'mmotu_val.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        print("MMOTU CSV files not found. Skipping Image Expert training.")
        return

    train_dataset = ImageExpertDataset(train_csv)
    val_dataset = ImageExpertDataset(val_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weights(train_loader, num_classes=config.NUM_CLASSES)
    
    model = ImageExpert(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    # Inject weights into the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    best_f1 = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        for images, labels in tqdm(train_loader, desc=f"Image Epoch {epoch}/{config.EPOCHS}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
        train_loss = running_loss / len(all_labels)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        val_loss, val_f1 = evaluate_image(model, val_loader, criterion)
        
        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")
        
        # Save model based on best F1-Score instead of accuracy
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(config.MODELS_DIR, 'image_expert.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"Image Expert Training Complete. Best Val Macro F1: {best_f1:.4f}")

def main():
    print("Triggering Data Preprocessing Pipeline...")
    run_tabular_preprocessing()
    run_image_metadata_preprocessing()
        
    train_tabular_expert()
    train_image_expert()

if __name__ == "__main__":
    main()