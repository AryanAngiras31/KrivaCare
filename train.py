import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from preprocessing import run_tabular_preprocessing, run_image_metadata_preprocessing
from datasets import TabularExpertDataset, ImageExpertDataset
from models import TabularExpert, ImageExpert

def evaluate_tabular(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

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
    
    # Infer input dimension from training dataset features
    input_dim = train_dataset.features.shape[1]
    model = TabularExpert(input_dim=input_dim, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        val_loss, val_acc = evaluate_tabular(model, test_loader, criterion)
        
        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.MODELS_DIR, 'tabular_expert.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"Tabular Expert Training Complete. Best Val Accuracy: {best_acc:.2f}%")


def evaluate_image(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


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
    
    model = ImageExpert(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    best_acc = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        val_loss, val_acc = evaluate_image(model, val_loader, criterion)
        
        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.MODELS_DIR, 'image_expert.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"Image Expert Training Complete. Best Val Accuracy: {best_acc:.2f}%")


def main():
    print("Triggering Data Preprocessing Pipeline...")
    run_tabular_preprocessing()
    run_image_metadata_preprocessing()
        
    train_tabular_expert()
    train_image_expert()

if __name__ == "__main__":
    main()