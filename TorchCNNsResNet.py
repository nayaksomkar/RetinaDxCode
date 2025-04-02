import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision import models
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, exiting program.")
        return

    # Step 2: Define Dataset Path
    DATASET_PATH = r"C:\someFiles\githubRepo\RetinaDx\RetinaDxDataSet\Dataset X"

    # Step 3: Image Transformations (Reduce size to 224 for faster training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Smaller size for faster training
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
    ])

    # Step 4: Load Dataset
    print("\nLoading dataset...")
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Classes Found: {dataset.classes}")

    # Step 5: Split Dataset (80% Train, 10% Validation, 10% Test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Use multiple workers & pin memory for speedup
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Step 6: Use Pretrained ResNet for Faster Training
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace final layer
    model = model.to(device)
    
    print("\nUsing Pretrained Model:\n", model)

    # Step 8: Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Training

    # Step 9: Train the Model (Reduce Epochs & Add Early Stopping)
    num_epochs = 5
    best_val_loss = float('inf')
    patience = 2  # Early stopping if no improvement for 2 epochs
    patience_counter = 0

    print("\nStarting Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Mixed precision for speed
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=train_loss / len(train_loader))

        # Validation step
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "RetinaDx.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered. Stopping training.")
                break

    print("\nTraining Completed!")

    # Step 10: Evaluate the Model
    print("\nEvaluating Model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Progress"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"\nFinal Model Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), os.path.join("models", "TorchCNNsResNet.pth"))  # Save final model
    
if __name__ == "__main__":
    train()