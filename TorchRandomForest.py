import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Step 5: Split Dataset (80% Train, 10% Validation, 10% Test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Use multiple workers & pin memory for speedup
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Step 6: Extract features from dataset
    train_features = []
    train_labels = []
    for images, labels in train_loader:
        images = images.to(device)
        images = images.view(images.size(0), -1)
        train_features.extend(images.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    test_features = []
    test_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        images = images.view(images.size(0), -1)
        test_features.extend(images.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

    # Step 7: Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Step 8: Train Random Forest model on GPU
    train_features = cp.asarray(train_features)
    train_labels = cp.asarray(train_labels)
    test_features = cp.asarray(test_features)
    test_labels = cp.asarray(test_labels)

    cu_rf = cuRF(n_estimators=100, max_depth=None, random_state=42)
    cu_rf.fit(train_features, train_labels)

    # Step 9: Evaluate Random Forest model
    predictions = cu_rf.predict(test_features)
    accuracy = accuracy_score(cp.asnumpy(test_labels), cp.asnumpy(predictions))
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()