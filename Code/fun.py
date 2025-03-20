import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Progress bar

def loadDatasets(path):
    #Image Transformations
    transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1]
    ])  

    print("\nLoading dataset...")
    dataset = datasets.ImageFolder(root=path, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Classes Found: {dataset.classes}")

    return {"dataset" : dataset, "num_classes" : num_classes}