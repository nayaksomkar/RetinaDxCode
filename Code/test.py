import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Progress bar

# Step 1: Check for GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device == 'cuda':
    print(f"Using device: {device}")
else:
    input("No GPU detected, using CPU. Continue? [Y/n]: ")
    if input.lower() == 'y':
        exit()