# Use an official Python image with GPU support (if needed)
FROM python:3.11.9

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision tensorflow numpy pandas scikit-learn tqdm

# Run the training script
CMD ["python", "mainTorch.py"]