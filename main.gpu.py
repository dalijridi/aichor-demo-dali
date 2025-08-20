import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import sys

# Force stdout flushing
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

print_flush("--- Starting Training Script ---")

# --- 1. Check for GPU availability ---
print_flush("Checking for GPU...")
print_flush(f"PyTorch version: {torch.__version__}")
print_flush(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print_flush(f"✅ GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    print_flush(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    print_flush("⚠️ GPU not found. Using CPU.")

# --- 2. Simple Model Definition ---
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# --- 3. Argument Parsing ---
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    args = parser.parse_args()
    print_flush(f"Training for {args.epochs} epochs with a learning rate of {args.lr}.")
except Exception as e:
    print_flush(f"Error parsing arguments: {e}")
    sys.exit(1)

# --- 4. Instantiate Model and move to GPU ---
try:
    model = SimpleModel().to(device)
    print_flush("\nModel architecture:")
    print_flush(model)
    
    # Check if model parameters are on the correct device
    model_device = next(model.parameters()).device
    print_flush(f"Model is on device: {model_device}")
except Exception as e:
    print_flush(f"Error creating or moving model: {e}")
    sys.exit(1)

# --- 5. Create Dummy Data and move to GPU ---
try:
    X_train = torch.randn(100, 10).to(device)
    y_train = torch.randn(100, 1).to(device)
    print_flush(f"Training data is on device: {X_train.device}")
    print_flush(f"Target data is on device: {y_train.device}")
except Exception as e:
    print_flush(f"Error creating training data: {e}")
    sys.exit(1)

# --- 6. Define Loss and Optimizer ---
try:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    print_flush("Loss function and optimizer initialized successfully.")
except Exception as e:
    print_flush(f"Error initializing loss/optimizer: {e}")
    sys.exit(1)

# --- 7. Simple Training Loop ---
print_flush("\n--- Starting Training Loop ---")
try:
    for epoch in range(args.epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print_flush(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")
        time.sleep(1)
        
except Exception as e:
    print_flush(f"Error during training: {e}")
    sys.exit(1)

print_flush("\n--- Training Complete --- ✅")
print_flush("Script finished successfully!")
