import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time

print("--- Starting Training Script ---")

# --- 1. Check for GPU availability ---
print("Checking for GPU...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ GPU not found. Using CPU.")

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

# --- 3. Argument Parsing (Good practice for Vertex AI jobs) ---
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
args = parser.parse_args()

print(f"Training for {args.epochs} epochs with a learning rate of {args.lr}.")

# --- 4. Instantiate Model and move to GPU ---
model = SimpleModel().to(device)
print("\nModel architecture:")
print(model)
print(f"\nModel is on device: {''.join(str(p.device) for p in model.parameters()).split('(')[0]}")

# --- 5. Create Dummy Data and move to GPU ---
X_train = torch.randn(100, 10).to(device)
y_train = torch.randn(100, 1).to(device)
print(f"Training data is on device: {X_train.device}")

# --- 6. Define Loss and Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# --- 7. Simple Training Loop ---
print("\n--- Starting Training Loop ---")
for epoch in range(args.epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")
    time.sleep(2) # Adding a small delay to make logs easier to follow

print("\n--- Training Complete --- ✅")
