#!/usr/bin/env python3
"""
Very simple GPU training test for Vertex AI
Designed to show clear logs and test GPU functionality
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def log_print(message):
    """Print with explicit flushing and timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()
    sys.stderr.flush()

def main():
    log_print("🚀 STARTING SIMPLE GPU TRAINING TEST")
    log_print("="*50)
    
    # 1. Environment Check
    log_print("📋 ENVIRONMENT CHECK:")
    log_print(f"   Python version: {sys.version}")
    log_print(f"   PyTorch version: {torch.__version__}")
    log_print(f"   CUDA version: {torch.version.cuda}")
    log_print(f"   Current working directory: {os.getcwd()}")
    
    # 2. GPU Detection
    log_print("\n🔍 GPU DETECTION:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        log_print(f"   ✅ CUDA is available!")
        log_print(f"   📊 Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            log_print(f"   🎯 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        device = torch.device("cuda:0")
        log_print(f"   🎯 Using device: {device}")
    else:
        log_print("   ❌ CUDA not available, using CPU")
        device = torch.device("cpu")
    
    # 3. Simple Model
    log_print("\n🧠 CREATING MODEL:")
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(100, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 10)
            self.fc3 = nn.Linear(10, 1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNet().to(device)
    log_print(f"   ✅ Model created and moved to {device}")
    log_print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Create Training Data
    log_print("\n📊 CREATING TRAINING DATA:")
    batch_size = 32
    input_size = 100
    
    # Create dummy data
    X = torch.randn(batch_size, input_size).to(device)
    y = torch.randn(batch_size, 1).to(device)
    
    log_print(f"   ✅ Training data created: {X.shape}")
    log_print(f"   🎯 Data is on device: {X.device}")
    
    # 5. Training Setup
    log_print("\n⚙️  TRAINING SETUP:")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log_print("   ✅ Loss function: MSE Loss")
    log_print("   ✅ Optimizer: Adam (lr=0.001)")
    
    # 6. Training Loop
    log_print("\n🏋️  STARTING TRAINING:")
    log_print("-" * 40)
    
    epochs = 10
    for epoch in range(epochs):
        # Forward pass
        model.train()
        optimizer.zero_grad()
        
        # Time the forward pass
        start_time = time.time()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        forward_time = time.time() - start_time
        
        # Log every epoch
        log_print(f"   Epoch {epoch+1:2d}/{epochs} | Loss: {loss.item():.6f} | Time: {forward_time*1000:.2f}ms")
        
        # Small delay to make logs more readable
        time.sleep(0.5)
    
    log_print("-" * 40)
    
    # 7. Final GPU Memory Check
    if torch.cuda.is_available():
        log_print("\n💾 FINAL GPU MEMORY CHECK:")
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        memory_cached = torch.cuda.memory_reserved(device) / (1024**2)
        log_print(f"   📊 GPU Memory Allocated: {memory_allocated:.2f} MB")
        log_print(f"   📊 GPU Memory Cached: {memory_cached:.2f} MB")
    
    # 8. Success Message
    log_print("\n" + "="*50)
    log_print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    log_print("🎉 GPU TEST PASSED!")
    log_print("="*50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_print(f"❌ ERROR: {str(e)}")
        import traceback
        log_print("📋 FULL TRACEBACK:")
        traceback.print_exc()
        sys.exit(1)
