#!/usr/bin/env python3

import sys
import os
import time
import argparse
from datetime import datetime
import logging

# Force Python logging to stdout (same as your working TF script)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def simple_torch_training(epochs=5, lr=0.01):
    """Minimal PyTorch training with proper error handling"""
    print("Starting simple PyTorch training...")
    
    try:
        # Ensure PyTorch (similar to your TF script)
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            print(f"✓ PyTorch version: {torch.__version__}")
        except ImportError:
            print("PyTorch not found, installing...")
            os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            import torch
            import torch.nn as nn
            import torch.optim as optim
            print(f"✓ PyTorch installed, version: {torch.__version__}")

        # Check for GPU availability
        print("Checking for GPU...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ GPU is available. Using device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device("cpu")
            print("⚠️ GPU not found. Using CPU.")

        # Simple Model Definition
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

        # Instantiate Model and move to device
        model = SimpleModel().to(device)
        print("\nModel architecture:")
        print(model)
        print(f"Model is on device: {next(model.parameters()).device}")

        # Create Dummy Data and move to device
        X_train = torch.randn(100, 10).to(device)
        y_train = torch.randn(100, 1).to(device)
        print(f"Training data is on device: {X_train.device}")

        # Define Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        print("Loss function and optimizer initialized successfully.")

        # Training Loop with live logging (similar to your TF callback)
        print(f"\n--- Starting Training Loop for {epochs} epochs ---")
        for epoch in range(epochs):
            model.train()
            
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Live logging (like your TF script)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}", flush=True)
            time.sleep(0.5)  # Small delay for readable logs

        print("\n✓ Training completed successfully!")
        return {
            'status': 'success',
            'final_loss': float(loss.item()),
            'device_used': str(device),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }

    except Exception as e:
        import traceback
        print(f"✗ Training error: {e}")
        traceback.print_exc()
        return {'status': 'training_error', 'error': str(e)}

def main():
    print("=== PYTORCH VERTEX AI TRAINING SCRIPT ===")
    print(f"Python version: {sys.version}")
    print(f"Arguments: {sys.argv}")
    print(f"Working directory: {os.getcwd()}")
    
    # Parse arguments (similar to your working script)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--sleep', type=int, default=0, help='Sleep time after training.')
    
    try:
        args = parser.parse_args()
        print(f"Training for {args.epochs} epochs with learning rate {args.lr}")
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    
    # Run training
    result = simple_torch_training(epochs=args.epochs, lr=args.lr)
    print(f"\nTraining result: {result}")
    
    # Optional sleep (like your working script)
    if args.sleep > 0:
        print(f"Sleeping for {args.sleep} seconds...")
        time.sleep(args.sleep)
    
    print("Script completed successfully!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
