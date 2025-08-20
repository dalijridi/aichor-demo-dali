#!/usr/bin/env python3

import sys
import os
import time

# Use the EXACT same logging setup as your working TensorFlow script
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def simple_torch_training():
    """Minimal PyTorch training - following your TF script pattern exactly"""
    print("Starting simple PyTorch training...")

    try:
        # Install PyTorch dynamically (just like your TF script does)
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            print(f"✓ PyTorch version: {torch.__version__}")
        except ImportError:
            print("PyTorch not found, installing...")
            os.system("pip install torch")
            import torch
            import torch.nn as nn  
            import torch.optim as optim
            print(f"✓ PyTorch installed, version: {torch.__version__}")

        # Check for GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️ GPU not found. Using CPU.")

        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(1, 1)
            
            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().to(device)
        
        # Simple training data
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
        y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], device=device)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        print("Starting training...")
        for epoch in range(5):
            outputs = model(X)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}", flush=True)

        print("✓ Training completed successfully!")
        return {'status': 'success', 'final_loss': float(loss.item())}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'training_error', 'error': str(e)}

def main():
    print("=== PYTORCH VERTEX AI TRAINING SCRIPT ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Parse simple arguments manually (like your TF script)
    sleep_time = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--sleep" and i + 1 < len(sys.argv):
            try:
                sleep_time = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid sleep value: {sys.argv[i + 1]}")
    
    print(f"Sleep time: {sleep_time}")
    
    # Run training
    result = simple_torch_training()
    print(f"\nTraining result: {result}")
    
    # Optional sleep
    if sleep_time > 0:
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    
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
