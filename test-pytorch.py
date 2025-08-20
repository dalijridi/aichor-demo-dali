#!/usr/bin/env python3

import sys
import os
import time

# Use EXACT same setup as your working TensorFlow script
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def test_pytorch():
    """Just test if we can install and import PyTorch"""
    print("Testing PyTorch installation...")

    try:
        # Try to import PyTorch
        try:
            import torch
            print(f"✓ PyTorch already available, version: {torch.__version__}")
        except ImportError:
            print("PyTorch not found, installing CPU version...")
            # Install CPU version first (safer)
            os.system("pip install torch --index-url https://download.pytorch.org/whl/cpu")
            import torch
            print(f"✓ PyTorch CPU installed, version: {torch.__version__}")

        # Simple test
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"✓ Simple tensor created: {x}")
        
        # Check CUDA (but don't fail if not available)
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available, using CPU (this is fine for testing)")

        print("✓ PyTorch test completed successfully!")
        return {'status': 'success'}

    except Exception as e:
        import traceback
        print(f"✗ PyTorch test failed: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

def main():
    print("=== PYTORCH INSTALLATION TEST ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Parse sleep argument (same as your working script)
    sleep_time = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--sleep" and i + 1 < len(sys.argv):
            try:
                sleep_time = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid sleep value: {sys.argv[i + 1]}")
    
    # Run test
    result = test_pytorch()
    print(f"\nTest result: {result}")
    
    # Sleep
    if sleep_time > 0:
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    
    print("Test completed!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
