#!/usr/bin/env python3

import sys
import os
import time
import subprocess

# Use EXACT same setup as your working TensorFlow script
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def run_command(cmd):
    """Run a command and capture output"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return False
    except Exception as e:
        print(f"Command failed: {e}")
        return False

def debug_environment():
    """Debug the environment step by step"""
    print("=== DEBUGGING PYTORCH INSTALLATION ===")
    
    # Test 1: Basic environment
    print("\n--- Test 1: Basic Environment ---")
    run_command("python3 --version")
    run_command("pip --version")
    run_command("which python3")
    run_command("which pip")
    
    # Test 2: Check available space and memory
    print("\n--- Test 2: System Resources ---")
    run_command("df -h")
    run_command("free -h")
    
    # Test 3: Test basic pip functionality
    print("\n--- Test 3: Basic pip install ---")
    if run_command("pip install requests"):
        print("✓ Basic pip install works")
        run_command("python3 -c \"import requests; print('requests imported successfully')\"")
    else:
        print("✗ Basic pip install failed")
    
    # Test 4: Try different PyTorch installation methods
    print("\n--- Test 4: PyTorch Installation Attempts ---")
    
    # Method 1: CPU only, minimal
    print("Trying CPU-only PyTorch...")
    if run_command("pip install torch --no-cache-dir"):
        print("✓ PyTorch CPU installation succeeded")
        if run_command("python3 -c \"import torch; print(f'PyTorch version: {torch.__version__}')\""):
            print("✓ PyTorch import succeeded")
            return True
        else:
            print("✗ PyTorch import failed")
    else:
        print("✗ PyTorch CPU installation failed")
    
    # Method 2: Specific CPU index
    print("Trying PyTorch with specific CPU index...")
    run_command("pip uninstall torch -y")
    if run_command("pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir"):
        print("✓ PyTorch CPU index installation succeeded")
        if run_command("python3 -c \"import torch; print(f'PyTorch version: {torch.__version__}')\""):
            print("✓ PyTorch import succeeded")
            return True
    
    # Method 3: Older version
    print("Trying older PyTorch version...")
    run_command("pip uninstall torch -y")
    if run_command("pip install torch==1.13.0 --no-cache-dir"):
        print("✓ Older PyTorch installation succeeded")
        if run_command("python3 -c \"import torch; print(f'PyTorch version: {torch.__version__}')\""):
            print("✓ PyTorch import succeeded")
            return True
    
    print("✗ All PyTorch installation methods failed")
    return False

def main():
    print("=== PYTORCH INSTALLATION DEBUG ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Parse sleep argument
    sleep_time = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--sleep" and i + 1 < len(sys.argv):
            try:
                sleep_time = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid sleep value: {sys.argv[i + 1]}")
    
    # Run debug
    success = debug_environment()
    
    if success:
        print("\n✓ PyTorch debugging completed successfully!")
    else:
        print("\n✗ PyTorch debugging failed - unable to install PyTorch")
    
    # Sleep
    if sleep_time > 0:
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    
    print("Debug completed!")
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
