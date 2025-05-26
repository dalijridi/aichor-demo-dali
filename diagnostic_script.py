#!/usr/bin/env python3

# Ultra-minimal diagnostic script - no imports except built-ins
import sys
import os

def main():
    print("=== BASIC DIAGNOSTIC START ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script arguments: {sys.argv}")
    
    # Check if we can import basic modules
    print("\nTesting basic imports...")
    
    try:
        import json
        print("✓ json module: OK")
    except Exception as e:
        print(f"✗ json module: {e}")
    
    try:
        import time
        print("✓ time module: OK")
    except Exception as e:
        print(f"✗ time module: {e}")
    
    try:
        from datetime import datetime
        print("✓ datetime module: OK")
        print(f"Current time: {datetime.now()}")
    except Exception as e:
        print(f"✗ datetime module: {e}")
    
    # Check directory structure
    print(f"\nDirectory contents:")
    try:
        for item in os.listdir('.'):
            print(f"  {item}")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Check if src directory exists
    print(f"\nChecking src directory...")
    if os.path.exists('src'):
        print("✓ src directory exists")
        try:
            for item in os.listdir('src'):
                print(f"  src/{item}")
        except Exception as e:
            print(f"Error listing src directory: {e}")
    else:
        print("✗ src directory not found")
    
    # Test file operations
    print(f"\nTesting file operations...")
    try:
        test_file = "test_write.txt"
        with open(test_file, 'w') as f:
            f.write("test content")
        with open(test_file, 'r') as f:
            content = f.read()
        os.remove(test_file)
        print("✓ File write/read: OK")
    except Exception as e:
        print(f"✗ File operations: {e}")
    
    # Test simple computation
    print(f"\nTesting basic computation...")
    try:
        result = sum(range(100))
        print(f"✓ Sum of 0-99: {result}")
    except Exception as e:
        print(f"✗ Basic computation: {e}")
    
    print("\n=== BASIC DIAGNOSTIC END ===")
    print("If you see this message, basic Python execution works!")
    
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
