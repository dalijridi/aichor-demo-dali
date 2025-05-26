import argparse
import sys
import os
import traceback
from datetime import datetime

def log_info(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def diagnostic_check():
    """Run basic diagnostic checks"""
    log_info("=== DIAGNOSTIC MODE ===")
    log_info(f"Python version: {sys.version}")
    log_info(f"Platform: {sys.platform}")
    log_info(f"Current working directory: {os.getcwd()}")
    log_info(f"Python executable: {sys.executable}")
    
    # Check environment variables
    log_info("Key environment variables:")
    for var in ['PATH', 'PYTHONPATH', 'HOME', 'USER']:
        value = os.environ.get(var, 'NOT_SET')
        log_info(f"  {var}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    # Check available packages
    log_info("Checking package availability...")
    packages_to_check = ['tensorflow', 'numpy', 'json', 'time', 'argparse']
    
    for package in packages_to_check:
        try:
            if package == 'json':
                import json
                log_info(f"  ✓ {package}: Available (built-in)")
            elif package == 'time':
                import time
                log_info(f"  ✓ {package}: Available (built-in)")
            elif package == 'argparse':
                import argparse
                log_info(f"  ✓ {package}: Available (built-in)")
            elif package == 'numpy':
                import numpy as np
                log_info(f"  ✓ {package}: {np.__version__}")
            elif package == 'tensorflow':
                import tensorflow as tf
                log_info(f"  ✓ {package}: {tf.__version__}")
        except ImportError as e:
            log_info(f"  ✗ {package}: NOT AVAILABLE - {e}")
        except Exception as e:
            log_info(f"  ? {package}: ERROR - {e}")
    
    # Check file system
    log_info("Checking file system...")
    try:
        test_dir = "/tmp/test_write"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        with open(test_file, 'r') as f:
            content = f.read()
        os.remove(test_file)
        os.rmdir(test_dir)
        log_info("  ✓ File system: Read/write operations work")
    except Exception as e:
        log_info(f"  ✗ File system: Error - {e}")
    
    log_info("=== DIAGNOSTIC COMPLETE ===")
    return True

def simple_training():
    """Minimal training without external dependencies"""
    log_info("=== SIMPLE TRAINING MODE ===")
    
    try:
        # Try to do minimal TensorFlow training
        import tensorflow as tf
        import numpy as np
        
        log_info("Creating simple dataset...")
        # Very simple dataset
        X = np.array([[1.], [2.], [3.], [4.]], dtype=np.float32)
        y = np.array([[2.], [4.], [6.], [8.]], dtype=np.float32)  # y = 2x
        
        log_info("Creating model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        log_info("Training model...")
        history = model.fit(X, y, epochs=10, verbose=1)
        
        log_info("Making prediction...")
        pred = model.predict(np.array([[5.]], dtype=np.float32))
        log_info(f"Prediction for input 5: {pred[0][0]:.2f} (expected ~10)")
        
        log_info("✓ Simple training completed successfully!")
        return {"status": "success", "final_loss": float(history.history['loss'][-1])}
        
    except Exception as e:
        log_info(f"✗ Simple training failed: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Diagnostic training script')
    parser.add_argument("--mode", type=str, default="both", 
                       choices=["diagnostic", "training", "both"],
                       help="Mode to run")
    
    args = parser.parse_args()
    
    log_info("Starting diagnostic script...")
    log_info(f"Mode: {args.mode}")
    
    try:
        if args.mode in ["diagnostic", "both"]:
            diagnostic_check()
        
        if args.mode in ["training", "both"]:
            result = simple_training()
            log_info(f"Training result: {result}")
        
        log_info("Script completed successfully!")
        
    except Exception as e:
        log_info(f"Script failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
