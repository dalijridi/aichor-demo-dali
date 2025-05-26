#!/usr/bin/env python3

import sys
import os
import time

def simple_tf_training():
    """Minimal TensorFlow training without external file imports"""
    print("Starting simple TensorFlow training...")
    
    try:
        # Check if TensorFlow is available
        print("Checking TensorFlow availability...")
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        # Create very simple data
        print("Creating simple dataset...")
        X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)  # y = 2*x
        print(f"Dataset created: X shape {X.shape}, y shape {y.shape}")
        
        # Create simple model
        print("Creating model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,), name='simple_dense')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model compiled successfully")
        
        # Train for just a few epochs
        print("Starting training...")
        history = model.fit(X, y, epochs=5, verbose=1, batch_size=2)
        
        # Test prediction
        print("Testing prediction...")
        test_input = np.array([[5.0]], dtype=np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"Input: 5.0, Prediction: {prediction[0][0]:.2f}, Expected: ~10.0")
        
        # Get final loss
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss:.4f}")
        
        print("✓ TensorFlow training completed successfully!")
        return {
            'status': 'success',
            'final_loss': float(final_loss),
            'prediction_test': float(prediction[0][0])
        }
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("TensorFlow or NumPy not available")
        return {'status': 'import_error', 'error': str(e)}
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'training_error', 'error': str(e)}

def main():
    print("=== MINIMAL TRAINING SCRIPT ===")
    print(f"Python version: {sys.version}")
    print(f"Arguments: {sys.argv}")
    print(f"Working directory: {os.getcwd()}")
    
    # Parse simple arguments manually (avoid argparse issues)
    sleep_time = 0
    operator = "tf"
    
    for i, arg in enumerate(sys.argv):
        if arg == "--sleep" and i + 1 < len(sys.argv):
            try:
                sleep_time = int(sys.argv[i + 1])
            except ValueError:
                print(f"Invalid sleep value: {sys.argv[i + 1]}")
        if arg == "--operator" and i + 1 < len(sys.argv):
            operator = sys.argv[i + 1]
    
    print(f"Using operator: {operator}")
    print(f"Sleep time: {sleep_time}")
    
    # Run training based on operator
    if operator == "tf":
        result = simple_tf_training()
        print(f"Training result: {result}")
    else:
        print(f"Operator '{operator}' not implemented in minimal version")
        result = {'status': 'not_implemented', 'operator': operator}
    
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
