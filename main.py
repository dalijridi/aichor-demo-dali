#!/usr/bin/env python3

import sys
import os
import time
import json
from datetime import datetime

def simple_tf_training():
    """Minimal TensorFlow training with GCS output support, extended to run longer."""
    print("Starting simple TensorFlow training...")
    
    try:
        # Check if TensorFlow is available, install if missing
        print("Checking TensorFlow availability...")
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow version: {tf.__version__}")
        except ImportError:
            print("TensorFlow not found, installing...")
            os.system("pip install tensorflow>=2.8.0")
            import tensorflow as tf
            print(f"✓ TensorFlow installed, version: {tf.__version__}")
        
        try:
            import numpy as np
            print(f"✓ NumPy version: {np.__version__}")
        except ImportError:
            print("NumPy not found, installing...")
            os.system("pip install numpy")
            import numpy as np
            print(f"✓ NumPy installed, version: {np.__version__}")
        
        # Install Google Cloud Storage client
        try:
            from google.cloud import storage
            print("✓ Google Cloud Storage client available")
        except ImportError:
            print("Installing Google Cloud Storage client...")
            os.system("pip install google-cloud-storage")
            from google.cloud import storage
            print("✓ Google Cloud Storage client installed")
        
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
        
        # Train for more epochs to extend training time
        print("Starting training for 300 epochs...")
        history = model.fit(X, y, epochs=300, verbose=1, batch_size=2)
        
        # Test prediction
        print("Testing prediction...")
        test_input = np.array([[5.0]], dtype=np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"Input: 5.0, Prediction: {prediction[0][0]:.2f}, Expected: ~10.0")
        
        # Get final loss
        final_loss = history.history['loss'][-1]
        print(f"Final training loss: {final_loss:.4f}")
        
        # Save model and outputs
        print("Saving model and outputs...")
        
        # Get output directory from environment variables (Vertex AI standard)
        model_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
        
        # Get additional info for logging
        job_id = os.environ.get('CLOUD_ML_JOB_ID', 'local-training')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"Output directory (includes pipeline name): {model_dir}")
        print(f"Training job ID: {job_id}")
        print(f"Training timestamp: {timestamp}")
        
        # Create local directories
        os.makedirs('/tmp/outputs', exist_ok=True)
        os.makedirs('/tmp/model', exist_ok=True)
        
        # Save the model locally first
        local_model_path = '/tmp/model/saved_model'
        model.save(local_model_path, save_format='tf')
        print(f"Model saved locally to: {local_model_path}")
        
        # Create training results
        results = {
            'status': 'success',
            'final_loss': float(final_loss),
            'prediction_test': float(prediction[0][0]),
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'mae': [float(x) for x in history.history['mae']]
            },
            'model_architecture': model.to_json(),
            'training_timestamp': datetime.now().isoformat(),
            'job_id': job_id,
            'tensorflow_version': tf.__version__,
            'numpy_version': np.__version__
        }
        
        # Save results to JSON
        results_path = '/tmp/outputs/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Save model weights separately
        weights_path = '/tmp/outputs/model_weights.h5'
        model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")
        
        # Upload to GCS if model directory is a GCS path
        if model_dir.startswith('gs://'):
            upload_to_gcs(model_dir, local_model_path, results_path, weights_path)
        else:
            print(f"Output directory is not a GCS path, files saved locally only")
            # Copy to the specified model directory if it's local
            import shutil
            if model_dir != '/tmp/model':
                os.makedirs(model_dir, exist_ok=True)
                shutil.copytree(local_model_path, os.path.join(model_dir, 'saved_model'), dirs_exist_ok=True)
                shutil.copy2(results_path, model_dir)
                shutil.copy2(weights_path, model_dir)
                print(f"Files copied to output directory: {model_dir}")

        # --- MODIFICATION START ---
        # Add a sleep period to ensure the job runs for more than 15 minutes.
        sleep_duration = 1500  # 25 minutes
        print(f"Training finished. Sleeping for {sleep_duration} seconds to extend job duration...")
        time.sleep(sleep_duration)
        # --- MODIFICATION END ---

        print("✓ TensorFlow training completed successfully!")
        return results
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Required packages not available")
        return {'status': 'import_error', 'error': str(e)}
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'training_error', 'error': str(e)}

def upload_to_gcs(output_dir, local_model_path, results_path, weights_path):
    """Upload model and results to Google Cloud Storage"""
    try:
        from google.cloud import storage
        
        print(f"Uploading to GCS output directory: {output_dir}")
        
        # Parse GCS path
        bucket_name = output_dir.replace('gs://', '').split('/')[0]
        blob_prefix = '/'.join(output_dir.replace('gs://', '').split('/')[1:])
        
        print(f"Bucket: {bucket_name}")
        print(f"Prefix: {blob_prefix}")
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Upload SavedModel directory
        print("Uploading SavedModel...")
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_model_path)
                blob_name = f"{blob_prefix}/saved_model/{relative_path}".replace('\\', '/')
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
                print(f"Uploaded: {blob_name}")
        
        # Upload training results
        print("Uploading training results...")
        results_blob_name = f"{blob_prefix}/training_results.json"
        results_blob = bucket.blob(results_blob_name)
        results_blob.upload_from_filename(results_path)
        print(f"Uploaded: {results_blob_name}")
        
        # Upload model weights
        print("Uploading model weights...")
        weights_blob_name = f"{blob_prefix}/model_weights.h5"
        weights_blob = bucket.blob(weights_blob_name)
        weights_blob.upload_from_filename(weights_path)
        print(f"Uploaded: {weights_blob_name}")
        
        print("✓ All files uploaded to GCS successfully!")
        
    except Exception as e:
        print(f"✗ GCS upload error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== VERTEX AI TRAINING SCRIPT ===")
    print(f"Python version: {sys.version}")
    print(f"Arguments: {sys.argv}")
    print(f"Working directory: {os.getcwd()}")
    
    # Print relevant environment variables
    print("\n=== Environment Variables ===")
    env_vars = ['AIP_MODEL_DIR', 'AIP_CHECKPOINT_DIR', 'AIP_TENSORBOARD_LOG_DIR',    
                'CLOUD_ML_PROJECT_ID', 'CLOUD_ML_JOB_ID']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Parse simple arguments manually
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
    
    print(f"\nUsing operator: {operator}")
    print(f"Sleep time: {sleep_time}")
    
    # Run training based on operator
    if operator == "tf":
        result = simple_tf_training()
        print(f"\nTraining result: {result}")
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
