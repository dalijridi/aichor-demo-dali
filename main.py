#!/usr/bin/env python3

import sys
import os
import time
import json
from datetime import datetime
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# Force Python logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def simple_tf_training():
    """Minimal TensorFlow training with GCS output support and live Vertex logs"""
    print("Starting simple TensorFlow training...")

    try:
        # Ensure TensorFlow
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('INFO')
            print(f"✓ TensorFlow version: {tf.__version__}")
        except ImportError:
            print("TensorFlow not found, installing...")
            os.system("pip install tensorflow>=2.8.0")
            import tensorflow as tf
            print(f"✓ TensorFlow installed, version: {tf.__version__}")

        # Ensure NumPy
        try:
            import numpy as np
            print(f"✓ NumPy version: {np.__version__}")
        except ImportError:
            print("NumPy not found, installing...")
            os.system("pip install numpy")
            import numpy as np
            print(f"✓ NumPy installed, version: {np.__version__}")

        # Ensure GCS client
        try:
            from google.cloud import storage
            print("✓ Google Cloud Storage client available")
        except ImportError:
            print("Installing Google Cloud Storage client...")
            os.system("pip install google-cloud-storage")
            from google.cloud import storage
            print("✓ Google Cloud Storage client installed")

        # Data
        X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y = np.array([[2.0], [4.0], [6.0], [8.0]], dtype=np.float32)  # y = 2*x

        # Model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,), name='simple_dense')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Custom callback for live logs
        class VertexLoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                msg = f"Epoch {epoch+1} - " + ", ".join(f"{k}: {v:.4f}" for k, v in logs.items())
                print(msg, flush=True)

        # Train
        print("Starting training with live epoch logging...")
        history = model.fit(
            X, y,
            epochs=10,
            batch_size=2,
            verbose=0,  # disable Keras' internal progress output
            callbacks=[VertexLoggingCallback()]
        )

        # Prediction test
        test_input = np.array([[5.0]], dtype=np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"Test input 5.0 → Prediction: {prediction[0][0]:.2f} (expected ~10.0)")

        # Save model/results as before...
        final_loss = history.history['loss'][-1]
        model_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
        job_id = os.environ.get('CLOUD_ML_JOB_ID', 'local-training')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        os.makedirs('/tmp/outputs', exist_ok=True)
        os.makedirs('/tmp/model', exist_ok=True)
        local_model_path = '/tmp/model/saved_model'
        model.save(local_model_path, save_format='tf')

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

        results_path = '/tmp/outputs/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        weights_path = '/tmp/outputs/model_weights.h5'
        model.save_weights(weights_path)

        if model_dir.startswith('gs://'):
            upload_to_gcs(model_dir, local_model_path, results_path, weights_path)
        else:
            import shutil
            if model_dir != '/tmp/model':
                os.makedirs(model_dir, exist_ok=True)
                shutil.copytree(local_model_path, os.path.join(model_dir, 'saved_model'), dirs_exist_ok=True)
                shutil.copy2(results_path, model_dir)
                shutil.copy2(weights_path, model_dir)

        print("✓ Training completed successfully with live logs!")
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'training_error', 'error': str(e)}

def upload_to_gcs(output_dir, local_model_path, results_path, weights_path):
    """Upload model and results to Google Cloud Storage"""
    try:
        from google.cloud import storage
        
        print(f"Uploading to GCS output directory: {output_dir}")
        
        # Parse GCS path
        # output_dir format: gs://vertex-trainings-outputs/vertexai/pipeline_name
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
                # Get relative path from the local model directory
                relative_path = os.path.relpath(local_file_path, local_model_path)
                # Clean up the blob name construction
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
