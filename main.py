#!/usr/bin/env python3

import sys
import os
import time
import json
from datetime import datetime

def simple_tf_training():
    """A more complex TensorFlow training script that trains a CNN on MNIST and benefits from a GPU."""
    print("Starting CNN TensorFlow training on MNIST...")
    
    try:
        # Check if TensorFlow is available
        print("Checking TensorFlow availability...")
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow version: {tf.__version__}")
        except ImportError:
            print("TensorFlow not found, installing...")
            os.system("pip install tensorflow>=2.8.0")
            import tensorflow as tf
            print(f"✓ TensorFlow installed, version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU(s) found: {len(gpus)}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("✗ No GPU found. This script will run on CPU, which may be slow.")

        try:
            from google.cloud import storage
            print("✓ Google Cloud Storage client available")
        except ImportError:
            print("Installing Google Cloud Storage client...")
            os.system("pip install google-cloud-storage")
            from google.cloud import storage
            print("✓ Google Cloud Storage client installed")
        
        # Load and preprocess the MNIST dataset
        print("Loading and preprocessing MNIST dataset...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add a channels dimension for the CNN
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")
        print(f"Dataset loaded: train shape {x_train.shape}, test shape {x_test.shape}")
        
        # Create a simple CNN model
        print("Creating CNN model...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        print("Model compiled successfully")
        
        # Train the model
        print("Starting training for 5 epochs...")
        history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
        
        # Evaluate the model
        print("Evaluating model...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save model and outputs
        print("Saving model and outputs...")
        
        model_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
        job_id = os.environ.get('CLOUD_ML_JOB_ID', 'local-training')
        
        print(f"Output directory: {model_dir}")
        print(f"Training job ID: {job_id}")
        
        os.makedirs('/tmp/outputs', exist_ok=True)
        os.makedirs('/tmp/model', exist_ok=True)
        
        local_model_path = '/tmp/model/saved_model'
        model.save(local_model_path, save_format='tf')
        print(f"Model saved locally to: {local_model_path}")
        
        results = {
            'status': 'success',
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'training_history': {k: [float(v) for v in val] for k, val in history.history.items()},
            'job_id': job_id,
            'tensorflow_version': tf.__version__,
        }
        
        results_path = '/tmp/outputs/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        if model_dir.startswith('gs://'):
            upload_to_gcs(model_dir, local_model_path, results_path, None)
        else:
            print("Output directory is not a GCS path, files saved locally only")

        # Add a sleep period to ensure the job runs for more than 15 minutes.
        sleep_duration = 960  # 16 minutes
        print(f"Training finished. Sleeping for {sleep_duration} seconds to extend job duration...")
        time.sleep(sleep_duration)

        print("✓ TensorFlow training completed successfully!")
        return results
        
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
        
        bucket_name = output_dir.replace('gs://', '').split('/')[0]
        blob_prefix = '/'.join(output_dir.replace('gs://', '').split('/')[1:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        print("Uploading SavedModel...")
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_model_path)
                blob_name = f"{blob_prefix}/saved_model/{relative_path}".replace('\\', '/')
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
        
        print("Uploading training results...")
        results_blob_name = f"{blob_prefix}/training_results.json"
        results_blob = bucket.blob(results_blob_name)
        results_blob.upload_from_filename(results_path)
        print(f"Uploaded: {results_blob_name}")
        
        if weights_path:
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
    
    # Parse simple arguments manually
    operator = "tf"
    for i, arg in enumerate(sys.argv):
        if arg == "--operator" and i + 1 < len(sys.argv):
            operator = sys.argv[i + 1]
    
    print(f"\nUsing operator: {operator}")
    
    if operator == "tf":
        result = simple_tf_training()
        print(f"\nTraining result: {result}")
    else:
        print(f"Operator '{operator}' not implemented.")
        result = {'status': 'not_implemented', 'operator': operator}
    
    print("Script completed successfully!")
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
