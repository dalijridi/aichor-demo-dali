#!/usr/bin/env python3

import sys
import os
import time
import json
from datetime import datetime
import logging

def simple_tf_training():
    """A more complex TensorFlow training script that trains a CNN on MNIST and benefits from a GPU."""
    logging.info("Starting CNN TensorFlow training on MNIST...")
    
    try:
        # Check if TensorFlow is available
        logging.info("Checking TensorFlow availability...")
        try:
            import tensorflow as tf
            logging.info(f"✓ TensorFlow version: {tf.__version__}")
        except ImportError:
            logging.info("TensorFlow not found, installing...")
            os.system("pip install tensorflow>=2.8.0")
            import tensorflow as tf
            logging.info(f"✓ TensorFlow installed, version: {tf.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info(f"✓ GPU(s) found: {len(gpus)}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logging.warning("✗ No GPU found. This script will run on CPU, which may be slow.")

        try:
            from google.cloud import storage
            logging.info("✓ Google Cloud Storage client available")
        except ImportError:
            logging.info("Installing Google Cloud Storage client...")
            os.system("pip install google-cloud-storage")
            from google.cloud import storage
            logging.info("✓ Google Cloud Storage client installed")
        
        # Load and preprocess the MNIST dataset
        logging.info("Loading and preprocessing MNIST dataset...")
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add a channels dimension for the CNN
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")
        logging.info(f"Dataset loaded: train shape {x_train.shape}, test shape {x_test.shape}")
        
        # Create a simple CNN model
        logging.info("Creating CNN model...")
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
        model.summary() # This prints to stdout, which the logger will capture
        logging.info("Model compiled successfully")
        
        # Train the model
        logging.info("Starting training for 5 epochs...")
        history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
        
        # Evaluate the model
        logging.info("Evaluating model...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        logging.info(f"Test accuracy: {test_acc:.4f}")
        
        # Save model and outputs
        logging.info("Saving model and outputs...")
        
        model_dir = os.environ.get('AIP_MODEL_DIR', '/tmp/model')
        job_id = os.environ.get('CLOUD_ML_JOB_ID', 'local-training')
        
        logging.info(f"Output directory: {model_dir}")
        logging.info(f"Training job ID: {job_id}")
        
        os.makedirs('/tmp/outputs', exist_ok=True)
        os.makedirs('/tmp/model', exist_ok=True)
        
        local_model_path = '/tmp/model/saved_model'
        model.save(local_model_path, save_format='tf')
        logging.info(f"Model saved locally to: {local_model_path}")
        
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
        logging.info(f"Results saved to: {results_path}")
        
        if model_dir.startswith('gs://'):
            upload_to_gcs(model_dir, local_model_path, results_path, None)
        else:
            logging.info("Output directory is not a GCS path, files saved locally only")

        # Add a sleep period to ensure the job runs for more than 15 minutes.
        sleep_duration = 180
        logging.info(f"Training finished. Sleeping for {sleep_duration} seconds to extend job duration...")
        time.sleep(sleep_duration)

        logging.info("✓ TensorFlow training completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"✗ Training error: {e}", exc_info=True)
        return {'status': 'training_error', 'error': str(e)}

def upload_to_gcs(output_dir, local_model_path, results_path, weights_path):
    """Upload model and results to Google Cloud Storage"""
    try:
        from google.cloud import storage
        
        logging.info(f"Uploading to GCS output directory: {output_dir}")
        
        bucket_name = output_dir.replace('gs://', '').split('/')[0]
        blob_prefix = '/'.join(output_dir.replace('gs://', '').split('/')[1:])
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        logging.info("Uploading SavedModel...")
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_model_path)
                blob_name = f"{blob_prefix}/saved_model/{relative_path}".replace('\\', '/')
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
        
        logging.info("Uploading training results...")
        results_blob_name = f"{blob_prefix}/training_results.json"
        results_blob = bucket.blob(results_blob_name)
        results_blob.upload_from_filename(results_path)
        logging.info(f"Uploaded: {results_blob_name}")
        
        if weights_path:
            logging.info("Uploading model weights...")
            weights_blob_name = f"{blob_prefix}/model_weights.h5"
            weights_blob = bucket.blob(weights_blob_name)
            weights_blob.upload_from_filename(weights_path)
            logging.info(f"Uploaded: {weights_blob_name}")
        
        logging.info("✓ All files uploaded to GCS successfully!")
        
    except Exception as e:
        logging.error(f"✗ GCS upload error: {e}", exc_info=True)

def main():
    # --- FIX START: Configure root logger ---
    # This ensures all log messages, including from libraries like TensorFlow,
    # are sent to standard output where Cloud Logging can capture them.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    # --- FIX END ---

    logging.info("=== VERTEX AI TRAINING SCRIPT ===")
    
    # Parse simple arguments manually
    operator = "tf"
    for i, arg in enumerate(sys.argv):
        if arg == "--operator" and i + 1 < len(sys.argv):
            operator = sys.argv[i + 1]
    
    logging.info(f"\nUsing operator: {operator}")
    
    if operator == "tf":
        result = simple_tf_training()
        logging.info(f"\nTraining result: {result}")
    else:
        logging.info(f"Operator '{operator}' not implemented.")
        result = {'status': 'not_implemented', 'operator': operator}
    
    logging.info("Script completed successfully!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        # Use logging to capture the final fatal error
        logging.critical(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)
