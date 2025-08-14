#!/usr/bin/env python3

import sys
import os
import time
import json
from datetime import datetime
import logging

# Configure logging and output for Vertex AI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

# Force all output to stdout and flush immediately
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Ensure line buffering for real-time output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Redirect stderr to stdout to capture all TensorFlow messages
sys.stderr = sys.stdout

def log_and_print(message):
    """Print message with immediate flush for Vertex AI visibility"""
    print(f"[TRAINING LOG] {message}", flush=True)
    logging.info(message)

def check_gpu_availability():
    """Check if GPU is available and configure TensorFlow to use it"""
    try:
        import tensorflow as tf
        
        log_and_print("=== GPU AVAILABILITY CHECK ===")
        log_and_print(f"TensorFlow version: {tf.__version__}")
        
        # List physical devices
        gpus = tf.config.list_physical_devices('GPU')
        log_and_print(f"Number of GPUs detected: {len(gpus)}")
        
        if len(gpus) == 0:
            log_and_print("❌ NO GPU DETECTED! This script requires a GPU machine.")
            log_and_print("Please ensure you're running on a GPU-enabled machine type.")
            sys.exit(1)
        
        # Print GPU details
        for i, gpu in enumerate(gpus):
            log_and_print(f"GPU {i}: {gpu}")
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                log_and_print(f"  Device details: {gpu_details}")
            except:
                log_and_print("  Device details not available")
        
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log_and_print("✓ GPU memory growth enabled")
        except RuntimeError as e:
            log_and_print(f"Memory growth setting failed (this is OK if already initialized): {e}")
        
        # Test GPU computation
        log_and_print("=== GPU COMPUTATION TEST ===")
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            log_and_print(f"GPU test computation successful: {result.numpy()}")
        
        log_and_print("✓ GPU is available and working correctly!")
        return True
        
    except Exception as e:
        log_and_print(f"❌ GPU check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def simple_tf_training():
    """GPU-accelerated TensorFlow training with detailed logging"""
    print("Starting GPU-accelerated TensorFlow training...")

    try:
        # Ensure TensorFlow
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('INFO')
            print(f"✓ TensorFlow version: {tf.__version__}")
        except ImportError:
            print("TensorFlow not found, installing GPU version...")
            os.system("pip install tensorflow[and-cuda]>=2.12.0")
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

        # Check GPU availability
        check_gpu_availability()

        print("\n=== CREATING GPU-OPTIMIZED MODEL ===")
        
        # Create more complex data to better utilize GPU
        np.random.seed(42)
        n_samples = 10000
        n_features = 100
        
        # Generate synthetic data that benefits from GPU acceleration
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        # Create a more complex relationship
        weights_true = np.random.randn(n_features, 1).astype(np.float32)
        y = X @ weights_true + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)
        
        print(f"Training data shape: X={X.shape}, y={y.shape}")

        # Create GPU-optimized model with more parameters
        with tf.device('/GPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu', input_shape=(n_features,), name='hidden1'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256, activation='relu', name='hidden2'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation='relu', name='hidden3'),
                tf.keras.layers.Dense(1, name='output')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )
        
        print(f"✓ Model created on GPU with {model.count_params():,} parameters")
        print("\nModel architecture:")
        model.summary()

        # Enhanced callback for detailed epoch logging
        class DetailedVertexLoggingCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_start_time = None
                self.batch_count = 0
            
            def on_train_begin(self, logs=None):
                log_and_print("🚀 TRAINING STARTED!")
                log_and_print(f"Total batches per epoch: {self.params['steps']}")
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                self.batch_count = 0
                log_and_print(f"📈 EPOCH {epoch+1}/{self.params['epochs']} STARTED")
            
            def on_batch_begin(self, batch, logs=None):
                self.batch_count += 1
            
            def on_batch_end(self, batch, logs=None):
                # Log progress every 20 batches
                if batch > 0 and batch % 20 == 0:
                    logs = logs or {}
                    batch_loss = logs.get('loss', 0)
                    log_and_print(f"  ⏳ Batch {batch+1}/{self.params['steps']}: loss={batch_loss:.6f}")
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                epoch_time = time.time() - self.epoch_start_time
                
                # Create detailed metrics summary
                metrics_parts = []
                val_metrics_parts = []
                
                for metric, value in logs.items():
                    if metric.startswith('val_'):
                        val_metrics_parts.append(f"{metric}: {value:.6f}")
                    else:
                        metrics_parts.append(f"{metric}: {value:.6f}")
                
                log_and_print(f"✅ EPOCH {epoch+1} COMPLETED in {epoch_time:.2f}s")
                log_and_print(f"   Training metrics: {' | '.join(metrics_parts)}")
                if val_metrics_parts:
                    log_and_print(f"   Validation metrics: {' | '.join(val_metrics_parts)}")
                log_and_print(f"   Processed {self.batch_count} batches")
                log_and_print("-" * 80)
            
            def on_train_end(self, logs=None):
                log_and_print("🎯 TRAINING COMPLETED!")
        
        # Add progress tracking callback
        class BatchProgressCallback(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                if batch % 50 == 0:  # Every 50 batches
                    logs = logs or {}
                    log_and_print(f"    Batch {batch}: loss={logs.get('loss', 0):.4f}")
                    sys.stdout.flush()  # Force flush

        # Split data for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\nTraining set: {X_train.shape}, Validation set: {X_val.shape}")

        # Train with GPU
        log_and_print("=== STARTING GPU TRAINING WITH DETAILED LOGGING ===")
        log_and_print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
        start_time = time.time()
        
        # Create callbacks with verbose logging
        callbacks = [
            DetailedVertexLoggingCallback(),
            BatchProgressCallback(),
            tf.keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True, 
                verbose=1,
                monitor='val_loss'
            )
        ]
        
        with tf.device('/GPU:0'):
            # Force immediate output by setting verbose=1 and using custom callbacks
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=128,  # Smaller batch size for more frequent updates
                verbose=1,  # Enable Keras verbose output
                callbacks=callbacks
            )
        
        total_time = time.time() - start_time
        print(f"\n✓ Training completed in {total_time:.2f} seconds")

        # Evaluation and prediction test
        print("\n=== MODEL EVALUATION ===")
        test_loss, test_mae, test_mse = model.evaluate(X_val, y_val, verbose=0)
        print(f"Final validation metrics - Loss: {test_loss:.6f}, MAE: {test_mae:.6f}, MSE: {test_mse:.6f}")

        # Prediction test
        test_input = X_val[:5]  # Use actual validation data
        with tf.device('/GPU:0'):
            predictions = model.predict(test_input, verbose=0)
        
        print("Sample predictions vs actual:")
        for i in range(5):
            print(f"  Sample {i+1}: Predicted={predictions[i][0]:.4f}, Actual={y_val[i][0]:.4f}")

        # Save model/results
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
            'gpu_training': True,
            'total_training_time_seconds': total_time,
            'epochs_completed': len(history.history['loss']),
            'final_metrics': {
                'loss': float(final_loss),
                'val_loss': float(history.history.get('val_loss', [-1])[-1]),
                'mae': float(history.history.get('mae', [-1])[-1]),
                'val_mae': float(history.history.get('val_mae', [-1])[-1])
            },
            'model_params': model.count_params(),
            'training_data_shape': {
                'samples': int(n_samples),
                'features': int(n_features)
            },
            'training_history': {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                'mae': [float(x) for x in history.history.get('mae', [])],
                'val_mae': [float(x) for x in history.history.get('val_mae', [])]
            },
            'gpu_info': {
                'gpus_available': len(tf.config.list_physical_devices('GPU')),
                'tensorflow_version': tf.__version__
            },
            'training_timestamp': datetime.now().isoformat(),
            'job_id': job_id
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

        print("\n✅ GPU training completed successfully with detailed epoch logging!")
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
        
        print("✅ All files uploaded to GCS successfully!")
        
    except Exception as e:
        print(f"❌ GCS upload error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== VERTEX AI GPU TRAINING SCRIPT ===")
    print(f"Python version: {sys.version}")
    print(f"Arguments: {sys.argv}")
    print(f"Working directory: {os.getcwd()}")
    
    # Print relevant environment variables
    print("\n=== Environment Variables ===")
    env_vars = ['AIP_MODEL_DIR', 'AIP_CHECKPOINT_DIR', 'AIP_TENSORBOARD_LOG_DIR', 
                'CLOUD_ML_PROJECT_ID', 'CLOUD_ML_JOB_ID', 'CUDA_VISIBLE_DEVICES']
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
        print(f"\nFinal training result: {result}")
    else:
        print(f"Operator '{operator}' not implemented in minimal version")
        result = {'status': 'not_implemented', 'operator': operator}
    
    # Optional sleep
    if sleep_time > 0:
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)
    
    print("🎉 Script completed successfully!")
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
