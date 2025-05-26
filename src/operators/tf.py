# src/operators/tf.py
import sys
import os

def tfop():
    """Simple TensorFlow training that generates real output data"""
    print("Starting TensorFlow training...")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Check and import required packages
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"Failed to import TensorFlow: {e}")
        print("Installing TensorFlow...")
        os.system("pip install tensorflow")
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"Failed to import NumPy: {e}")
        print("Installing NumPy...")
        os.system("pip install numpy")
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    
    try:
        import json
        from datetime import datetime
    except ImportError as e:
        print(f"Failed to import standard library modules: {e}")
        return {"error": "Missing standard library modules"}
    
    # Create synthetic dataset (simple regression problem)
    print("Generating synthetic dataset...")
    np.random.seed(42)
    X = np.random.randn(1000, 1).astype(np.float32)
    y = 2 * X + 1 + 0.1 * np.random.randn(1000, 1).astype(np.float32)  # y = 2x + 1 + noise
    
    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create simple linear model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,), name='linear_layer')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=['mae']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Setup callbacks for logging
    callbacks = []
    
    # Try to setup TensorBoard logging if possible
    try:
        log_dir = "/tmp/tensorboard_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    except Exception as e:
        print(f"Could not setup TensorBoard logging: {e}")
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Get model weights (should be close to [2, 1] for our synthetic data)
    weights = model.get_weights()
    print(f"Learned weight: {weights[0][0][0]:.4f} (target: 2.0)")
    print(f"Learned bias: {weights[1][0]:.4f} (target: 1.0)")
    
    # Make some predictions
    print("Making sample predictions...")
    sample_X = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    predictions = model.predict(sample_X, verbose=0)
    for i, (x, pred) in enumerate(zip(sample_X, predictions)):
        expected = 2 * x[0] + 1
        print(f"Input: {x[0]:.1f}, Predicted: {pred[0]:.4f}, Expected: {expected:.4f}")
    
    # Save training results
    results = {
        'training_completed': True,
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'learned_weight': float(weights[0][0][0]),
        'learned_bias': float(weights[1][0]),
        'epochs_trained': len(history.history['loss']),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to file
    try:
        output_dir = "/tmp/training_output"
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Training results saved to: {results_file}")
        
        # Save model
        model_path = os.path.join(output_dir, "trained_model")
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Save training history
        history_file = os.path.join(output_dir, "training_history.json")
        with open(history_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
            json.dump(history_dict, f, indent=2)
        print(f"Training history saved to: {history_file}")
        
    except Exception as e:
        print(f"Could not save results to file: {e}")
        print("Results in memory:")
        print(json.dumps(results, indent=2))
    
    print("TensorFlow training completed successfully!")
    return results
