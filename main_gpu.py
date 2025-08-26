import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import os

print("--- Starting TensorFlow Training Script ---")
# Ensure logs are sent immediately without buffering
os.environ['PYTHONUNBUFFERED'] = '1'

# --- 1. Check for GPU availability ---
print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ GPU is available. Using {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("⚠️ GPU not found. Using CPU.")

# --- 2. Simple Model Definition using Keras ---
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# --- 3. Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
args = parser.parse_args()

print(f"Training for {args.epochs} epochs with a learning rate of {args.lr}.")

# --- 4. Instantiate Model ---
model = create_model()
print("\nModel architecture:")
model.summary()

# --- 5. Create Dummy Data ---
X_train = np.random.rand(100, 20).astype(np.float32)
y_train = np.random.rand(100, 1).astype(np.float32)
print(f"\nCreated training data with shape: {X_train.shape}")

# --- 6. Define Loss and Optimizer ---
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# --- 7. Simple Training Loop using model.fit() ---
print("\n--- Starting Training Loop ---")
for epoch in range(args.epochs):
    print(f"\nEpoch [{epoch+1}/{args.epochs}]")
    # Keras's model.fit handles the training loop, loss calculation, and backpropagation
    model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=2)
    time.sleep(2) # Adding a small delay to make logs easier to follow

print("\n--- Training Complete --- ✅")
