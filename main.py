import argparse
import time
import sys
import traceback
from datetime import datetime
from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.tf import tfop
from src.utils.tensorboard import dummy_tb_write

def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Ensure immediate output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Training on any operator')
    parser.add_argument("--operator", type=str, default="tf", choices=["ray", "jax", "tf"],
                       help="operator name")
    parser.add_argument("--sleep", type=int, default=0, 
                       help="sleep time in seconds after training")
    parser.add_argument("--tb-write", action="store_true", 
                       help="test write to tensorboard")
    parser.add_argument("--epochs", type=int, default=50,
                       help="number of training epochs (for supported operators)")
    
    args = parser.parse_args()
    
    log_with_timestamp(f"Starting training with {args.operator} operator")
    log_with_timestamp(f"Arguments: {vars(args)}")
    
    # Track training start time
    start_time = time.time()
    training_successful = False
    results = None
    
    try:
        if args.operator == "ray":
            log_with_timestamp("Initializing Ray operator...")
            results = rayop()
        elif args.operator == "jax":
            log_with_timestamp("Initializing JAX operator...")
            results = jaxop()
        elif args.operator == "tf":
            log_with_timestamp("Initializing TensorFlow operator...")
            results = tfop()
        
        training_successful = True
        training_time = time.time() - start_time
        log_with_timestamp(f"Training completed successfully in {training_time:.2f} seconds")
        
        if results:
            log_with_timestamp("Training results summary:")
            if isinstance(results, dict):
                for key, value in results.items():
                    log_with_timestamp(f"  {key}: {value}")
            else:
                log_with_timestamp(f"  Results: {results}")
        
    except Exception as e:
        training_time = time.time() - start_time
        log_with_timestamp(f"Training failed after {training_time:.2f} seconds")
        log_with_timestamp(f"Error: {str(e)}")
        log_with_timestamp("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    # Optional TensorBoard write test
    if args.tb_write:
        try:
            log_with_timestamp("Testing TensorBoard write...")
            dummy_tb_write()
            log_with_timestamp("TensorBoard write test completed")
        except Exception as e:
            log_with_timestamp(f"TensorBoard write test failed: {e}")
    
    # Optional sleep (useful for debugging or keeping job alive)
    if args.sleep > 0:
        log_with_timestamp(f"Sleeping for {args.sleep}s before exiting...")
        time.sleep(args.sleep)
    
    # Final status
    total_time = time.time() - start_time
    log_with_timestamp(f"Script completed successfully in {total_time:.2f} seconds")
    log_with_timestamp("All operations finished - exiting cleanly")
